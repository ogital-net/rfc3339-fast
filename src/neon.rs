use core::arch::aarch64::{
    uint16x8_t, uint32x4_t, uint8x16_t, uint8x16x2_t, vandq_u8, vcgtq_u8, vcltq_u8, vget_high_u8,
    vget_lane_u64, vget_low_u16, vget_low_u8, vld1q_u16, vld1q_u8, vmaxvq_u8, vmovl_u8,
    vmull_high_u16, vmull_u16, vmulq_u16, vpaddq_u32, vqaddq_u16, vqtbl2q_u8, vreinterpret_u64_u8,
    vreinterpretq_u16_u8, vshrn_n_u16, vst1q_u16, vst1q_u32, vsubq_u8, vuzp1q_u16, vuzp2q_u16,
};

use crate::TimestampError;

#[target_feature(enable = "neon")]
pub(super) unsafe fn decode_seconds(ascii: &mut &[u8]) -> Result<i64, TimestampError> {
    const LOWER_BOUND: [u8; 16] = *b"00000000000000T:";

    const UPPER_BOUND: [u8; 16] = [
        9, 9, 9, 9, // Year
        1, 9, // Month
        3, 9, // Day
        2, 9, // Hour
        5, 9, // Minute
        5, 9, // Second
        0, 0,
    ];

    const MULT_10: [u8; 16] = [10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0];

    // Gather table for `vqtbl2q_u8`. Indices 0..=15 select from `vec0` (bytes
    // 0..16 of input); indices 16..=31 select from `vec1` (bytes 3..19 of
    // input, loaded at offset 3 so that the high indices stay in-range).
    //   source byte 17 → vec1[14] → index 16+14 = 30
    //   source byte 18 → vec1[15] → index 16+15 = 31
    //   source byte 16 → vec1[13] → index 16+13 = 29
    const SHUFFLE: [u8; 16] = [
        0, 1, 2, 3, // Year
        5, 6, // Month
        8, 9, // Day
        11, 12, // Hour
        14, 15, // Minute
        30, 31, // Second  (source bytes 17,18)
        10, // 'T'
        29, // last colon (source byte 16)
    ];

    unsafe {
        // YYYY-MM-DDTHH:MM:SS
        if ascii.len() < 20 {
            return Err(TimestampError::InvalidFormat);
        }
        let vec0 = vld1q_u8(ascii.as_ptr());
        let vec1 = vld1q_u8(ascii.as_ptr().add(3));
        let table = uint8x16x2_t(vec0, vec1);
        let shuffle = vld1q_u8(SHUFFLE.as_ptr());
        let mut tmp = vqtbl2q_u8(table, shuffle);

        let lower_bound = vld1q_u8(LOWER_BOUND.as_ptr());
        let upper_bound = vld1q_u8(UPPER_BOUND.as_ptr());

        // value sanity check
        tmp = vsubq_u8(tmp, lower_bound);
        let higher = vcgtq_u8(tmp, upper_bound);

        if vmaxvq_u8(higher) != 0 {
            return Err(TimestampError::InvalidFormat);
        }

        let mult_10 = vld1q_u8(MULT_10.as_ptr());

        let tmp = maddubs_neon(tmp, mult_10);

        let mut res: [u8; 16] = [0u8; 16];
        vst1q_u16(res.as_mut_ptr().cast(), tmp);

        let year = res[0] as i32 * 100 + res[2] as i32;

        *ascii = &ascii[19..];

        Ok(crate::jsondec_unixtime(
            year,
            res[4] as i32,
            res[6] as i32,
            res[8] as i32,
            res[10] as i32,
            res[12] as i32,
        ))
    }
}

#[target_feature(enable = "neon")]
pub(super) unsafe fn decode_nanos(ascii: &mut &[u8]) -> Result<i32, TimestampError> {
    if ascii[0] != b'.' {
        return Ok(0);
    }

    const ASCII_ZERO: [u8; 16] = [b'0'; 16];
    const DIGIT_MAX: [u8; 16] = [10; 16];

    const THREE_DIGITS: u64 = 0x0000_0000_0000_FFF0;
    const SIX_DIGITS: u64 = 0x0000_0000_FFF0_FFF0;
    const NINE_DIGITS: u64 = 0x0000_FFF0_FFF0_FFF0;

    const MULT_100_10: [u8; 16] = [0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1];
    const MULT_1000: [u16; 8] = [1000, 1000, 1000, 1000, 1, 1, 0, 0];

    unsafe {
        let mut tmp = match ascii.len() {
            // .123Z
            5 => {
                let t: [u8; 16] = [
                    0, ascii[1], ascii[2], ascii[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                ];
                vld1q_u8(t.as_ptr())
            }
            // .123456Z
            8 => {
                let t: [u8; 16] = [
                    0, ascii[1], ascii[2], ascii[3], 0, ascii[4], ascii[5], ascii[6], 0, 0, 0, 0,
                    0, 0, 0, 0,
                ];
                vld1q_u8(t.as_ptr())
            }
            // .123456789Z
            11..=16 => {
                let t: [u8; 16] = [
                    0, ascii[1], ascii[2], ascii[3], 0, ascii[4], ascii[5], ascii[6], 0, ascii[7],
                    ascii[8], ascii[9], 0, 0, 0, 0,
                ];
                vld1q_u8(t.as_ptr())
            }
            _ => {
                return Err(TimestampError::InvalidFormat);
            }
        };

        // input validation
        let ascii_zero = vld1q_u8(ASCII_ZERO.as_ptr());
        let digit_max = vld1q_u8(DIGIT_MAX.as_ptr());
        tmp = vsubq_u8(tmp, ascii_zero);
        let valid_digits = vcltq_u8(tmp, digit_max);

        let mask = neon_nibblemask(valid_digits);

        // including the leading '.'
        let offset = if mask | NINE_DIGITS == mask {
            10
        } else if mask | SIX_DIGITS == mask {
            7
        } else if mask | THREE_DIGITS == mask {
            4
        } else {
            return Err(TimestampError::InvalidFormat);
        };

        tmp = vandq_u8(tmp, valid_digits);

        let tmp = maddubs_neon(tmp, vld1q_u8(MULT_100_10.as_ptr()));
        let tmp = madd_neon(tmp, vld1q_u16(MULT_1000.as_ptr()));

        let mut out: [i32; 4] = [0; 4];
        vst1q_u32(out.as_mut_ptr().cast(), tmp);

        *ascii = &ascii[offset..];

        Ok(out[0] * 1000 + out[1] + out[2])
    }
}

/// Multiplies corresponding pairs of packed 8-bit unsigned integer
/// values contained in the first source operand and packed 8-bit unsigned
/// integer values contained in the second source operand, add pairs of
/// contiguous products with unsigned saturation, and writes the 16-bit sums to
/// the corresponding bits in the destination.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn maddubs_neon(a: uint8x16_t, b: uint8x16_t) -> uint16x8_t {
    #[allow(unused_unsafe)]
    unsafe {
        let tl = vmulq_u16(vmovl_u8(vget_low_u8(a)), vmovl_u8(vget_low_u8(b)));
        let th = vmulq_u16(vmovl_u8(vget_high_u8(a)), vmovl_u8(vget_high_u8(b)));
        vqaddq_u16(vuzp1q_u16(tl, th), vuzp2q_u16(tl, th))
    }
}

/// Multiplies and then horizontally add signed 16 bit integers in `a` and `b`.
///
/// Multiplies packed signed 16-bit integers in `a` and `b`, producing
/// intermediate signed 32-bit integers. Horizontally add adjacent pairs of
/// intermediate 32-bit integers.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn madd_neon(a: uint16x8_t, b: uint16x8_t) -> uint32x4_t {
    #[allow(unused_unsafe)]
    unsafe {
        let low = vmull_u16(vget_low_u16(a), vget_low_u16(b));
        let high = vmull_high_u16(a, b);

        vpaddq_u32(low, high)
    }
}

/// Packs the most significant bit of each byte in `input` into a 64-bit
/// integer, using 4 bits per input byte (a "nibble mask").
///
/// This is the canonical AArch64 idiom for emulating x86 `pmovmskb`. It
/// uses a single `shrn #4` to narrow 16 bytes into 8 nibble-pairs, packed
/// into a `u64` (low nibble = even input byte, high nibble = odd input byte).
/// Each input byte is expected to be either `0xFF` or `0x00` (a compare
/// result); the corresponding output nibble is then `0xF` or `0x0`.
///
/// Significantly cheaper on AArch64 than the SSE2-style emulation using
/// `vshlq_u8` + `vaddv_u8` reductions.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn neon_nibblemask(input: uint8x16_t) -> u64 {
    #[allow(unused_unsafe)]
    unsafe {
        let narrowed = vshrn_n_u16::<4>(vreinterpretq_u16_u8(input));
        vget_lane_u64::<0>(vreinterpret_u64_u8(narrowed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_seconds() {
        let s = "2026-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert_eq!(unsafe { decode_seconds(input).unwrap() }, 1772029800);
        assert_eq!(input, b"Z");
    }

    #[test]
    fn test_decode_seconds_invalid_chars() {
        let s = "20/6-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_seconds(input).is_err() });

        let s = "20:6-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_seconds(input).is_err() });
    }

    #[test]
    fn test_decode_nanos() {
        let s = ".987654321Z";
        let input = &mut s.as_bytes();
        assert_eq!(unsafe { decode_nanos(input).unwrap() }, 987654321);
        assert_eq!(input, b"Z");

        let s = ".987654+00:00";
        let input = &mut s.as_bytes();
        assert_eq!(unsafe { decode_nanos(input).unwrap() }, 987654000);
        assert_eq!(input, b"+00:00");
    }

    #[test]
    fn test_decode_nanos_invalid_chars() {
        let s = ".98/654321Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_nanos(input).is_err() });

        let s = ".98:654321Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_nanos(input).is_err() });
    }

    /// Covers the catch-all length arm of the NEON `decode_nanos` (lengths
    /// other than 5, 8, or 11..=16 must return `InvalidFormat`).
    #[test]
    fn test_decode_nanos_invalid_length() {
        // Length 4: too short to be a 3-digit `.123Z`.
        let s = ".12Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_nanos(input).is_err() });

        // Length 7: between the 3- and 6-digit cases.
        let s = ".12345Z";
        let input = &mut s.as_bytes();
        assert!(unsafe { decode_nanos(input).is_err() });
    }
}
