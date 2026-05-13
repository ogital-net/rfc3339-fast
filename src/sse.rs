// SSSE3 backend. The clippy::pedantic lints disabled below are all
// fundamental to the way SIMD intrinsic code is written:
//
// * `cast_possible_wrap` / `cast_sign_loss`: shuffle/mask tables are
//   declared as `i8` because that's what `_mm_setr_epi8` takes, but the
//   sentinel value -128 is most naturally written as `-128i8` while real
//   indices are <128, so `u8 as i8` round-trips losslessly.
// * `ptr_as_ptr`, `borrow_as_ptr`, `ref_as_ptr`: `_mm_loadu_si128` /
//   `_mm_load_si128` / `_mm_store_si128` take `*const __m128i`, which can
//   only be obtained by an `as` cast from a `*const u8` / `&AlignedBuffer`.
// * `items_after_statements`: shuffle-table `const`s are placed next to
//   the intrinsic that consumes them, where they're easiest to read.
// * `unreadable_literal`: short SIMD constants like `2447` and `1461` are
//   reproduced verbatim from the F/VF literature; underscore-grouping
//   would just obscure the reference.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::ptr_as_ptr,
    clippy::borrow_as_ptr,
    clippy::ref_as_ptr,
    clippy::items_after_statements,
    clippy::unreadable_literal
)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    __m128i, _mm_and_si128, _mm_cmpgt_epi8, _mm_cmplt_epi8, _mm_load_si128, _mm_loadu_si128,
    _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
    _mm_setr_epi8, _mm_setr_epi16, _mm_shuffle_epi8, _mm_store_si128, _mm_sub_epi8,
};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    __m128i, _mm_and_si128, _mm_cmpgt_epi8, _mm_cmplt_epi8, _mm_load_si128, _mm_loadu_si128,
    _mm_madd_epi16, _mm_maddubs_epi16, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
    _mm_setr_epi8, _mm_setr_epi16, _mm_shuffle_epi8, _mm_store_si128, _mm_sub_epi8,
};

use crate::TimestampError;

#[repr(align(16))]
struct AlignedBuffer([u8; size_of::<__m128i>()]);

impl AlignedBuffer {
    fn new() -> Self {
        unsafe { core::mem::zeroed() }
    }

    fn as_ptr<T>(&self) -> *const T {
        &self.0 as *const _ as *const T
    }

    fn as_mut_ptr<T>(&mut self) -> *mut T {
        &mut self.0 as *mut _ as *mut T
    }

    fn as_bytes(&self) -> &[u8] {
        &self.0
    }

    fn as_ints(&self) -> &[i32] {
        unsafe { core::slice::from_raw_parts(self.as_ptr(), size_of::<__m128i>() / 4) }
    }
}

// inspired by:
// https://movermeyer.com/2023-01-04-rfc-3339-simd
// https://stackoverflow.com/questions/75680256/most-insanely-fast-way-to-convert-yymmdd-hhmmss-timestamp-to-uint64-t-number
// https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/165046e665893afd14b5a3ec2e99ee4c9e23bfd3/2023/07/01/src/sse_date.c#L72
// most X86 CPUs support these instructions so we only do compile time checks
#[target_feature(enable = "ssse3")]
pub(super) unsafe fn decode_seconds(ascii: &mut &[u8]) -> Result<i64, TimestampError> {
    unsafe {
        // YYYY-MM-DDTHH:MM:SS  (we require 19 + at least one terminator byte
        // so that the second 16-byte load at offset 3 stays in-bounds.)
        if ascii.len() < 20 {
            return Err(TimestampError::InvalidFormat);
        }

        // Two unaligned 16-byte loads cover bytes 0..16 and 3..19, between
        // them every byte we need (0..=18). Both loads are safe because
        // `ascii.len() >= 20`.
        let vec0 = _mm_loadu_si128(ascii.as_ptr().cast());
        let vec1 = _mm_loadu_si128(ascii.as_ptr().add(3).cast());

        // PSHUFB shuffle tables. The high bit (0x80, i.e. -128i8) zeros the
        // output byte; otherwise the low 4 bits select a source lane in the
        // same vector.
        //
        // Output layout (matches the original `_mm_setr_epi8` build):
        //   YYYY MM DD HH mm SS T :
        //   src bytes:  0 1 2 3 | 5 6 | 8 9 | 11 12 | 14 15 | 17 18 | 10 | 16
        //
        // For `vec1` (which is loaded at offset +3), the source-byte index `i`
        // appears at lane `i - 3`, so 17→14, 18→15, 16→13.
        const Z: i8 = -128; // 0x80, signals "zero lane" to PSHUFB
        let shuf0 = _mm_setr_epi8(
            0, 1, 2, 3, // Year     (vec0[0..=3])
            5, 6, // Month    (vec0[5..=6])
            8, 9, // Day      (vec0[8..=9])
            11, 12, // Hour     (vec0[11..=12])
            14, 15, // Minute   (vec0[14..=15])
            Z, Z,  // Second   (not in vec0)
            10, // 'T'      (vec0[10])
            Z,  // ':'      (not in vec0)
        );
        let shuf1 = _mm_setr_epi8(
            Z, Z, Z, Z, // Year
            Z, Z, // Month
            Z, Z, // Day
            Z, Z, // Hour
            Z, Z, // Minute
            14, 15, // Second   (src 17,18 → vec1[14..=15])
            Z,  // 'T'
            13, // ':'      (src 16 → vec1[13])
        );
        let mut tmp = _mm_or_si128(_mm_shuffle_epi8(vec0, shuf0), _mm_shuffle_epi8(vec1, shuf1));

        let lower_bound = _mm_setr_epi8(
            b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8,
            b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8, b'0' as i8,
            b'T' as i8, b':' as i8,
        );

        let upper_bound = _mm_setr_epi8(
            b'9' as i8, b'9' as i8, b'9' as i8, b'9' as i8, // Year
            b'1' as i8, b'9' as i8, // Month
            b'3' as i8, b'9' as i8, // Day
            b'2' as i8, b'9' as i8, // Hour
            b'5' as i8, b'9' as i8, // Minute
            b'5' as i8, b'9' as i8, // Second
            b'T' as i8, b':' as i8,
        );

        // value sanity check
        let lower = _mm_cmplt_epi8(tmp, lower_bound);
        let higher = _mm_cmpgt_epi8(tmp, upper_bound);
        if _mm_movemask_epi8(_mm_or_si128(lower, higher)) != 0 {
            return Err(TimestampError::InvalidFormat);
        }

        tmp = _mm_sub_epi8(tmp, lower_bound);

        let mult_10 = _mm_setr_epi8(10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 10, 1, 0, 0);
        tmp = _mm_maddubs_epi16(tmp, mult_10);

        let mut buf = AlignedBuffer::new();
        _mm_store_si128(buf.as_mut_ptr(), tmp);

        let res = buf.as_bytes();

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

#[target_feature(enable = "ssse3")]
pub(super) unsafe fn decode_nanos(ascii: &mut &[u8]) -> Result<i32, TimestampError> {
    if ascii[0] != b'.' {
        return Ok(0);
    }

    const THREE_DIGITS: i32 = 0xE;
    const SIX_DIGITS: i32 = 0xEE;
    const NINE_DIGITS: i32 = 0xEEE;

    unsafe {
        // Stage the (variable-length) digit bytes into a 16-byte aligned
        // buffer in the layout the rest of the routine expects:
        //
        //   lane:  0  1  2  3   4  5  6  7   8  9  10 11   12 13 14 15
        //   bytes: .  d  d  d   .  d  d  d   .  d  d  d    0  0  0  0
        //
        // Lane 0 holds the literal '.' from the input; the multiply-add
        // table below has weight 0 in that position so it is dropped. Unused
        // trailing lanes are zeroed.
        //
        // We can't `_mm_loadu_si128(ascii.as_ptr())` directly: for the 5- and
        // 8-byte cases (`.dddZ`, `.ddddddZ`), reading 16 bytes from `ascii`
        // could cross a page boundary. Instead, dispatch on length and use a
        // *fixed-size* copy in each arm so the compiler emits inline mov
        // instructions instead of a `memcpy` call. This collapses what was a
        // chain of 9 serially-dependent `vpinsrb` instructions in the
        // 9-digit (common, "now()") case into one qword + one word load.
        let mut stage = AlignedBuffer::new();
        let dst = stage.as_mut_ptr::<u8>();
        let src = ascii.as_ptr();
        match ascii.len() {
            5 => {
                // ".dddZ" — copy 4 bytes (".ddd").
                core::ptr::copy_nonoverlapping(src, dst, 4);
            }
            8 => {
                // ".ddddddZ" — copy 7 bytes (".dddddd"); use one qword copy
                // then mask off the trailing byte by re-zeroing lane 7 below.
                core::ptr::copy_nonoverlapping(src, dst, 7);
            }
            11..=16 => {
                // ".ddddddddd[Z|+HH:MM]" — copy 10 bytes (".ddddddddd"):
                // one qword + one word.
                core::ptr::copy_nonoverlapping(src, dst, 8);
                core::ptr::copy_nonoverlapping(src.add(8), dst.add(8), 2);
            }
            _ => return Err(TimestampError::InvalidFormat),
        }

        // Spread the contiguous bytes [.,d1,d2,d3,d4,d5,d6,d7,d8,d9,...] into
        // the 4-byte-per-group layout the maddubs step expects, zeroing every
        // 4th lane (PSHUFB's high-bit signals "zero this lane").
        const Z: i8 = -128;
        let spread = _mm_setr_epi8(
            Z, 1, 2, 3, // .d1 d2 d3
            Z, 4, 5, 6, // .d4 d5 d6
            Z, 7, 8, 9, // .d7 d8 d9
            Z, Z, Z, Z, // unused
        );
        let mut tmp = _mm_shuffle_epi8(_mm_load_si128(stage.as_ptr()), spread);

        // input validation
        let t0 = _mm_set1_epi8((b'0' + 128) as i8);
        let t1 = _mm_set1_epi8((128 + 10) as i8);
        let valid_digits = _mm_cmplt_epi8(_mm_sub_epi8(tmp, t0), t1);
        let mask = _mm_movemask_epi8(valid_digits);

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

        let ascii_digit_mask = _mm_set1_epi8(0x0f);
        tmp = _mm_and_si128(tmp, valid_digits);
        tmp = _mm_and_si128(tmp, ascii_digit_mask);

        let mut mult = _mm_setr_epi8(0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1, 0, 100, 10, 1);
        tmp = _mm_maddubs_epi16(tmp, mult);

        mult = _mm_setr_epi16(1000, 1000, 1000, 1000, 1, 1, 0, 0);
        tmp = _mm_madd_epi16(tmp, mult);

        let mut buf = AlignedBuffer::new();
        _mm_store_si128(buf.as_mut_ptr(), tmp);

        *ascii = &ascii[offset..];
        let out = buf.as_ints();

        Ok(out[0] * 1000 + out[1] + out[2])
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
}
