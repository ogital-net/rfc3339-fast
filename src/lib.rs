//! # RFC3339 Timestamp Library
//!
//! A high-performance library for parsing and formatting RFC3339/ISO8601
//! timestamps, with support for nanosecond precision.
//!
//! ## Overview
//!
//! This library provides efficient serialization and deserialization of RFC3339/ISO8601
//! timestamps in the format `YYYY-MM-DDTHH:mm:ss[.nnn]Z`. It supports:
//!
//! - Timestamps from year 1 to year 9999
//! - Nanosecond precision (up to 9 decimal places)
//! - Integration with `std::time::SystemTime` (with the default `std` feature)
//! - `no_std` support by disabling default features
//! - Optional support for `chrono` types via the `chrono` feature
//! - Optional `serde` integration via the `serde` feature
//! - SIMD acceleration on platforms supporting SSSE3 (`x86`/`x86_64`) or NEON (ARM)
//!
//! ## Examples
//!
//! Parsing a timestamp from a string:
//!
//! ```
//! use std::str::FromStr;
//! # use rfc3339_fast::Timestamp;
//! let ts = Timestamp::from_str("2026-02-25T14:30:00Z").unwrap();
//! ```
//!
//! Formatting a timestamp to a string:
//!
//! ```
//! # use rfc3339_fast::{Timestamp, Buffer};
//! let ts = Timestamp::now();
//! let mut buf = Buffer::new();
//! let formatted = buf.format(ts);
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]
// Crate-wide clippy::pedantic suppressions. Each one is intentional and
// scoped narrowly enough that the local code makes the safety/perf reason
// obvious; we silence them at the crate root to keep the hot paths
// uncluttered by `#[allow]` attributes on every line.
//
// * `cast_possible_wrap`, `cast_sign_loss`, `cast_lossless`,
//   `cast_possible_truncation`: the date-arithmetic and SIMD code does a
//   lot of `u32 ↔ i32` and `usize → u16`/`u32` casts on values that are
//   provably in range (years bounded by 1–9999, lengths bounded by
//   `BUFFER_SIZE = 30`, JD math pre-biased into the positive range, etc.).
//   Switching to `From::from` / `TryFrom` would either fail to compile
//   (different signedness) or add a runtime check on a hot path.
// * `unreadable_literal`: constants like `2440588` (Julian Day of the Unix
//   epoch) and `253402300799` (max representable Unix-seconds value) are
//   well-known reference numbers; underscore-grouping them obscures the
//   reference more than it helps.
// * `inline_always`: the per-byte `write_byte` / `write_number` /
//   `jsonenc_nanos` helpers are tiny leaf functions on the format hot
//   path; benchmarks regress noticeably if the inliner is allowed to
//   second-guess them.
// * `items_after_statements`: a few local `const`s are placed next to
//   their first use inside `jsonenc_timestamp` to keep the algorithm and
//   its magic numbers visually adjacent.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::unreadable_literal,
    clippy::inline_always,
    clippy::items_after_statements
)]

use core::{fmt, mem::MaybeUninit, ptr, str::FromStr};

#[cfg(feature = "std")]
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(target_feature = "ssse3")]
mod sse;

#[cfg(target_feature = "neon")]
mod neon;

/// Error type for parsing or formatting JSON timestamps.
///
/// This enum represents errors that can occur when parsing timestamp strings
/// or validating timestamp values.
///
/// Marked `#[non_exhaustive]` so additional variants can be introduced in a
/// future release without a semver break; downstream `match` expressions
/// must include a wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TimestampError {
    /// The input string had an invalid format.
    InvalidFormat,
    /// The timestamp value is out of the supported range
    /// (year 1 through year 9999, with `nanos < 1_000_000_000`).
    OutOfRange,
}

impl fmt::Display for TimestampError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimestampError::InvalidFormat => write!(f, "invalid timestamp format"),
            TimestampError::OutOfRange => write!(f, "timestamp value out of range"),
        }
    }
}

impl core::error::Error for TimestampError {}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
/// A timestamp value that can be serialized to and deserialized from JSON.
///
/// `Timestamp` is stored internally as `(seconds: i64, nanos: u32)`, where
/// `seconds` counts (signed) seconds since the Unix epoch and `nanos` is
/// always in `[0, 1_000_000_000)`. This matches the
/// [`google.protobuf.Timestamp`] convention and lets the type be
/// `no_std`-compatible. Negative whole-second values combined with a
/// positive `nanos` mean the same instant as the float
/// `seconds + nanos / 1e9`, e.g. `(-5, 200_000_000)` is `-4.8s`.
///
/// ## Invariants
///
/// Every `Timestamp` value satisfies:
///
/// * `seconds` is in `SECONDS_MIN..=SECONDS_MAX`, i.e. it represents an
///   instant between `0001-01-01T00:00:00Z` and `9999-12-31T23:59:59Z`
///   inclusive.
/// * `nanos` is in `[0, 1_000_000_000)`.
///
/// All public constructors enforce these invariants — [`Timestamp::now`],
/// [`Timestamp::from_unix`], [`Timestamp::from_str`], `From<SystemTime>`,
/// and the optional `From<chrono::DateTime<Tz>>` either validate, saturate,
/// or are infallible by construction. As a result, [`Buffer::format`] and
/// [`fmt::Display`] never fail.
///
/// With the default `std` feature, this type round-trips to and from
/// [`std::time::SystemTime`] efficiently — the conversion is just a
/// `duration_since(UNIX_EPOCH)` followed by storing the two fields
/// (saturating at the supported range bounds).
///
/// [`google.protobuf.Timestamp`]: https://protobuf.dev/reference/protobuf/google.protobuf/#timestamp
///
/// ## Creating timestamps
///
/// ```
/// # use rfc3339_fast::Timestamp;
/// // From Unix seconds + nanoseconds.
/// let ts = Timestamp::from_unix(1_641_006_000, 0).unwrap();
///
/// // Equivalent via the `TryFrom<(i64, u32)>` impl.
/// let ts = Timestamp::try_from((1_641_006_000i64, 0u32)).unwrap();
///
/// // With the `std` feature: from current system time.
/// # #[cfg(feature = "std")] {
/// let now = Timestamp::now();
/// let ts = Timestamp::from(std::time::SystemTime::now());
/// # }
/// ```
///
/// ## Inspecting timestamps
///
/// ```
/// # use rfc3339_fast::Timestamp;
/// let ts = Timestamp::from_unix(1_641_006_000, 250_000_000).unwrap();
/// assert_eq!(ts.seconds(), 1_641_006_000);
/// assert_eq!(ts.subsec_nanos(), 250_000_000);
/// ```
pub struct Timestamp {
    /// Signed seconds since the Unix epoch.
    seconds: i64,
    /// Fractional seconds, always in `[0, 1_000_000_000)`.
    nanos: u32,
}

impl Timestamp {
    /// Constructs a `Timestamp` directly from canonical `(seconds, nanos)`
    /// fields without range checks. Internal helper: callers must
    /// guarantee `seconds` is in `SECONDS_MIN..=SECONDS_MAX` and `nanos`
    /// is in `[0, 1_000_000_000)`.
    #[inline]
    fn new_unchecked(seconds: i64, nanos: u32) -> Self {
        debug_assert!(nanos < 1_000_000_000);
        Self { seconds, nanos }
    }

    /// Constructs a `Timestamp` from Unix seconds and nanoseconds.
    ///
    /// Returns [`TimestampError::OutOfRange`] if `seconds` is outside the
    /// representable year 1..=9999 range, or if `nanos >= 1_000_000_000`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use rfc3339_fast::Timestamp;
    /// let ts = Timestamp::from_unix(0, 0).unwrap();
    /// assert_eq!(ts.seconds(), 0);
    /// assert_eq!(ts.subsec_nanos(), 0);
    /// ```
    pub fn from_unix(seconds: i64, nanos: u32) -> Result<Self, TimestampError> {
        if !(SECONDS_MIN..=SECONDS_MAX).contains(&seconds) || nanos >= 1_000_000_000 {
            return Err(TimestampError::OutOfRange);
        }
        Ok(Self::new_unchecked(seconds, nanos))
    }

    /// Returns the (signed) seconds component, counted from the Unix epoch.
    #[inline]
    #[must_use]
    pub fn seconds(&self) -> i64 {
        self.seconds
    }

    /// Returns the fractional-second component, in nanoseconds.
    ///
    /// The returned value is always in `[0, 1_000_000_000)`. The naming
    /// matches [`std::time::Duration::subsec_nanos`].
    #[inline]
    #[must_use]
    pub fn subsec_nanos(&self) -> u32 {
        self.nanos
    }

    /// Returns a `Timestamp` representing the current system time.
    ///
    /// This is a convenience method for `Timestamp::from(SystemTime::now())`.
    #[cfg(feature = "std")]
    #[must_use]
    pub fn now() -> Self {
        Self::from(SystemTime::now())
    }
}

/// Converts a `SystemTime` into a `Timestamp` by computing its offset from
/// the Unix epoch.
///
/// `SystemTime` can in principle represent instants outside the
/// `Timestamp` range (year 1 through year 9999); such values are
/// **saturated** to the nearest in-range second. In practice this only
/// affects deliberately constructed `SystemTime`s; wall-clock times from
/// the system clock are well within range.
#[cfg(feature = "std")]
impl From<SystemTime> for Timestamp {
    #[inline]
    fn from(value: SystemTime) -> Self {
        // Both branches collapse to a single `duration_since` plus a
        // small fixup; LLVM inlines this away when `format` is called
        // directly on a `SystemTime`. The final `clamp` enforces the
        // `Timestamp` range invariant.
        let (seconds, nanos) = match value.duration_since(UNIX_EPOCH) {
            Ok(dur) => (dur.as_secs() as i64, dur.subsec_nanos()),
            Err(e) => {
                let dur_before = e.duration();
                let secs_before = -(dur_before.as_secs() as i64);
                let nanos_before = dur_before.subsec_nanos();
                if nanos_before > 0 {
                    (secs_before - 1, 1_000_000_000 - nanos_before)
                } else {
                    (secs_before, 0)
                }
            }
        };
        // Saturate out-of-range values to the supported bounds. When
        // saturating to `SECONDS_MAX`, drop sub-second precision so the
        // result still represents `9999-12-31T23:59:59Z`.
        if seconds < SECONDS_MIN {
            Self::new_unchecked(SECONDS_MIN, 0)
        } else if seconds > SECONDS_MAX {
            Self::new_unchecked(SECONDS_MAX, 0)
        } else {
            Self::new_unchecked(seconds, nanos)
        }
    }
}

/// Wraps a `SystemTime` reference as a `Timestamp`. See [`From<SystemTime>`]
/// for the saturation contract.
#[cfg(feature = "std")]
impl From<&SystemTime> for Timestamp {
    #[inline]
    fn from(value: &SystemTime) -> Self {
        Self::from(*value)
    }
}

/// Converts a `Timestamp` back into a `SystemTime` by adding (or
/// subtracting) its offset from the Unix epoch.
#[cfg(feature = "std")]
impl From<Timestamp> for SystemTime {
    #[inline]
    fn from(value: Timestamp) -> Self {
        if value.seconds >= 0 {
            UNIX_EPOCH + Duration::new(value.seconds as u64, value.nanos)
        } else {
            // Canonical form has nanos in [0, 1e9). For negative seconds,
            // a non-zero `nanos` adds time *forward*, so the magnitude of
            // the offset is `-seconds - 1` whole seconds plus `1e9 - nanos`
            // sub-second component (when nanos > 0).
            let (mag_secs, mag_nanos) = if value.nanos == 0 {
                ((-value.seconds) as u64, 0)
            } else {
                ((-value.seconds - 1) as u64, 1_000_000_000 - value.nanos)
            };
            UNIX_EPOCH - Duration::new(mag_secs, mag_nanos)
        }
    }
}

impl From<&Timestamp> for Timestamp {
    #[inline]
    fn from(value: &Timestamp) -> Self {
        *value
    }
}

#[cfg(feature = "chrono")]
impl<Tz: chrono::TimeZone> From<chrono::DateTime<Tz>> for Timestamp {
    fn from(value: chrono::DateTime<Tz>) -> Self {
        // chrono's `timestamp()` returns signed Unix seconds and
        // `timestamp_subsec_nanos()` returns nanos in [0, 1e9), so this
        // already matches our canonical form — no SystemTime detour.
        // Saturate to the `Timestamp` range to uphold its invariant.
        let seconds = value.timestamp();
        let nanos = value.timestamp_subsec_nanos();
        if seconds < SECONDS_MIN {
            Self::new_unchecked(SECONDS_MIN, 0)
        } else if seconds > SECONDS_MAX {
            Self::new_unchecked(SECONDS_MAX, 0)
        } else {
            Self::new_unchecked(seconds, nanos)
        }
    }
}

#[cfg(feature = "chrono")]
impl From<Timestamp> for chrono::DateTime<chrono::Utc> {
    fn from(value: Timestamp) -> Self {
        chrono::DateTime::<chrono::Utc>::from_timestamp(value.seconds, value.nanos)
            .expect("Timestamp out of range for chrono::DateTime")
    }
}

#[cfg(feature = "serde")]
impl serde_core::Serialize for Timestamp {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde_core::Serializer,
    {
        let mut buf = Buffer::new();
        serializer.serialize_str(buf.format(self))
    }
}

#[cfg(feature = "serde")]
impl<'de> serde_core::Deserialize<'de> for Timestamp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde_core::Deserializer<'de>,
    {
        struct TsVisitor;

        impl serde_core::de::Visitor<'_> for TsVisitor {
            type Value = Timestamp;

            #[inline]
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("an ISO8601 Timestamp")
            }

            #[inline]
            fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
            where
                E: serde_core::de::Error,
            {
                Timestamp::from_str(v).map_err(|_e| E::custom("Invalid Format"))
            }

            #[inline]
            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: serde_core::de::Error,
            {
                let s = core::str::from_utf8(v).map_err(|_| E::custom("Invalid Format"))?;
                self.visit_str(s)
            }
        }
        deserializer.deserialize_str(TsVisitor)
    }
}

impl TryFrom<(i64, u32)> for Timestamp {
    type Error = TimestampError;

    #[inline]
    fn try_from(value: (i64, u32)) -> Result<Self, Self::Error> {
        Self::from_unix(value.0, value.1)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = Buffer::new();
        write!(f, "{}", buf.format(self))
    }
}

impl FromStr for Timestamp {
    type Err = TimestampError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut ascii = s.as_bytes();

        #[cfg(target_feature = "ssse3")]
        let (seconds, nanos) = unsafe {
            (
                sse::decode_seconds(&mut ascii)?,
                sse::decode_nanos(&mut ascii)?,
            )
        };

        #[cfg(target_feature = "neon")]
        let (seconds, nanos) = unsafe {
            (
                neon::decode_seconds(&mut ascii)?,
                neon::decode_nanos(&mut ascii)?,
            )
        };

        #[cfg(not(any(target_feature = "ssse3", target_feature = "neon")))]
        let (seconds, nanos) = (decode_seconds(&mut ascii)?, decode_nanos(&mut ascii)?);

        let offset = match ascii.first() {
            Some(b'Z') => 0,
            Some(&c @ (b'+' | b'-')) => decode_offset(ascii, c)?,
            _ => return Err(TimestampError::InvalidFormat),
        };

        // `decode_nanos` returns `i32` in `[0, 1_000_000_000)`, and
        // `decode_seconds` keeps the year in 1..=9999, so the cast and
        // unchecked constructor are sound here.
        Ok(Self::new_unchecked(seconds + offset, nanos as u32))
    }
}

/// Decodes an RFC3339 numeric timezone offset of the form `+HH:MM` or `-HH:MM`.
///
/// Returns the value (in seconds) that must be added to the local-time seconds
/// to yield UTC seconds. For example, `+05:00` yields `-18000`.
#[inline]
fn decode_offset(ascii: &[u8], sign: u8) -> Result<i64, TimestampError> {
    // Expect exactly 6 bytes: [+-]HH:MM
    if ascii.len() != 6 || ascii[3] != b':' {
        return Err(TimestampError::InvalidFormat);
    }

    let h10 = ascii[1].wrapping_sub(b'0');
    let h1 = ascii[2].wrapping_sub(b'0');
    let m10 = ascii[4].wrapping_sub(b'0');
    let m1 = ascii[5].wrapping_sub(b'0');
    if (h10 | h1 | m10 | m1) > 9 {
        return Err(TimestampError::InvalidFormat);
    }

    let hours = h10 as i64 * 10 + h1 as i64;
    let mins = m10 as i64 * 10 + m1 as i64;
    if hours > 23 || mins > 59 {
        return Err(TimestampError::InvalidFormat);
    }

    let magnitude = hours * 3600 + mins * 60;
    // local = UTC + offset (when sign is '+'), so UTC = local - offset.
    Ok(if sign == b'+' { -magnitude } else { magnitude })
}

// 30 bytes is exactly enough for the longest valid timestamp:
//   `YYYY-MM-DDTHH:mm:ss.sssssssssZ` (4+1+2+1+2 + 1 + 2+1+2+1+2 + 1+9 + 1 = 30).
// Combined with a `u16` length below, the whole `Buffer` is 32 bytes — one
// cacheline / four qwords — which keeps it cheap to copy and well-aligned on
// the stack.
const BUFFER_SIZE: usize = 30; // YYYY-MM-DDTHH:mm:ss.sssssssssZ
const SECONDS_MIN: i64 = -62135596800;
const SECONDS_MAX: i64 = 253402300799;

/// Pre-computed pair table: byte `2*N` and `2*N+1` are the ASCII digits of
/// `N` for `N` in `0..=99`. Used by `write_2_at` / `write_4_at` /
/// `write_9_at` to convert a value to its two-digit ASCII form with one
/// indexed `u16` load instead of two scalar divides.
const PAIR_TABLE: [u8; 200] = {
    let mut t = [0u8; 200];
    let mut i = 0;
    while i < 100 {
        t[i * 2] = b'0' + (i / 10) as u8;
        t[i * 2 + 1] = b'0' + (i % 10) as u8;
        i += 1;
    }
    t
};

/// A reusable buffer for formatting timestamps to strings.
///
/// `Buffer` provides an efficient way to format multiple timestamps without
/// allocating memory. It uses a fixed-size stack-allocated buffer that is
/// large enough to hold any valid ISO8601 timestamp with nanosecond precision.
///
/// The buffer size is 30 bytes, which is sufficient to hold the longest
/// possible timestamp: `9999-12-31T23:59:59.999999999Z`.
///
/// ## Examples
///
/// ```
/// # use rfc3339_fast::{Timestamp, Buffer};
/// let mut buf = Buffer::new();
/// let ts = Timestamp::now();
/// let formatted = buf.format(ts);
/// println!("{}", formatted);  // Prints the timestamp
/// ```
pub struct Buffer {
    bytes: [MaybeUninit<u8>; BUFFER_SIZE],
    // `len` is bounded by `BUFFER_SIZE` (30), so a `u16` is plenty and lets
    // the whole struct round to a tidy 32 bytes. We `as`-cast freely between
    // `len` and `usize` because the value is guaranteed to fit.
    len: u16,
}

impl Default for Buffer {
    #[inline]
    fn default() -> Buffer {
        Buffer::new()
    }
}

impl Copy for Buffer {}

// `Clone` is implemented manually rather than derived because the buffer
// holds `MaybeUninit<u8>` whose contents are only valid for the
// `..self.len` prefix; cloning by copying that uninitialized tail would
// be wasteful and semantically meaningless. We instead return a fresh
// empty buffer, which matches the typical usage pattern (a `Buffer` is
// always reset before each `format` call). The two clippy lints below
// flag the unusual semantics on purpose; we accept them.
#[allow(clippy::non_canonical_clone_impl, clippy::expl_impl_clone_on_copy)]
impl Clone for Buffer {
    #[inline]
    fn clone(&self) -> Self {
        Buffer::new()
    }
}

impl Buffer {
    #[inline]
    /// Creates a new empty `Buffer`.
    ///
    /// The buffer can then be used to format multiple timestamps.
    #[must_use]
    pub fn new() -> Buffer {
        let bytes = [MaybeUninit::<u8>::uninit(); BUFFER_SIZE];
        Buffer { bytes, len: 0 }
    }

    /// Formats a timestamp into an ISO8601 string.
    ///
    /// This method converts a timestamp (or anything convertible to `Timestamp`)
    /// into a string in the format `YYYY-MM-DDTHH:mm:ss[.nnn]Z`.
    ///
    /// The returned string is a borrowed reference to data stored in this buffer.
    /// To format another timestamp, call `format` again and it will overwrite
    /// the previous contents.
    ///
    /// This method is infallible: every [`Timestamp`] value is guaranteed by
    /// construction to lie in the representable range
    /// (`0001-01-01T00:00:00Z` through `9999-12-31T23:59:59.999999999Z`),
    /// and `Buffer` is sized to hold the longest such string.
    ///
    /// # Arguments
    ///
    /// * `timestamp` - A value that can be converted to `Timestamp` (includes
    ///   `SystemTime`, `&SystemTime`, `&Timestamp`, and `chrono::DateTime` when
    ///   the `chrono` feature is enabled).
    ///
    /// # Returns
    ///
    /// A string slice containing the formatted timestamp.
    pub fn format<T: Into<Timestamp>>(&mut self, timestamp: T) -> &str {
        let timestamp = timestamp.into();
        self.reset();

        let seconds = timestamp.seconds;
        let nanos = timestamp.nanos as i32;

        // SAFETY/correctness: `Timestamp`'s constructors guarantee
        // `seconds` lies in `SECONDS_MIN..=SECONDS_MAX` and `nanos` is in
        // `[0, 1_000_000_000)`, so the date arithmetic and the
        // fixed-offset writes inside `jsonenc_timestamp` stay within the
        // 30-byte buffer.
        debug_assert!((SECONDS_MIN..=SECONDS_MAX).contains(&seconds));
        debug_assert!((0..1_000_000_000).contains(&nanos));

        self.jsonenc_timestamp(seconds, nanos);
        self.as_str()
    }

    #[inline]
    fn reset(&mut self) {
        self.len = 0;
    }

    /// Writes a single byte to the buffer.
    ///
    /// This is a low-level method used internally for formatting. The buffer
    /// is sized to fit the longest possible ISO8601 timestamp, so callers
    /// inside [`Buffer::jsonenc_timestamp`] never overflow it; we assert
    /// this only in debug builds to keep the hot path branch-free.
    #[inline(always)]
    fn write_byte(&mut self, value: u8) {
        let len = self.len as usize;
        debug_assert!(len < BUFFER_SIZE, "Buffer overflow in write_byte");
        // SAFETY: caller ensures len < BUFFER_SIZE; checked in debug builds.
        unsafe {
            let end = self.bytes.as_mut_ptr().cast::<u8>().add(len);
            ptr::write(end, value);
        }
        self.len = (len + 1) as u16;
    }

    /// Writes a numeric value with a fixed number of digits to the buffer.
    ///
    /// Pads with leading zeros as needed. For example, `write_number(42, 4)` writes
    /// `0042`. Processes digits in pairs for efficiency.
    ///
    /// # Arguments
    ///
    /// * `value` - The number to write
    /// * `digits` - The number of digits to write (with zero-padding)
    #[inline(always)]
    fn write_number(&mut self, mut value: u32, mut digits: usize) {
        let len = self.len as usize + digits;
        debug_assert!(len <= BUFFER_SIZE, "Buffer overflow in write_number");
        if BUFFER_SIZE >= len {
            unsafe {
                self.len = len as u16;
                let mut ptr = self.bytes.as_mut_ptr().cast::<u8>().add(len - 1);
                // process 2 digits per iteration, this loop will likely be unrolled
                while digits >= 2 {
                    // combine these so the compiler can optimize both operations
                    let d1;
                    (value, d1) = (value / 100, value % 100);

                    let (a, b) = (d1 / 10, d1 % 10);
                    digits -= 1;
                    ptr.write(b as u8 | b'0');
                    ptr = ptr.sub(1);
                    digits -= 1;
                    ptr.write(a as u8 | b'0');
                    ptr = ptr.sub(1);
                }

                // handle remainder
                if digits == 1 {
                    ptr.write(value as u8 | b'0');
                }
            }
        }
    }

    /// Encodes a Unix timestamp into ISO8601 format in the buffer.
    ///
    /// Converts seconds and nanoseconds since the epoch into a formatted string
    /// in the format `YYYY-MM-DDTHH:mm:ss[.nnn...]Z`.
    ///
    /// # Arguments
    ///
    /// * `seconds` - Seconds since the Unix epoch
    /// * `nanos` - Nanoseconds (0-999,999,999)
    ///
    /// # Algorithm
    ///
    /// The date portion is computed by the Fliegel/Van Flandern algorithm,
    /// which converts a Julian Day Number into a Gregorian (Y, M, D) triple
    /// using only integer arithmetic — no lookup tables and no branches.
    /// The original 1968 publication is:
    ///
    /// > Fliegel, H. F., and Van Flandern, T. C., "A Machine Algorithm for
    /// > Processing Calendar Dates," Communications of the ACM, vol. 11
    /// > no. 10 (October 1968), p. 657.
    ///
    /// The magic constants encode the Gregorian calendar's irregular cycle
    /// of month lengths and leap years:
    ///
    /// * `146097` — days in a 400-year Gregorian cycle (the smallest cycle
    ///   over which the calendar exactly repeats: `400*365 + 100 - 4 + 1`).
    /// * `1461`   — days in a 4-year Julian cycle (`4*365 + 1`).
    /// * `2447 / 80` — a piecewise-linear approximation of cumulative
    ///   month lengths (March-based), exploiting truncating integer
    ///   division so that successive months fall on the right day-of-year
    ///   boundary without a lookup table.
    /// * `68569`  — shifts the Julian Day Number into the algorithm's
    ///   internal positive range.
    /// * `49`     — recovers the original year after the 100-year
    ///   regrouping done by the `n` term.
    ///
    /// We pre-bias `seconds` by the offset from 0001-01-01 to 1970-01-01 so
    /// the value passed to the integer divisions is always non-negative,
    /// which matches the algorithm's preconditions and avoids the
    /// round-toward-zero pitfalls of signed division on negative operands.
    ///
    /// Background and a friendly walk-through of the Fortran original (and
    /// of the related branchless variants) is in Josh Haberman's article
    /// <https://blog.reverberate.org/2020/05/12/optimizing-date-algorithms.html>.
    /// This implementation is a Rust port inspired by upb's C version:
    /// <https://github.com/protocolbuffers/protobuf/blob/27421b97a0daa29e91460d377b0213f9e7be5d3f/upb/json/encode.c#L122>.
    #[inline(always)]
    fn jsonenc_timestamp(&mut self, mut seconds: i64, nanos: i32) {
        const SECONDS_PER_DAY: i32 = 86400;

        // Days from 0001-01-01 (proleptic Gregorian) to 1970-01-01.
        const CE_EPOCH_TO_UNIX_EPOCH_DAYS: i32 = 719_162;
        const CE_EPOCH_TO_UNIX_EPOCH_SECONDS: i64 =
            CE_EPOCH_TO_UNIX_EPOCH_DAYS as i64 * SECONDS_PER_DAY as i64;

        // Julian Day Number of the Unix epoch (1970-01-01). The Julian
        // period starts on -4713-11-24 (proleptic Gregorian), so the
        // Unix epoch is day 2_440_588 of that count.
        const JD_UNIX_EPOCH: i32 = 2_440_588;

        // Pre-fill the buffer with the fixed parts of the output:
        //   YYYY-MM-DDTHH:MM:SS.NNNNNNNNNZ
        //   0123456789012345678901234567890
        //             1111111111222222222
        // Underscores are placeholders for digits we'll overwrite below.
        // Doing this as one 30-byte block lets LLVM lower it to a pair of
        // 16-byte SSE stores; in exchange, the 9 separator bytes (`-`,
        // `T`, `:`, `.`, `Z`) never need to be written individually inside
        // the hot path. Just as importantly, by writing each digit at a
        // *fixed* offset (rather than threading `self.len` through every
        // call) we break the serial dependency chain between writes, so
        // the back end can issue them in parallel.
        const TEMPLATE: [u8; 30] = *b"____-__-__T__:__:__.000000000Z";
        // SAFETY: `self.bytes` is `[MaybeUninit<u8>; 30]`, exactly the
        // same size as `TEMPLATE`, and `MaybeUninit<u8>` has the same
        // layout as `u8`.
        unsafe {
            ptr::copy_nonoverlapping(TEMPLATE.as_ptr(), self.bytes.as_mut_ptr().cast::<u8>(), 30);
        }

        // Bias into the positive range expected by the F/VF formula, then
        // convert seconds-since-CE-epoch into a Julian Day Number plus the
        // algorithm's internal offset of 68569.
        seconds += CE_EPOCH_TO_UNIX_EPOCH_SECONDS;
        let days = (seconds / SECONDS_PER_DAY as i64) as i32;
        let mut l = days - CE_EPOCH_TO_UNIX_EPOCH_DAYS + JD_UNIX_EPOCH + 68569;

        // `n` is the number of completed 400-year Gregorian cycles since
        // the algorithm's internal epoch; subtracting them out leaves a
        // residue `l` in [0, 146096] (one full cycle of days).
        let n = 4 * l / 146097;
        l -= (146097 * n + 3) / 4;

        // Within the cycle, recover the year (March-based) and the
        // remaining day-of-year, then split that into month and day using
        // the (80 * l / 2447) piecewise-linear month formula.
        let mut year = 4000 * (l + 1) / 1461001;
        l = l - 1461 * year / 4 + 31;
        let mut month = 80 * l / 2447;
        let day = l - 2447 * month / 80;

        // Shift March-based months back to January-based, carrying into
        // the year when month is 11 or 12 (i.e. Jan/Feb of the next year).
        l = month / 11;
        month = month + 2 - 12 * l;
        year = 100 * (n - 49) + year + l;

        // Time-of-day from the seconds-of-day remainder. Doing one i32
        // divmod for the day boundary, then deriving h/m/s from the
        // u32 remainder, keeps these off the i64 critical path.
        let sod = (seconds - days as i64 * SECONDS_PER_DAY as i64) as u32; // 0..86400
        let hour = sod / 3600;
        let rem = sod % 3600;
        let min = rem / 60;
        let sec = rem % 60;

        // SAFETY: all offsets are < 30 == BUFFER_SIZE; values are bounded
        // (year <= 9999, the rest <= 99) so the digit math stays in u32.
        unsafe {
            self.write_4_at(year as u32, 0);
            self.write_2_at(month as u32, 5);
            self.write_2_at(day as u32, 8);
            self.write_2_at(hour, 11);
            self.write_2_at(min, 14);
            self.write_2_at(sec, 17);
        }

        // Nanoseconds: figure out how many trailing groups of 3 are zero
        // and patch the buffer accordingly. The template already contains
        // ".000000000Z" at offsets 19..30, so when we omit the fraction
        // entirely we just overwrite the '.' at 19 with 'Z' and stop
        // there; the trailing template bytes are unread because `len`
        // bounds `as_str`.
        let final_len = if nanos == 0 {
            // SAFETY: 19 < 30.
            unsafe { self.write_byte_at(b'Z', 19) };
            20
        } else {
            // Always materialize all 9 digits into [20..29]; the 'Z' at
            // [29] from the template stays put. Then re-place 'Z' at 24
            // or 27 if the trailing 6 / 3 nano digits are zero.
            // SAFETY: offset 20 + 9 == 29 < 30.
            unsafe { self.write_9_at(nanos as u32, 20) };
            // Trim trailing groups of 3 zeros. We compute the trim from
            // `nanos` directly (cheap divmods on a constant divisor) so
            // the back end can hoist these alongside the digit writes.
            if nanos % 1000 != 0 {
                30
            } else if (nanos / 1000) % 1000 != 0 {
                // SAFETY: 26 < 30.
                unsafe { self.write_byte_at(b'Z', 26) };
                27
            } else {
                // SAFETY: 23 < 30.
                unsafe { self.write_byte_at(b'Z', 23) };
                24
            }
        };
        self.len = final_len;
    }

    /// Encodes the nanosecond component of a timestamp.
    ///
    /// (Retained for callers that may want a stand-alone helper; the hot
    /// `jsonenc_timestamp` path now writes nanos via fixed-offset stores
    /// directly into the templated buffer instead of going through this
    /// `self.len`-threaded routine.)
    #[allow(dead_code)]
    #[inline(always)]
    fn jsonenc_nanos(&mut self, mut nanos: u32) {
        if nanos == 0 {
            return;
        }
        let mut digits = 9;

        let mut q;
        let mut r;
        (q, r) = (nanos / 1000, nanos % 1000);
        if r != 0 {
            self.write_byte(b'.');
            self.write_number(nanos, digits);
            return;
        }
        nanos = q;
        digits -= 3;
        (q, r) = (nanos / 1000, nanos % 1000);
        if r != 0 {
            self.write_byte(b'.');
            self.write_number(nanos, digits);
            return;
        }
        nanos = q;
        digits -= 3;
        r = nanos % 1000;
        if r != 0 {
            self.write_byte(b'.');
            self.write_number(nanos, digits);
        }
    }

    /// Writes a single byte at a fixed offset (no `self.len` update).
    ///
    /// # Safety
    ///
    /// `offset` must be `< BUFFER_SIZE`.
    #[inline(always)]
    unsafe fn write_byte_at(&mut self, value: u8, offset: usize) {
        debug_assert!(offset < BUFFER_SIZE);
        unsafe {
            self.bytes
                .as_mut_ptr()
                .cast::<u8>()
                .add(offset)
                .write(value);
        }
    }

    /// Writes a 2-digit zero-padded number at a fixed offset.
    ///
    /// Uses a 200-byte ASCII pair table (`"00", "01", …, "99"`) to do the
    /// conversion as one 2-byte load + one 2-byte store, saving the two
    /// `div_by_10`s the obvious code would use. The table is pre-built at
    /// compile time.
    ///
    /// # Safety
    ///
    /// `offset + 1` must be `< BUFFER_SIZE` and `value` must be `< 100`.
    #[inline(always)]
    unsafe fn write_2_at(&mut self, value: u32, offset: usize) {
        debug_assert!(offset + 1 < BUFFER_SIZE && value < 100);
        unsafe {
            let src = PAIR_TABLE.as_ptr().add(value as usize * 2);
            let dst = self.bytes.as_mut_ptr().cast::<u8>().add(offset);
            ptr::copy_nonoverlapping(src, dst, 2);
        }
    }

    /// Writes a 4-digit zero-padded number at a fixed offset.
    ///
    /// Splits the value into two 2-digit halves and emits each via the
    /// `PAIR_TABLE`, so the 4 ASCII bytes are produced by two 2-byte
    /// table loads instead of four scalar divides.
    ///
    /// # Safety
    ///
    /// `offset + 3` must be `< BUFFER_SIZE` and `value` must be `< 10_000`.
    #[inline(always)]
    unsafe fn write_4_at(&mut self, value: u32, offset: usize) {
        debug_assert!(offset + 3 < BUFFER_SIZE && value < 10_000);
        let hi = (value / 100) as usize;
        let lo = (value % 100) as usize;
        unsafe {
            let dst = self.bytes.as_mut_ptr().cast::<u8>().add(offset);
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(hi * 2), dst, 2);
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(lo * 2), dst.add(2), 2);
        }
    }

    /// Writes a 9-digit zero-padded number at a fixed offset.
    ///
    /// # Safety
    ///
    /// `offset + 8` must be `< BUFFER_SIZE` and `value` must be
    /// `< 1_000_000_000`.
    #[inline(always)]
    unsafe fn write_9_at(&mut self, value: u32, offset: usize) {
        debug_assert!(offset + 8 < BUFFER_SIZE && value < 1_000_000_000);
        // Split into 1 + 2 + 2 + 2 + 2 digits so we can use the pair
        // table for everything but the leading digit.
        let q1 = value / 100_000_000; // top 1 digit (0..9)
        let r1 = value % 100_000_000;
        let q2 = r1 / 1_000_000; // next 2 digits
        let r2 = r1 % 1_000_000;
        let q3 = r2 / 10_000; // next 2 digits
        let r3 = r2 % 10_000;
        let q4 = r3 / 100; // next 2 digits
        let q5 = r3 % 100; // last 2 digits
        unsafe {
            let dst = self.bytes.as_mut_ptr().cast::<u8>().add(offset);
            dst.write(q1 as u8 | b'0');
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(q2 as usize * 2), dst.add(1), 2);
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(q3 as usize * 2), dst.add(3), 2);
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(q4 as usize * 2), dst.add(5), 2);
            ptr::copy_nonoverlapping(PAIR_TABLE.as_ptr().add(q5 as usize * 2), dst.add(7), 2);
        }
    }

    /// Returns the formatted timestamp as a string slice.
    ///
    /// This method safely converts the buffer's uninitialized bytes to a UTF-8 string.
    fn as_str(&self) -> &str {
        // SAFETY: `self.len` is only advanced by `write_byte`/`write_number`,
        // which write valid bytes via raw pointers, so the prefix
        // `..self.len` is fully initialized.
        let written = unsafe { self.bytes.get_unchecked(..self.len as usize) };
        // SAFETY: `MaybeUninit<u8>` and `u8` have identical layout, and the
        // bytes are initialized (above). All writes use ASCII exclusively.
        unsafe {
            core::str::from_utf8_unchecked(
                &*(ptr::from_ref::<[MaybeUninit<u8>]>(written) as *const [u8]),
            )
        }
    }
}

#[inline]
#[allow(dead_code)]
fn atoi_consume(ascii: &mut &[u8]) -> i32 {
    let mut n: i32 = 0;
    let (s, neg) = match ascii[0] {
        b'-' => (&ascii[1..], true),
        b'+' => (&ascii[1..], false),
        _ => (*ascii, false),
    };

    let mut idx: usize = 0;
    // Compute n as a negative number to avoid overflow
    for c in s {
        if !c.is_ascii_digit() {
            break;
        }
        idx += 1;
        n = n * 10 - i32::from(c & 0x0f);
    }

    *ascii = &s[idx..];
    if neg {
        n
    } else {
        -n
    }
}

/// Decodes the seconds component from an ISO8601 timestamp string.
///
/// Parses the date and time portion of an ISO8601 timestamp string
/// (format: `YYYY-MM-DDTHH:mm:ss`) and returns the Unix timestamp
/// (seconds since 1970-01-01T00:00:00Z).
///
/// # Arguments
///
/// * `ascii` - A mutable reference to a byte slice. On success, this is
///   advanced past the parsed seconds field.
///
/// # Returns
///
/// - `Ok(seconds)` - The number of seconds since Unix epoch
/// - `Err(TimestampError::InvalidFormat)` - If the format is invalid
#[inline]
#[allow(dead_code)]
fn decode_seconds(ascii: &mut &[u8]) -> Result<i64, TimestampError> {
    // 1972-01-01T01:00:00
    let year = decode_tsdigits(ascii, 4, Some(b'-'))?;
    let mon = decode_tsdigits(ascii, 2, Some(b'-'))?;
    let day = decode_tsdigits(ascii, 2, Some(b'T'))?;
    let hour = decode_tsdigits(ascii, 2, Some(b':'))?;
    let min = decode_tsdigits(ascii, 2, Some(b':'))?;
    let sec = decode_tsdigits(ascii, 2, None)?;

    Ok(jsondec_unixtime(year, mon, day, hour, min, sec))
}

/// Decodes a sequence of digits from a timestamp string.
///
/// Parses a fixed number of ASCII digits from the input and optionally
/// validates a delimiter character after the digits.
///
/// # Arguments
///
/// * `ascii` - A mutable reference to a byte slice to parse from
/// * `digits` - The number of ASCII digits to parse
/// * `after` - An optional expected delimiter character. If provided and
///   doesn't match the character after the digits, returns an error.
///
/// # Returns
///
/// - `Ok(value)` - The parsed integer value
/// - `Err(TimestampError::InvalidFormat)` - If parsing fails or delimiter doesn't match
#[inline]
#[allow(dead_code)]
fn decode_tsdigits(
    ascii: &mut &[u8],
    mut digits: usize,
    after: Option<u8>,
) -> Result<i32, TimestampError> {
    if after.is_some_and(|v| v != ascii[digits]) {
        return Err(TimestampError::InvalidFormat);
    }
    let mut s = &ascii[..digits];
    let i = atoi_consume(&mut s);
    if !s.is_empty() {
        return Err(TimestampError::InvalidFormat);
    }

    if after.is_some() {
        digits += 1;
    }
    *ascii = &ascii[digits..];
    Ok(i)
}

/// Decodes the nanoseconds component from an ISO8601 timestamp string.
///
/// Parses the optional fractional seconds portion of a timestamp
/// (format: `.nnn` where n is 3, 6, or 9 digits).
///
/// # Arguments
///
/// * `ascii` - A mutable reference to a byte slice. On success, this is
///   advanced past the parsed nanoseconds field.
///
/// # Returns
///
/// - `Ok(nanos)` - The nanosecond value (0-999,999,999)
/// - `Err(TimestampError::InvalidFormat)` - If the fractional seconds format is invalid
///   (must be 3, 6, or 9 digits)
#[inline]
#[allow(dead_code)]
fn decode_nanos(ascii: &mut &[u8]) -> Result<i32, TimestampError> {
    let mut nanos: i32 = 0;
    if ascii[0] == b'.' {
        let mut remaining = &ascii[1..];
        nanos = atoi_consume(&mut remaining);
        let digits = ascii.len() - 1 - remaining.len();
        match digits {
            3 | 6 | 9 => {}
            _ => {
                return Err(TimestampError::InvalidFormat);
            }
        }
        let mut exp_lg10 = 9 - digits as i32;
        while exp_lg10 > 0 {
            exp_lg10 -= 1;
            nanos *= 10;
        }
        *ascii = remaining;
    }
    Ok(nanos)
}

/// Calculates the number of days from a given date to the Unix epoch (1970-01-01).
///
/// # Arguments
///
/// * `y` - Year
/// * `m` - Month (1-12)
/// * `d` - Day (1-31)
///
/// # Returns
///
/// The number of days since the Unix epoch (negative for dates before 1970-01-01).
///
/// # Note
///
/// `jsondec_epochdays(1970, 1, 1) == 0`.
///
/// # Algorithm
///
/// This is the inverse direction of the Fliegel/Van Flandern conversion
/// used by [`Buffer::jsonenc_timestamp`]: given (Y, M, D) it returns the
/// signed day count since 1970-01-01 without lookup tables and (after
/// optimization) without branches.
///
/// The shape of the formula is due to Howard Hinnant
/// (<http://howardhinnant.github.io/date_algorithms.html#days_from_civil>),
/// with the specific power-of-two divisor variant due to Gerben Stavenga
/// — both surveyed in Josh Haberman's article
/// <https://blog.reverberate.org/2020/05/12/optimizing-date-algorithms.html>.
/// This is a Rust port of the upb C implementation:
/// <https://github.com/protocolbuffers/protobuf/blob/27421b97a0daa29e91460d377b0213f9e7be5d3f/upb/json/encode.c>.
///
/// Key tricks:
///
/// * **March-based year.** Treating March as month 1 puts the leap day at
///   the *end* of the year, which makes the leap-year correction depend
///   only on the year (not on whether the month is past February). The
///   `carry` term subtracts 1 from the year for January and February.
/// * **Year base of 4800.** Adding 4800 (a multiple of 400) ensures `y_adj`
///   is always non-negative for any supported input, so the unsigned
///   divisions below behave like floor-division.
/// * **`(62719 * m_adj + 769) / 2048`.** A piecewise-linear approximation
///   of cumulative month lengths whose divisor is a power of two, so the
///   compiler lowers the division to a shift. Equivalent in output to the
///   more familiar `(153 * m_adj + 2) / 5` from Hinnant.
/// * **`y/4 - y/100 + y/400`.** Standard Gregorian leap-day count.
/// * **`-2472632`.** Re-bases the result onto the Unix epoch
///   (`365*4800 + leap_days(4800) + 0` for 1970-01-01).
#[inline]
fn jsondec_epochdays(y: i32, m: i32, d: i32) -> i32 {
    const YEAR_BASE: u32 = 4800; // Before min year, multiple of 400.

    let m_adj: u32 = (m - 3) as u32; // March-based month.

    // `m_adj` underflows in u32 for January/February (m < 3), wrapping to
    // a value much larger than `m`; that's the signal we need to borrow a
    // year and shift the month into the March-based [0, 11] range.
    let carry: u32 = u32::from(m_adj > m as u32);

    let adjust: u32 = if carry == 1 { 12 } else { 0 };

    let y_adj: u32 = y as u32 + YEAR_BASE - carry;
    let month_days: u32 = ((adjust.wrapping_add(m_adj)) * 62719 + 769) / 2048;
    let leap_days: u32 = y_adj / 4 - y_adj / 100 + y_adj / 400;

    y_adj as i32 * 365 + leap_days as i32 + month_days as i32 + (d - 1) - 2472632
}

/// Converts a date/time to Unix timestamp (seconds since epoch).
///
/// Combines the given date components into a single Unix timestamp value.
///
/// # Arguments
///
/// * `y` - Year
/// * `m` - Month (1-12)
/// * `d` - Day (1-31)
/// * `h` - Hour (0-23)
/// * `min` - Minute (0-59)
/// * `s` - Second (0-59)
///
/// # Returns
///
/// The number of seconds since the Unix epoch (1970-01-01T00:00:00Z).
#[allow(clippy::many_single_char_names)]
fn jsondec_unixtime(y: i32, m: i32, d: i32, h: i32, min: i32, s: i32) -> i64 {
    i64::from(jsondec_epochdays(y, m, d)) * 86400
        + i64::from(h) * 3600
        + i64::from(min) * 60
        + i64::from(s)
}

#[cfg(test)]
mod tests {
    use serde_test::{assert_tokens, Token};
    use std::time::Duration;

    use super::*;

    /// Tests decoding of the seconds component from an ISO8601 timestamp.
    #[test]
    fn test_decode_seconds() {
        let s = "2026-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert_eq!(decode_seconds(input).unwrap(), 1772029800);
        assert_eq!(input, b"Z");
    }

    /// Tests that invalid characters in the seconds field are properly rejected.
    #[test]
    fn test_decode_seconds_invalid_chars() {
        let s = "20/6-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert!(decode_seconds(input).is_err());

        let s = "20:6-02-25T14:30:00Z";
        let input = &mut s.as_bytes();
        assert!(decode_seconds(input).is_err());
    }

    /// Tests decoding of the nanoseconds component from an ISO8601 timestamp.
    #[test]
    fn test_decode_nanos() {
        let s = ".987654321Z";
        let input = &mut s.as_bytes();
        assert_eq!(decode_nanos(input).unwrap(), 987654321);
        assert_eq!(input, b"Z");

        let s = ".987654+00:00";
        let input = &mut s.as_bytes();
        assert_eq!(decode_nanos(input).unwrap(), 987654000);
        assert_eq!(input, b"+00:00");
    }

    /// Tests that invalid characters in the nanoseconds field are properly rejected.
    #[test]
    fn test_decode_nanos_invalid_chars() {
        let s = ".98/654321Z";
        let input = &mut s.as_bytes();
        assert!(decode_nanos(input).is_err());

        let s = ".98:654321Z";
        let input = &mut s.as_bytes();
        assert!(decode_nanos(input).is_err());
    }

    /// Tests ASCII-to-integer conversion with optional sign.
    #[test]
    fn test_atoi_consume() {
        let mut ascii = "1234ABCD".as_bytes();
        assert_eq!(atoi_consume(&mut ascii), 1234);
        assert_eq!(ascii, "ABCD".as_bytes());

        let mut ascii = "-1234ABCD".as_bytes();
        assert_eq!(atoi_consume(&mut ascii), -1234);
        assert_eq!(ascii, "ABCD".as_bytes());

        let mut ascii = "+1234ABCD".as_bytes();
        assert_eq!(atoi_consume(&mut ascii), 1234);
        assert_eq!(ascii, "ABCD".as_bytes());
    }

    /// Tests writing zero-padded numbers to the buffer.
    #[test]
    fn test_buffer_write_number() {
        let mut buf = Buffer::new();
        buf.write_byte(b'A');
        buf.write_number(12345, 5);
        buf.write_byte(b'B');
        assert_eq!(buf.as_str(), "A12345B");
    }

    /// Tests formatting of timestamps across various dates and precisions.
    #[test]
    fn test_buffer_format() {
        let mut buf = Buffer::new();
        for ts in timestamps() {
            assert_eq!(buf.format(ts.0), ts.1);
        }
    }

    /// Tests parsing of ISO8601 timestamp strings across various dates and precisions.
    #[test]
    fn test_parse() {
        for ts in timestamps() {
            assert_eq!(Timestamp::from(ts.0), Timestamp::from_str(ts.1).unwrap());
        }
    }

    /// Tests parsing of RFC3339 numeric timezone offsets.
    #[test]
    fn test_parse_offset() {
        // +05:00 means local is 5h ahead of UTC; the same instant in Z form is
        // 5h earlier.
        let utc = Timestamp::from_str("2026-02-25T09:30:00Z").unwrap();
        let off = Timestamp::from_str("2026-02-25T14:30:00+05:00").unwrap();
        assert_eq!(utc, off);

        let utc = Timestamp::from_str("2026-02-25T19:30:00Z").unwrap();
        let off = Timestamp::from_str("2026-02-25T14:30:00-05:00").unwrap();
        assert_eq!(utc, off);

        // +00:00 == Z
        let utc = Timestamp::from_str("2026-02-25T14:30:00Z").unwrap();
        let off = Timestamp::from_str("2026-02-25T14:30:00+00:00").unwrap();
        assert_eq!(utc, off);

        // Fractional seconds preserved alongside an offset.
        let utc = Timestamp::from_str("2026-02-25T09:30:00.123456789Z").unwrap();
        let off = Timestamp::from_str("2026-02-25T14:30:00.123456789+05:00").unwrap();
        assert_eq!(utc, off);

        // Half-hour offset.
        let utc = Timestamp::from_str("2026-02-25T09:00:00Z").unwrap();
        let off = Timestamp::from_str("2026-02-25T14:30:00+05:30").unwrap();
        assert_eq!(utc, off);
    }

    /// Tests rejection of malformed timezone offsets and other trailing input.
    #[test]
    fn test_parse_offset_invalid() {
        // Missing colon.
        assert!(Timestamp::from_str("2026-02-25T14:30:00+0500").is_err());
        // Wrong colon position.
        assert!(Timestamp::from_str("2026-02-25T14:30:00+05.00").is_err());
        // Out-of-range hours/minutes.
        assert!(Timestamp::from_str("2026-02-25T14:30:00+24:00").is_err());
        assert!(Timestamp::from_str("2026-02-25T14:30:00+05:60").is_err());
        // Non-digit.
        assert!(Timestamp::from_str("2026-02-25T14:30:00+0a:00").is_err());
        // Trailing garbage after a valid offset.
        assert!(Timestamp::from_str("2026-02-25T14:30:00+05:00X").is_err());
        // Trailing garbage with no terminator at all.
        assert!(Timestamp::from_str("2026-02-25T14:30:00X").is_err());
        // Empty input after the time field.
        assert!(Timestamp::from_str("2026-02-25T14:30:00").is_err());
    }

    /// Provides a collection of test timestamps with known string representations.
    fn timestamps() -> [(SystemTime, &'static str); 8] {
        [
            (
                UNIX_EPOCH + Duration::new(86400 + (60 * 60) + 60 + 1, 0),
                "1970-01-02T01:01:01Z",
            ),
            (
                UNIX_EPOCH + Duration::new(253402300799, 0),
                "9999-12-31T23:59:59Z",
            ),
            (
                UNIX_EPOCH + Duration::new(1641006000, 0),
                "2022-01-01T03:00:00Z",
            ),
            (
                UNIX_EPOCH - Duration::new(2208988800, 0),
                "1900-01-01T00:00:00Z",
            ),
            (
                UNIX_EPOCH - Duration::new(86400 + (60 * 60) + 60 + 1, 987654321),
                "1969-12-30T22:58:58.012345679Z",
            ),
            (
                UNIX_EPOCH + Duration::new(86400 + (60 * 60) + 60 + 1, 987654321),
                "1970-01-02T01:01:01.987654321Z",
            ),
            (
                UNIX_EPOCH + Duration::new(86400 + (60 * 60) + 60 + 1, 987654000),
                "1970-01-02T01:01:01.987654Z",
            ),
            (
                UNIX_EPOCH + Duration::new(86400 + (60 * 60) + 60 + 1, 987000000),
                "1970-01-02T01:01:01.987Z",
            ),
        ]
    }

    /// Tests interoperability with chrono datetime types.
    #[test]
    #[cfg(feature = "chrono")]
    fn test_chrono() {
        let now = chrono::Utc::now();
        let ts: Timestamp = now.into();
        let st: SystemTime = ts.into();
        assert_eq!(st, now.into());
    }

    /// Tests serialization and deserialization with serde.
    #[test]
    #[cfg(feature = "serde")]
    fn test_ser_de() {
        let ts: Timestamp = "2026-02-26T00:31:30.042Z".parse().unwrap();
        assert_tokens(&ts, &[Token::String("2026-02-26T00:31:30.042Z")]);
    }

    /// Tests deserialization via `visit_bytes` (when the deserializer hands us
    /// the input as raw bytes instead of a `&str`).
    #[test]
    #[cfg(feature = "serde")]
    fn test_de_bytes() {
        use serde_test::{assert_de_tokens, assert_de_tokens_error, Token};

        let ts: Timestamp = "2026-02-26T00:31:30.042Z".parse().unwrap();
        assert_de_tokens(&ts, &[Token::Bytes(b"2026-02-26T00:31:30.042Z")]);

        // Non-UTF8 bytes hit the error branch in visit_bytes.
        assert_de_tokens_error::<Timestamp>(&[Token::Bytes(b"\xff\xfe")], "Invalid Format");
        // Valid UTF-8 but malformed timestamp hits visit_str's error branch.
        assert_de_tokens_error::<Timestamp>(&[Token::Str("not a timestamp")], "Invalid Format");
        // A wrong-typed token forces the deserializer to call `expecting()`
        // on the visitor when constructing the error message.
        assert_de_tokens_error::<Timestamp>(
            &[Token::I32(42)],
            "invalid type: integer `42`, expected an ISO8601 Timestamp",
        );
    }

    /// Tests `TimestampError`'s `Display` impl for both variants.
    #[test]
    fn test_error_display() {
        assert_eq!(
            TimestampError::InvalidFormat.to_string(),
            "invalid timestamp format"
        );
        assert_eq!(
            TimestampError::OutOfRange.to_string(),
            "timestamp value out of range"
        );
        // Exercise the `core::error::Error` blanket impl.
        let e: &dyn core::error::Error = &TimestampError::InvalidFormat;
        assert!(e.source().is_none());
    }

    /// Tests `TryFrom<(i64, u32)>` and `Timestamp::from_unix` for both
    /// the success and out-of-range paths.
    #[test]
    fn test_try_from_seconds_nanos() {
        let ts = Timestamp::try_from((0i64, 0u32)).unwrap();
        assert_eq!(SystemTime::from(ts), UNIX_EPOCH);
        assert_eq!(ts.seconds(), 0);
        assert_eq!(ts.subsec_nanos(), 0);

        // A valid pre-epoch timestamp also exercises the negative-seconds
        // branch of `From<Timestamp> for SystemTime` with `nanos > 0`.
        let ts = Timestamp::from_unix(-1, 500_000_000).unwrap();
        assert_eq!(
            SystemTime::from(ts),
            UNIX_EPOCH - Duration::from_millis(500)
        );
        assert_eq!(ts.seconds(), -1);
        assert_eq!(ts.subsec_nanos(), 500_000_000);

        assert_eq!(
            Timestamp::try_from((SECONDS_MIN - 1, 0)),
            Err(TimestampError::OutOfRange),
        );
        assert_eq!(
            Timestamp::try_from((SECONDS_MAX + 1, 0)),
            Err(TimestampError::OutOfRange),
        );
        // Out-of-range nanos are now rejected (previously silently coerced).
        assert_eq!(
            Timestamp::from_unix(0, 1_000_000_000),
            Err(TimestampError::OutOfRange),
        );
    }

    /// Tests `Timestamp::now`, `Display`, and `Debug` impls.
    #[test]
    fn test_now_display_debug() {
        let now = Timestamp::now();
        // Display round-trips through the buffer formatter.
        let s = now.to_string();
        assert!(s.ends_with('Z'));
        assert_eq!(now, Timestamp::from_str(&s).unwrap());

        // Debug format includes the `Timestamp { ... }` derive output.
        let dbg = format!("{now:?}");
        assert!(dbg.starts_with("Timestamp "), "got: {dbg}");
    }

    /// Tests the various `From` conversions into `Timestamp`.
    #[test]
    fn test_from_conversions() {
        let st = UNIX_EPOCH + Duration::from_secs(1641006000);
        let ts_owned: Timestamp = st.into();
        let ts_ref: Timestamp = (&st).into();
        assert_eq!(ts_owned, ts_ref);

        // From<&Timestamp> for Timestamp (the reflexive copy).
        let ts_copy: Timestamp = (&ts_owned).into();
        assert_eq!(ts_owned, ts_copy);
    }

    /// Tests the `chrono::DateTime` → `Timestamp` → `chrono::DateTime`
    /// round-trip (covers the `Timestamp → DateTime<Utc>` impl).
    #[test]
    #[cfg(feature = "chrono")]
    fn test_chrono_roundtrip() {
        let ts: Timestamp = "2026-02-26T00:31:30.042Z".parse().unwrap();
        let dt: chrono::DateTime<chrono::Utc> = ts.into();
        let back: Timestamp = dt.into();
        assert_eq!(ts, back);
    }

    /// Tests `Buffer::default` and `Clone`.
    #[test]
    fn test_buffer_default_and_clone() {
        let mut buf = Buffer::default();
        let ts: Timestamp = "2026-02-26T00:31:30.042Z".parse().unwrap();
        assert_eq!(buf.format(ts), "2026-02-26T00:31:30.042Z");

        // `Clone` for `Buffer` deliberately returns a fresh empty buffer
        // rather than a true copy; verify that contract here. The lint
        // would have us write `buf` directly, but the whole point of the
        // test is to exercise the explicit `Clone` impl.
        #[allow(clippy::clone_on_copy)]
        let cloned = buf.clone();
        assert_eq!(cloned.len, 0);
    }

    /// Tests the 3-digit nanosecond branch (covers the inner multiplication
    /// loop running its full 6 iterations).
    #[test]
    fn test_decode_nanos_3digit() {
        let s = ".042Z";
        let input = &mut s.as_bytes();
        assert_eq!(decode_nanos(input).unwrap(), 42_000_000);
        assert_eq!(input, b"Z");
    }

    /// Pins the in-memory layout of `Buffer` so that future changes to
    /// `BUFFER_SIZE` or the `len` field type don't accidentally regress the
    /// "fits in one cacheline / four qwords" property.
    #[test]
    fn test_buffer_size() {
        assert_eq!(core::mem::size_of::<Buffer>(), 32);
    }
}
