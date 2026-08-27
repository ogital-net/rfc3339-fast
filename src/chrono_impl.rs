//! Optional `chrono` integration.
//!
//! Enabled by the `chrono` Cargo feature. Provides infallible conversions
//! between [`Timestamp`](crate::Timestamp) and `chrono::DateTime`. Out-of-range
//! chrono values are **saturated** to the year 1 / year 9999 bounds of
//! `Timestamp`, matching the contract used by `From<SystemTime>`.

use crate::{Timestamp, SECONDS_MAX, SECONDS_MIN};

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

impl From<Timestamp> for chrono::DateTime<chrono::Utc> {
    fn from(value: Timestamp) -> Self {
        chrono::DateTime::<chrono::Utc>::from_timestamp(value.seconds, value.nanos)
            .expect("Timestamp out of range for chrono::DateTime")
    }
}
