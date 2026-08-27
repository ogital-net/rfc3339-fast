//! Optional `tokio-postgres` / `postgres` integration.
//!
//! Enabled by the `postgres` Cargo feature (which pulls in
//! [`postgres-types`]). Provides [`ToSql`] and [`FromSql`] impls so a
//! [`Timestamp`](crate::Timestamp) can be used directly as a query parameter
//! or row value with [`tokio_postgres`], [`postgres`], or any of the
//! connection pools built on top of them (`deadpool-postgres`, etc.).
//!
//! ## Postgres type
//!
//! `Timestamp` maps to **`TIMESTAMP WITH TIME ZONE`** (`timestamptz`) only.
//! We deliberately do *not* accept plain `TIMESTAMP`, because Postgres
//! treats it as session-local time and silently converting through it
//! would produce surprising results.
//!
//! ## Binary format
//!
//! Postgres stores `timestamptz` as 8 bytes of big-endian microseconds since
//! the Postgres epoch (2000-01-01T00:00:00Z). On encode we truncate
//! sub-microsecond nanoseconds (the `Timestamp`'s 9-digit nanos are reduced
//! to the lower 6 digits). On decode we recover `nanos` as
//! `micros.rem_euclid(1_000_000) * 1_000`, so the bottom 3 digits of any
//! stored nanosecond value are always zero.
//!
//! ## Range policy
//!
//! On decode, a value outside the `Timestamp` range (year 1..=9999) returns
//! an error rather than saturating. Saturation would mask data corruption
//! or migration bugs from database callers, which we consider the wrong
//! tradeoff for the I/O path.
//!
//! [`tokio_postgres`]: https://docs.rs/tokio-postgres

use bytes::BytesMut;
use postgres_types::{FromSql, IsNull, ToSql, Type};
use std::error::Error;

use crate::{Timestamp, TimestampError};

/// Seconds from the Unix epoch (1970-01-01T00:00:00Z) to the Postgres
/// `timestamptz` epoch (2000-01-01T00:00:00Z). Postgres stores the value
/// as microseconds *since* this epoch.
pub(crate) const POSTGRES_EPOCH_UNIX_SECS: i64 = 946_684_800;

impl ToSql for Timestamp {
    fn to_sql(
        &self,
        _ty: &Type,
        out: &mut BytesMut,
    ) -> Result<IsNull, Box<dyn Error + Sync + Send>> {
        // Truncate sub-microsecond nanos. `Timestamp`'s invariant guarantees
        // `subsec_nanos() < 1_000_000_000`, so the divide is exact and the
        // resulting `micros` fits in `i64` (Postgres's own `timestamptz`
        // range is the binding constraint, and we error upstream of that
        // via the constructor invariant).
        let micros = (self.seconds() - POSTGRES_EPOCH_UNIX_SECS) * 1_000_000
            + i64::from(self.subsec_nanos() / 1_000);
        out.extend_from_slice(&micros.to_be_bytes());
        Ok(IsNull::No)
    }

    postgres_types::accepts!(TIMESTAMPTZ);
    postgres_types::to_sql_checked!();
}

impl<'a> FromSql<'a> for Timestamp {
    fn from_sql(
        _ty: &Type,
        raw: &'a [u8],
    ) -> Result<Self, Box<dyn Error + Sync + Send>> {
        if raw.len() != 8 {
            return Err(format!(
                "invalid timestamptz length: expected 8 bytes, got {}",
                raw.len()
            )
            .into());
        }
        let micros = i64::from_be_bytes(raw.try_into().unwrap());
        // Euclidean division so `nanos` is always in `[0, 1_000_000_000)`
        // even when `micros` is negative (i.e. pre-2000 values such as
        // anything from the 1960s or earlier).
        let secs = micros.div_euclid(1_000_000) + POSTGRES_EPOCH_UNIX_SECS;
        let nanos = (micros.rem_euclid(1_000_000) as u32) * 1_000;
        Timestamp::from_unix(secs, nanos)
            .map_err(|e: TimestampError| Box::new(e) as Box<dyn Error + Sync + Send>)
    }

    postgres_types::accepts!(TIMESTAMPTZ);
}
