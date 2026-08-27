# rfc3339-fast

[![crates.io](https://img.shields.io/crates/v/rfc3339-fast.svg)](https://crates.io/crates/rfc3339-fast)
[![docs.rs](https://docs.rs/rfc3339-fast/badge.svg)](https://docs.rs/rfc3339-fast)
[![license](https://img.shields.io/crates/l/rfc3339-fast.svg)](LICENSE)

A high-performance Rust library for parsing and formatting RFC3339 timestamps with support for nanosecond precision.

## Features

- **Wide timestamp range**: Supports timestamps from year 1 to year 9999
- **Nanosecond precision**: Up to 9 decimal places of fractional seconds
- **Zero-copy formatting**: Reusable stack-allocated buffer for efficient string generation
- **SystemTime integration**: Direct conversion to/from `std::time::SystemTime`
- **Chrono support**: Optional integration with the `chrono` crate via the `chrono` feature
- **Serde support**: Optional serialization/deserialization support via the `serde` feature
- **Postgres support**: Optional `tokio-postgres` / `postgres` integration via the `postgres` feature, mapping to `TIMESTAMP WITH TIME ZONE`
- **SIMD acceleration**: Automatic optimization for platforms supporting SSSE3 (x86/x86_64) or NEON (ARM)
- **Efficient parsing**: Fast parsing of RFC3339/ISO8601 timestamps

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
rfc3339-fast = "0.2"
```

### Optional Features

The default feature set is `["std", "serde"]`. Available features:

- `std` (default) — enables `std::time`-dependent APIs (`Timestamp::now`,
  conversions to/from `std::time::SystemTime`, and the `std::error::Error`
  impl for `TimestampError`). Disabling default features makes the crate
  `#![no_std]`.
- `serde` (default) — enables `Serialize`/`Deserialize` for `Timestamp`.
- `chrono` — enables conversions from `chrono::DateTime<Tz>` to `Timestamp`
  (implies `std`).
- `postgres` — enables `ToSql` / `FromSql` for `Timestamp` so it can be used
  directly with `tokio-postgres`, `postgres`, and connection pools built on
  top (such as `deadpool-postgres`). The value maps to Postgres
  `TIMESTAMP WITH TIME ZONE` (`timestamptz`). Sub-microsecond nanoseconds
  are truncated to microsecond precision on encode; out-of-range values
  yield `TimestampError::OutOfRange` on decode (implies `std`).

For example, to enable `chrono` alongside the defaults:

```toml
[dependencies]
rfc3339-fast = { version = "0.1", features = ["chrono"] }
```

For a `no_std` build without `serde`:

```toml
[dependencies]
rfc3339-fast = { version = "0.1", default-features = false }
```

## Usage

### Parsing Timestamps

Parse an RFC3339/ISO8601 timestamp string:

```rust
use std::str::FromStr;
use rfc3339_fast::Timestamp;

let ts = Timestamp::from_str("2026-02-25T14:30:00Z")?;
```

### Formatting Timestamps

Format a timestamp to an RFC3339 string:

```rust
use rfc3339_fast::{Timestamp, Buffer};

let ts = Timestamp::now();
let mut buf = Buffer::new();
let formatted = buf.format(ts);
println!("{}", formatted);  // Prints: 2026-02-25T...Z
```

### Constructing from raw components

Build a `Timestamp` from Unix seconds and nanoseconds, or read those
components back out:

```rust
use rfc3339_fast::Timestamp;

let ts = Timestamp::from_unix(1_641_006_000, 250_000_000)?;
assert_eq!(ts.seconds(), 1_641_006_000);
assert_eq!(ts.subsec_nanos(), 250_000_000);
```

`from_unix` returns `TimestampError::OutOfRange` if the seconds are
outside year 1..=9999 or `nanos >= 1_000_000_000`. The same validation is
also available via `TryFrom<(i64, u32)>`.

### Working with SystemTime

Convert between `SystemTime` and `Timestamp`:

```rust
use std::time::SystemTime;
use rfc3339_fast::Timestamp;

let sys_time = SystemTime::now();
let ts = Timestamp::from(sys_time);
```

### With Chrono (requires `chrono` feature)

```rust
use rfc3339_fast::Timestamp;

let now = chrono::Utc::now();
let ts: Timestamp = now.into();
```

### With Serde (requires `serde` feature)

```rust
use serde::{Deserialize, Serialize};
use rfc3339_fast::Timestamp;

#[derive(Serialize, Deserialize)]
struct Event {
    name: String,
    timestamp: Timestamp,
}

let json = r#"{"name": "event", "timestamp": "2026-02-25T14:30:00Z"}"#;
let event: Event = serde_json::from_str(json)?;
```

### With tokio-postgres (requires `postgres` feature)

`Timestamp` implements `ToSql` and `FromSql` and maps to Postgres
`TIMESTAMP WITH TIME ZONE` (`timestamptz`), so it can be used directly as
a parameter or row column with `tokio-postgres`:

```rust,ignore
use rfc3339_fast::Timestamp;
use tokio_postgres::types::Type;

let ts = Timestamp::now();
let row = client
    .query_one("SELECT $1::timestamptz", &[&ts])
    .await?;
let back: Timestamp = row.get(0);
```

Postgres stores `timestamptz` at microsecond precision, so any
sub-microsecond nanoseconds in a `Timestamp` are truncated on encode.

## Timestamp Format

The library uses the ISO8601/RFC3339 format:

```
YYYY-MM-DDTHH:mm:ss[.nnn]Z
```

Where:
- **YYYY** - Year (0001-9999)
- **MM** - Month (01-12)
- **DD** - Day (01-31)
- **HH** - Hour (00-23)
- **mm** - Minute (00-59)
- **ss** - Second (00-59)
- **[.nnn]** - Optional fractional seconds (3, 6, or 9 digits for millisecond, microsecond, or nanosecond precision)
- **Z** - UTC timezone indicator

## Performance

The `Buffer` type is designed for efficient formatting without heap allocations:

```rust
let mut buf = Buffer::new();

// Format many timestamps using the same buffer
for ts in timestamps {
    let formatted = buf.format(ts);
    // Use the formatted string
}
```

The buffer uses a fixed stack-allocated array, making it suitable for high-throughput scenarios.

### SIMD acceleration

This crate uses SIMD-accelerated parsing paths for x86/x86_64 (SSSE3) and
AArch64 (NEON). Selection is done at **compile time** via `#[cfg(target_feature
= ...)]`, not runtime detection.

- **AArch64 (Apple Silicon, modern ARM servers):** NEON is part of the
  baseline target features, so the SIMD path is enabled automatically.
- **x86_64:** SSSE3 is *not* part of the default `x86_64` baseline, so a
  stock `cargo build --release` will use the scalar fallback even on a CPU
  that supports SSSE3. To opt in, build with one of:

  ```bash
  RUSTFLAGS="-C target-cpu=native"      cargo build --release
  RUSTFLAGS="-C target-feature=+ssse3"  cargo build --release
  ```

  Or set this in `.cargo/config.toml` for your workspace:

  ```toml
  [build]
  rustflags = ["-C", "target-cpu=native"]
  ```

  Note that binaries built this way will not run on CPUs that lack the
  enabled features. If you need a single binary that works everywhere and
  still picks the fast path on capable CPUs, you'll need to build with the
  scalar baseline (the default).

## Building

To build the project:

```bash
cargo build
```

With optimizations:

```bash
cargo build --release
```

## Testing

Run the test suite:

```bash
cargo test
```

With all features:

```bash
cargo test --all-features
```

## Benchmarking

Run the criterion benchmarks:

```bash
cargo bench
```

This will measure parsing and formatting performance.

## Error Handling

The library provides a `TimestampError` enum for error handling:

```rust
use rfc3339_fast::{Timestamp, TimestampError};
use std::str::FromStr;

match Timestamp::from_str("invalid-timestamp") {
    Ok(ts) => println!("Parsed: {:?}", ts),
    Err(TimestampError::InvalidFormat) => println!("Invalid format"),
    Err(TimestampError::OutOfRange) => println!("Timestamp out of range"),
}
```

## Limitations

- Parses UTC (`Z`) and numeric offsets (`+HH:MM` / `-HH:MM`); offsets are
  normalized to UTC on parse.
- Formatting always emits UTC (`Z`); the original offset is not preserved on
  round-trip.
- Fractional seconds must be exactly 3, 6, or 9 digits.

## Algorithm Notes

- Date calculations use the Fliegel/Van Flandern algorithm for efficient conversion between calendar dates and day counts
- SIMD implementations leverage platform-specific CPU instructions for faster digit parsing when available

## No-std Support

The crate is `#![no_std]`-compatible when built with
`default-features = false`. In that mode `Timestamp` operates entirely on
`(i64 seconds, u32 nanos)`; the `std`-only APIs (`Timestamp::now`,
`From<SystemTime>` / `From<&SystemTime>`, `From<Timestamp> for SystemTime`,
and the `std::error::Error` impl for `TimestampError`) are gated behind the
`std` feature.

## License

Licensed under the [BSD 2-Clause License](LICENSE).

## Contributing

Contributions are welcome! Please ensure all tests pass and add tests for new features:

```bash
cargo test --all-features
```
