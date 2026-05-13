use std::str::FromStr;
use std::{hint::black_box, time::SystemTime};

use criterion::{Criterion, criterion_group, criterion_main};
use iso8601_timestamp::Timestamp as IsoTimestamp;
use rfc3339_fast::{Buffer, Timestamp};

fn criterion_benchmark(c: &mut Criterion) {
    let mut format_group = c.benchmark_group("format");

    format_group.bench_function("format", |b| {
        let mut buf = Buffer::new();
        let now = SystemTime::now();

        b.iter(move || {
            let now = black_box(now);
            let fmt = buf.format(now);
            black_box(fmt);
        });
    });

    format_group.bench_function("iso8601", |b| {
        let now = SystemTime::now();
        b.iter(move || {
            let now = black_box(now);
            let fmt_nanos = IsoTimestamp::from(now).format_nanoseconds();
            let fmt: &str = &fmt_nanos;
            black_box(fmt);
        });
    });

    format_group.finish();

    let mut parse_group = c.benchmark_group("parse");

    parse_group.bench_function("parse", |b| {
        let mut buf = Buffer::new();

        let ts_str = buf.format(Timestamp::now());

        b.iter(move || {
            let ts_str = black_box(ts_str);
            let st: SystemTime = Timestamp::from_str(ts_str).unwrap().into();
            black_box(st);
        });
    });

    parse_group.bench_function("iso8601", |b| {
        let mut buf = Buffer::new();

        let ts_str = buf.format(Timestamp::now());

        b.iter(move || {
            let ts_str = black_box(ts_str);
            let st: SystemTime = IsoTimestamp::parse(ts_str).unwrap().into();
            black_box(st);
        });
    });

    parse_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
