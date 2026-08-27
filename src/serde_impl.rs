//! Optional `serde` integration.
//!
//! Enabled by the `serde` Cargo feature. Serializes a
//! [`Timestamp`](crate::Timestamp) as its RFC3339 string form via the
//! stack-allocated [`Buffer`](crate::Buffer) (no allocation on the encode
//! path), and deserializes from any string-shaped token by routing through
//! `Timestamp::from_str`.

use core::{fmt, str::FromStr};

use serde_core::de::Visitor;
use serde_core::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{Buffer, Timestamp};

impl Serialize for Timestamp {
    #[inline]
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut buf = Buffer::new();
        serializer.serialize_str(buf.format(self))
    }
}

impl<'de> Deserialize<'de> for Timestamp {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TsVisitor;

        impl Visitor<'_> for TsVisitor {
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
