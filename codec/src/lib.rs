//! Binary codec for compact little-endian serialization.
//!
//! Provides the [`Codec`] trait for encoding Rust values to `Vec<u8>` and
//! decoding them back. All multi-byte integers and floats use little-endian
//! byte order.
//!
//! # Examples
//!
//! ```
//! use codec::*;
//!
//! let value: u32 = 42;
//! let mut buf = Vec::new();
//! value.encode(&mut buf);
//! let (decoded, len) = u32::decode(&buf).unwrap();
//! assert_eq!(decoded, 42);
//! assert_eq!(len, 4);
//! ```

extern crate self as codec;

pub use codec_derive::Codec;

use std::fmt;

// -- error type --

/// Errors that can occur during decoding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CodecError {
    /// Buffer too short for the expected type.
    NotEnoughData {
        /// Bytes needed.
        expected: usize,
        /// Bytes available.
        available: usize,
    },
    /// Byte value is not a valid `bool` (must be `0x00` or `0x01`).
    InvalidBool(u8),
    /// Byte sequence is not valid UTF-8.
    InvalidUtf8,
    /// Enum variant index is not recognized.
    InvalidVariant(u32),
}

impl fmt::Display for CodecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CodecError::NotEnoughData { expected, available } => {
                write!(f, "not enough data: expected {} bytes, got {}", expected, available)
            }
            CodecError::InvalidBool(v) => write!(f, "invalid bool byte: 0x{:02X}", v),
            CodecError::InvalidUtf8 => write!(f, "invalid UTF-8"),
            CodecError::InvalidVariant(idx) => write!(f, "invalid enum variant index: {}", idx),
        }
    }
}

// -- trait --

/// Binary serialization trait.
///
/// Types implementing `Codec` can be encoded to a byte buffer and decoded
/// back. The encoding is compact little-endian binary with no framing or
/// field names.
pub trait Codec: Sized {
    /// Append the binary representation of `self` to `buf`.
    fn encode(&self, buf: &mut Vec<u8>);

    /// Decode a value from the start of `buf`.
    ///
    /// Returns the decoded value and the number of bytes consumed.
    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError>;
}

// -- primitive impls --

impl Codec for bool {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(if *self { 0x01 } else { 0x00 });
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        if buf.is_empty() {
            return Err(CodecError::NotEnoughData { expected: 1, available: 0 });
        }
        match buf[0] {
            0x00 => Ok((false, 1)),
            0x01 => Ok((true, 1)),
            other => Err(CodecError::InvalidBool(other)),
        }
    }
}

impl Codec for u8 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(*self);
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        if buf.is_empty() {
            return Err(CodecError::NotEnoughData { expected: 1, available: 0 });
        }
        Ok((buf[0], 1))
    }
}

impl Codec for i8 {
    fn encode(&self, buf: &mut Vec<u8>) {
        buf.push(*self as u8);
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        if buf.is_empty() {
            return Err(CodecError::NotEnoughData { expected: 1, available: 0 });
        }
        Ok((buf[0] as i8, 1))
    }
}

macro_rules! impl_codec_for_numeric {
    ($($t:ty),+) => {
        $(
            impl Codec for $t {
                fn encode(&self, buf: &mut Vec<u8>) {
                    buf.extend_from_slice(&self.to_le_bytes());
                }

                fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
                    const SIZE: usize = std::mem::size_of::<$t>();
                    if buf.len() < SIZE {
                        return Err(CodecError::NotEnoughData { expected: SIZE, available: buf.len() });
                    }
                    let bytes: [u8; SIZE] = buf[..SIZE].try_into().unwrap();
                    Ok((<$t>::from_le_bytes(bytes), SIZE))
                }
            }
        )+
    };
}

impl_codec_for_numeric!(u16, u32, u64, u128, i16, i32, i64, i128, f32, f64);

// -- string --

impl Codec for String {
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u64).encode(buf);
        buf.extend_from_slice(self.as_bytes());
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        let (len, mut off) = u64::decode(buf)?;
        let len = len as usize;
        if buf.len() - off < len {
            return Err(CodecError::NotEnoughData { expected: off + len, available: buf.len() });
        }
        let s = String::from_utf8(buf[off..off + len].to_vec())
            .map_err(|_| CodecError::InvalidUtf8)?;
        off += len;
        Ok((s, off))
    }
}

// -- vec --

impl<T: Codec> Codec for Vec<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        (self.len() as u64).encode(buf);
        for item in self {
            item.encode(buf);
        }
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        let (count, mut off) = u64::decode(buf)?;
        let count = count as usize;
        let mut vec = Vec::with_capacity(count);
        for _ in 0..count {
            let (item, n) = T::decode(&buf[off..])?;
            vec.push(item);
            off += n;
        }
        Ok((vec, off))
    }
}

// -- option --

impl<T: Codec> Codec for Option<T> {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            None => 0u8.encode(buf),
            Some(v) => {
                1u8.encode(buf);
                v.encode(buf);
            }
        }
    }

    fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
        let (tag, mut off) = u8::decode(buf)?;
        match tag {
            0 => Ok((None, off)),
            1 => {
                let (v, n) = T::decode(&buf[off..])?;
                off += n;
                Ok((Some(v), off))
            }
            _ => Err(CodecError::InvalidBool(tag)),
        }
    }
}

// -- tuples --

macro_rules! impl_codec_for_tuple {
    ($($idx:tt : $T:ident / $v:ident),+) => {
        impl<$($T: Codec),+> Codec for ($($T,)+) {
            fn encode(&self, buf: &mut Vec<u8>) {
                $(self.$idx.encode(buf);)+
            }

            #[allow(unused_assignments)]
            fn decode(buf: &[u8]) -> Result<(Self, usize), CodecError> {
                let mut _off = 0usize;
                $(
                    let ($v, _n) = $T::decode(&buf[_off..])?;
                    _off += _n;
                )+
                Ok((($($v,)+), _off))
            }
        }
    };
}

impl_codec_for_tuple!(0: A/a, 1: B/b);
impl_codec_for_tuple!(0: A/a, 1: B/b, 2: C/c);
impl_codec_for_tuple!(0: A/a, 1: B/b, 2: C/c, 3: D/d);
impl_codec_for_tuple!(0: A/a, 1: B/b, 2: C/c, 3: D/d, 4: E/e);
impl_codec_for_tuple!(0: A/a, 1: B/b, 2: C/c, 3: D/d, 4: E/e, 5: F/ff);

// -- tests --

#[cfg(test)]
mod tests {
    use super::*;

    // -- bool --

    #[test]
    fn test_bool_roundtrip() {
        for &val in &[true, false] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            let (decoded, len) = bool::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 1);
        }
    }

    #[test]
    fn test_bool_encoding() {
        let mut buf = Vec::new();
        false.encode(&mut buf);
        assert_eq!(buf, [0x00]);
        buf.clear();
        true.encode(&mut buf);
        assert_eq!(buf, [0x01]);
    }

    #[test]
    fn test_bool_invalid_byte() {
        assert_eq!(bool::decode(&[0x02]), Err(CodecError::InvalidBool(0x02)));
        assert_eq!(bool::decode(&[0xFF]), Err(CodecError::InvalidBool(0xFF)));
    }

    #[test]
    fn test_bool_empty_buffer() {
        assert_eq!(bool::decode(&[]), Err(CodecError::NotEnoughData { expected: 1, available: 0 }));
    }

    // -- u8 / i8 --

    #[test]
    fn test_u8_roundtrip() {
        for val in [0u8, 1, 127, 255] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            let (decoded, len) = u8::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 1);
        }
    }

    #[test]
    fn test_i8_roundtrip() {
        for val in [0i8, 1, -1, 127, -128] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            let (decoded, len) = i8::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 1);
        }
    }

    // -- multi-byte integers --

    macro_rules! test_numeric_roundtrip {
        ($name:ident, $t:ty, $size:expr, $($val:expr),+) => {
            #[test]
            fn $name() {
                for val in [$($val as $t),+] {
                    let mut buf = Vec::new();
                    val.encode(&mut buf);
                    assert_eq!(buf.len(), $size);
                    let (decoded, len) = <$t>::decode(&buf).unwrap();
                    assert_eq!(decoded, val);
                    assert_eq!(len, $size);
                }
            }
        };
    }

    test_numeric_roundtrip!(test_u16_roundtrip, u16, 2, 0, 1, 0xFF, 0xFFFF);
    test_numeric_roundtrip!(test_u32_roundtrip, u32, 4, 0, 1, 0xDEAD_BEEF);
    test_numeric_roundtrip!(test_u64_roundtrip, u64, 8, 0, 1, 0xDEAD_BEEF_CAFE_BABE);
    test_numeric_roundtrip!(test_u128_roundtrip, u128, 16, 0, 1, 0xDEAD_BEEF_CAFE_BABE);
    test_numeric_roundtrip!(test_i16_roundtrip, i16, 2, 0, 1, -1, 32767, -32768);
    test_numeric_roundtrip!(test_i32_roundtrip, i32, 4, 0, 1, -1, 2147483647);
    test_numeric_roundtrip!(test_i64_roundtrip, i64, 8, 0, 1, -1);
    test_numeric_roundtrip!(test_i128_roundtrip, i128, 16, 0, 1, -1);

    // -- floats --

    #[test]
    fn test_f32_roundtrip() {
        for val in [0.0f32, 1.0, -1.0, std::f32::consts::PI, f32::INFINITY, f32::NEG_INFINITY] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            assert_eq!(buf.len(), 4);
            let (decoded, len) = f32::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 4);
        }
    }

    #[test]
    fn test_f32_nan_roundtrip() {
        let mut buf = Vec::new();
        f32::NAN.encode(&mut buf);
        let (decoded, _) = f32::decode(&buf).unwrap();
        assert!(decoded.is_nan());
    }

    #[test]
    fn test_f64_roundtrip() {
        for val in [0.0f64, 1.0, -1.0, std::f64::consts::PI, f64::INFINITY] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            assert_eq!(buf.len(), 8);
            let (decoded, len) = f64::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 8);
        }
    }

    // -- little-endian byte order --

    #[test]
    fn test_u32_little_endian() {
        let mut buf = Vec::new();
        0x04030201u32.encode(&mut buf);
        assert_eq!(buf, [0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_f32_little_endian() {
        let mut buf = Vec::new();
        1.0f32.encode(&mut buf);
        assert_eq!(buf, 1.0f32.to_le_bytes());
    }

    // -- truncated buffer errors --

    #[test]
    fn test_u8_empty_buffer() {
        assert_eq!(u8::decode(&[]), Err(CodecError::NotEnoughData { expected: 1, available: 0 }));
    }

    #[test]
    fn test_u16_truncated() {
        assert_eq!(u16::decode(&[0x01]), Err(CodecError::NotEnoughData { expected: 2, available: 1 }));
    }

    #[test]
    fn test_u32_truncated() {
        assert_eq!(u32::decode(&[0x01, 0x02]), Err(CodecError::NotEnoughData { expected: 4, available: 2 }));
    }

    #[test]
    fn test_u64_truncated() {
        assert_eq!(u64::decode(&[0x01]), Err(CodecError::NotEnoughData { expected: 8, available: 1 }));
    }

    #[test]
    fn test_u128_truncated() {
        assert_eq!(u128::decode(&[0x01; 8]), Err(CodecError::NotEnoughData { expected: 16, available: 8 }));
    }

    #[test]
    fn test_f32_truncated() {
        assert_eq!(f32::decode(&[0x01, 0x02]), Err(CodecError::NotEnoughData { expected: 4, available: 2 }));
    }

    #[test]
    fn test_f64_truncated() {
        assert_eq!(f64::decode(&[0x01; 4]), Err(CodecError::NotEnoughData { expected: 8, available: 4 }));
    }

    // -- string --

    #[test]
    fn test_string_empty() {
        let val = String::new();
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 8); // just the u64 length prefix
        let (decoded, len) = String::decode(&buf).unwrap();
        assert_eq!(decoded, "");
        assert_eq!(len, 8);
    }

    #[test]
    fn test_string_ascii() {
        let val = "hello".to_string();
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 8 + 5);
        let (decoded, len) = String::decode(&buf).unwrap();
        assert_eq!(decoded, "hello");
        assert_eq!(len, 13);
    }

    #[test]
    fn test_string_unicode() {
        let val = "héllo 世界 🦀".to_string();
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = String::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_string_invalid_utf8() {
        let mut buf = Vec::new();
        2u64.encode(&mut buf);
        buf.extend_from_slice(&[0xFF, 0xFE]);
        assert_eq!(String::decode(&buf), Err(CodecError::InvalidUtf8));
    }

    #[test]
    fn test_string_truncated_length() {
        assert!(String::decode(&[0x01, 0x02]).is_err());
    }

    #[test]
    fn test_string_truncated_data() {
        let mut buf = Vec::new();
        10u64.encode(&mut buf);
        buf.extend_from_slice(b"hi");
        assert!(String::decode(&buf).is_err());
    }

    // -- vec --

    #[test]
    fn test_vec_empty() {
        let val: Vec<u32> = vec![];
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 8);
        let (decoded, len) = Vec::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
        assert_eq!(len, 8);
    }

    #[test]
    fn test_vec_u32() {
        let val = vec![1u32, 2, 3];
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 8 + 3 * 4);
        let (decoded, _) = Vec::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_vec_string() {
        let val = vec!["hello".to_string(), "world".to_string()];
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Vec::<String>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_vec_truncated() {
        let mut buf = Vec::new();
        3u64.encode(&mut buf);
        1u32.encode(&mut buf);
        // only 1 element but claimed 3
        assert!(Vec::<u32>::decode(&buf).is_err());
    }

    // -- option --

    #[test]
    fn test_option_none() {
        let val: Option<u32> = None;
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf, [0x00]);
        let (decoded, len) = Option::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, None);
        assert_eq!(len, 1);
    }

    #[test]
    fn test_option_some() {
        let val = Some(42u32);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 1 + 4);
        let (decoded, len) = Option::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, Some(42));
        assert_eq!(len, 5);
    }

    #[test]
    fn test_option_some_string() {
        let val = Some("hello".to_string());
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Option::<String>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    // -- tuples --

    #[test]
    fn test_tuple2() {
        let val = (42u32, true);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = <(u32, bool)>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
        assert_eq!(len, 5);
    }

    #[test]
    fn test_tuple3() {
        let val = (1u8, 2u16, 3u32);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = <(u8, u16, u32)>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_tuple4() {
        let val = (1u8, 2u8, 3u8, 4u8);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = <(u8, u8, u8, u8)>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_tuple5() {
        let val = (1u8, 2u8, 3u8, 4u8, 5u8);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = <(u8, u8, u8, u8, u8)>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_tuple6() {
        let val = (1u8, 2u8, 3u8, 4u8, 5u8, 6u8);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = <(u8, u8, u8, u8, u8, u8)>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    // -- nested compound types --

    #[test]
    fn test_vec_option_string() {
        let val: Vec<Option<String>> = vec![
            Some("hello".to_string()),
            None,
            Some("world".to_string()),
        ];
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Vec::<Option<String>>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_option_vec_u32() {
        let val: Option<Vec<u32>> = Some(vec![1, 2, 3]);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Option::<Vec<u32>>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    // -- derive: structs --

    #[derive(Debug, PartialEq, Codec)]
    struct Point {
        x: f32,
        y: f32,
    }

    #[test]
    fn test_derive_struct_simple() {
        let val = Point { x: 1.0, y: 2.0 };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 8);
        let (decoded, len) = Point::decode(&buf).unwrap();
        assert_eq!(decoded, val);
        assert_eq!(len, 8);
    }

    #[derive(Debug, PartialEq, Codec)]
    struct Person {
        name: String,
        age: u32,
        scores: Vec<f64>,
    }

    #[test]
    fn test_derive_struct_compound_fields() {
        let val = Person {
            name: "Alice".to_string(),
            age: 30,
            scores: vec![95.5, 88.0],
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Person::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[derive(Debug, PartialEq, Codec)]
    struct Pair<T> {
        a: T,
        b: T,
    }

    #[test]
    fn test_derive_struct_generic() {
        let val = Pair { a: 10u32, b: 20u32 };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Pair::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_derive_struct_generic_string() {
        let val = Pair { a: "x".to_string(), b: "y".to_string() };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Pair::<String>::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[derive(Debug, PartialEq, Codec)]
    struct Nested {
        point: Point,
        label: String,
    }

    #[test]
    fn test_derive_struct_nested() {
        let val = Nested {
            point: Point { x: 3.0, y: 4.0 },
            label: "origin".to_string(),
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Nested::decode(&buf).unwrap();
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_derive_struct_truncated() {
        let val = Point { x: 1.0, y: 2.0 };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert!(Point::decode(&buf[..4]).is_err());
    }

    // -- derive: enums --

    #[derive(Debug, PartialEq, Codec)]
    enum Color {
        Red,
        Green,
        Blue,
    }

    #[test]
    fn test_derive_enum_unit_variants() {
        for (val, expected_idx) in [(Color::Red, 0u32), (Color::Green, 1), (Color::Blue, 2)] {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            assert_eq!(buf.len(), 4);
            let (decoded, len) = Color::decode(&buf).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(len, 4);
            // verify wire format: first 4 bytes are the variant index in LE
            assert_eq!(&buf[..4], &expected_idx.to_le_bytes());
        }
    }

    #[derive(Debug, PartialEq, Codec)]
    enum Shape {
        Circle(f32),
        Rectangle(f32, f32),
    }

    #[test]
    fn test_derive_enum_tuple_variants() {
        let val = Shape::Circle(5.0);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        assert_eq!(buf.len(), 4 + 4); // u32 index + f32
        let (decoded, _) = Shape::decode(&buf).unwrap();
        assert_eq!(decoded, val);

        let val2 = Shape::Rectangle(3.0, 4.0);
        buf.clear();
        val2.encode(&mut buf);
        assert_eq!(buf.len(), 4 + 4 + 4); // u32 index + 2 × f32
        let (decoded2, _) = Shape::decode(&buf).unwrap();
        assert_eq!(decoded2, val2);
    }

    #[derive(Debug, PartialEq, Codec)]
    enum Event {
        Click { x: i32, y: i32 },
        KeyPress { key: u32 },
    }

    #[test]
    fn test_derive_enum_struct_variants() {
        let val = Event::Click { x: 10, y: 20 };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Event::decode(&buf).unwrap();
        assert_eq!(decoded, val);

        let val2 = Event::KeyPress { key: 65 };
        buf.clear();
        val2.encode(&mut buf);
        let (decoded2, _) = Event::decode(&buf).unwrap();
        assert_eq!(decoded2, val2);
    }

    #[derive(Debug, PartialEq, Codec)]
    enum Mixed {
        Empty,
        Single(u8),
        Named { label: String },
        Pair(u32, u32),
    }

    #[test]
    fn test_derive_enum_mixed_variants() {
        let cases: Vec<Mixed> = vec![
            Mixed::Empty,
            Mixed::Single(42),
            Mixed::Named { label: "test".to_string() },
            Mixed::Pair(1, 2),
        ];
        for val in cases {
            let mut buf = Vec::new();
            val.encode(&mut buf);
            let (decoded, _) = Mixed::decode(&buf).unwrap();
            assert_eq!(decoded, val);
        }
    }

    #[test]
    fn test_derive_enum_wire_format_stability() {
        // variant index 1 (Green) should encode as 1u32 LE
        let mut buf = Vec::new();
        Color::Green.encode(&mut buf);
        assert_eq!(&buf[..4], &1u32.to_le_bytes());

        // variant index 2 (Rectangle) should encode as 1u32 LE
        buf.clear();
        Shape::Rectangle(1.0, 2.0).encode(&mut buf);
        assert_eq!(&buf[..4], &1u32.to_le_bytes());
    }

    #[test]
    fn test_derive_enum_invalid_variant() {
        let mut buf = Vec::new();
        99u32.encode(&mut buf);
        assert_eq!(Color::decode(&buf), Err(CodecError::InvalidVariant(99)));
    }

    #[derive(Debug, PartialEq, Codec)]
    enum Wrapper<T> {
        None,
        Some(T),
    }

    #[test]
    fn test_derive_enum_generic() {
        let val = Wrapper::Some(42u32);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, _) = Wrapper::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded, val);

        let val2: Wrapper<u32> = Wrapper::None;
        buf.clear();
        val2.encode(&mut buf);
        let (decoded2, _) = Wrapper::<u32>::decode(&buf).unwrap();
        assert_eq!(decoded2, val2);
    }

    // -- error display --

    #[test]
    fn test_error_display() {
        assert_eq!(
            CodecError::NotEnoughData { expected: 4, available: 2 }.to_string(),
            "not enough data: expected 4 bytes, got 2"
        );
        assert_eq!(
            CodecError::InvalidBool(0x42).to_string(),
            "invalid bool byte: 0x42"
        );
        assert_eq!(CodecError::InvalidUtf8.to_string(), "invalid UTF-8");
    }
}
