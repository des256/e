//! IEEE 754 half-precision (binary16) floating-point type.
//!
//! Layout: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits.
//! Arithmetic promotes to `f32`, computes, and rounds back to `F16`.

use {
    crate::{One, Zero},
    std::{
        cmp::Ordering,
        fmt::{self, Display, Formatter},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign},
    },
};

// -- Internal conversion functions --

/// Convert f16 bit pattern to f32 bit pattern (lossless).
fn f16_to_f32_bits(h: u16) -> u32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h >> 10) & 0x1F;
    let mant = (h & 0x03FF) as u32;

    match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                // Subnormal: renormalize by shifting mantissa until implicit 1 is at bit 10.
                let mut e = 0u32;
                let mut m = mant;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e += 1;
                }
                let m = (m & 0x03FF) << 13;
                let e = (127 - 15 + 1 - e) << 23;
                sign | e | m
            }
        }
        31 => {
            // Inf (mant==0) or NaN (mant!=0).
            sign | 0x7F80_0000 | (mant << 13)
        }
        _ => {
            // Normal: rebias exponent (bias 15 → bias 127).
            let e = (exp as u32 + 112) << 23;
            let m = mant << 13;
            sign | e | m
        }
    }
}

/// Convert f32 bit pattern to f16 bit pattern (lossy, round-to-nearest-even).
fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 0xFF {
        // NaN or Infinity.
        if mant != 0 {
            // NaN: keep sign, set quiet bit, preserve some payload.
            sign | 0x7E00 | ((mant >> 13) as u16).max(1)
        } else {
            sign | 0x7C00
        }
    } else if exp > 142 {
        // Overflow: unbiased exponent > 15.
        sign | 0x7C00
    } else if exp >= 113 {
        // Normal f16 range (unbiased -14..+15).
        let h_exp = ((exp - 112) as u16) << 10;
        let h_mant = (mant >> 13) as u16;
        let round_bit = (mant >> 12) & 1;
        let sticky = mant & 0xFFF;
        let mut result = sign | h_exp | h_mant;
        if round_bit != 0 && (sticky != 0 || (h_mant & 1) != 0) {
            result += 1; // May overflow exponent → infinity; correct.
        }
        result
    } else if exp >= 103 {
        // Subnormal f16 range.
        let shift = (126 - exp) as u32;
        let mant_full = mant | 0x0080_0000;
        let h_mant = (mant_full >> shift) as u16;
        let round_bit = (mant_full >> (shift - 1)) & 1;
        let sticky_mask = (1u32 << (shift - 1)) - 1;
        let sticky = mant_full & sticky_mask;
        let mut result = sign | h_mant;
        if round_bit != 0 && (sticky != 0 || (h_mant & 1) != 0) {
            result += 1; // May promote to smallest normal; correct.
        }
        result
    } else {
        // Too small: flush to ±zero.
        sign
    }
}

// -- The type --

/// IEEE 754 half-precision floating-point type (binary16).
///
/// A 16-bit float stored as a `u16` bit pattern. All arithmetic is performed
/// by promoting to `f32`, computing, and rounding back. This matches CPU
/// behavior — native f16 ALU only exists on GPUs and tensor cores.
///
/// Suitable for tensor data, texture formats, and GPU interop where storage
/// size matters more than compute precision.
///
/// # Examples
///
/// ```
/// use base::F16;
///
/// let a = F16::from_f32(3.14);
/// let b = F16::from_f32(2.0);
/// let c = a * b;
/// assert!((c.to_f32() - 6.28).abs() < 0.01);
/// ```
#[derive(Copy, Clone, Default)]
#[repr(transparent)]
pub struct F16(u16);

impl F16 {
    // -- Constants --

    /// Not a Number.
    pub const NAN: Self = F16(0x7E00);

    /// Positive infinity (+inf).
    pub const INFINITY: Self = F16(0x7C00);

    /// Negative infinity (-inf).
    pub const NEG_INFINITY: Self = F16(0xFC00);

    /// Negative zero (-0.0).
    pub const NEG_ZERO: Self = F16(0x8000);

    /// Negative one (-1.0).
    pub const NEG_ONE: Self = F16(0xBC00);

    /// Two (2.0).
    pub const TWO: Self = F16(0x4000);

    /// One half (0.5).
    pub const HALF: Self = F16(0x3800);

    /// Largest finite value (65504.0).
    pub const MAX: Self = F16(0x7BFF);

    /// Smallest finite value (-65504.0).
    pub const MIN: Self = F16(0xFBFF);

    /// Smallest positive normal value (2^-14 ≈ 6.104e-5).
    pub const MIN_POSITIVE: Self = F16(0x0400);

    /// Machine epsilon (2^-10 ≈ 9.766e-4).
    pub const EPSILON: Self = F16(0x1400);

    /// Number of significant digits in base 2 (including implicit bit).
    pub const MANTISSA_DIGITS: u32 = 11;

    /// One greater than the maximum exponent (unbiased).
    pub const MAX_EXP: i32 = 16;

    /// One greater than the minimum normal exponent (unbiased).
    pub const MIN_EXP: i32 = -13;

    /// Maximum power of 10 representable.
    pub const MAX_10_EXP: i32 = 4;

    /// Minimum power of 10 representable (normal).
    pub const MIN_10_EXP: i32 = -4;

    /// Radix of the representation.
    pub const RADIX: u32 = 2;

    // -- Construction --

    /// Create from raw bit pattern.
    pub const fn from_bits(bits: u16) -> Self {
        F16(bits)
    }

    /// Return raw bit pattern.
    pub const fn to_bits(self) -> u16 {
        self.0
    }

    /// Convert from `f32` (lossy, round-to-nearest-even).
    pub fn from_f32(value: f32) -> Self {
        F16(f32_to_f16_bits(value))
    }

    /// Convert from `f64` (lossy, round-to-nearest-even).
    pub fn from_f64(value: f64) -> Self {
        // Go through f32 first — f64→f32 rounds once, f32→f16 rounds again.
        // Double rounding can differ from direct f64→f16 in rare tie cases,
        // but matches the common implementation strategy (including `half`).
        F16::from_f32(value as f32)
    }

    /// Convert to `f32` (lossless).
    pub fn to_f32(self) -> f32 {
        f32::from_bits(f16_to_f32_bits(self.0))
    }

    /// Convert to `f64` (lossless).
    pub fn to_f64(self) -> f64 {
        self.to_f32() as f64
    }

    // -- Classification --

    /// Returns `true` if NaN.
    pub fn is_nan(self) -> bool {
        (self.0 & 0x7FFF) > 0x7C00
    }

    /// Returns `true` if positive or negative infinity.
    pub fn is_infinite(self) -> bool {
        (self.0 & 0x7FFF) == 0x7C00
    }

    /// Returns `true` if neither NaN nor infinite.
    pub fn is_finite(self) -> bool {
        (self.0 & 0x7C00) != 0x7C00
    }

    /// Returns `true` if normal (not zero, subnormal, infinite, or NaN).
    pub fn is_normal(self) -> bool {
        let exp = self.0 & 0x7C00;
        exp != 0 && exp != 0x7C00
    }

    /// Returns `true` if zero or subnormal.
    pub fn is_subnormal(self) -> bool {
        let exp = self.0 & 0x7C00;
        let mant = self.0 & 0x03FF;
        exp == 0 && mant != 0
    }

    /// Returns `true` if the sign bit is clear.
    pub fn is_sign_positive(self) -> bool {
        (self.0 & 0x8000) == 0
    }

    /// Returns `true` if the sign bit is set.
    pub fn is_sign_negative(self) -> bool {
        (self.0 & 0x8000) != 0
    }

    // -- Math methods --

    /// Absolute value (clears sign bit).
    pub fn abs(self) -> Self {
        F16(self.0 & 0x7FFF)
    }

    /// Returns `1.0` if positive, `-1.0` if negative, `NaN` if NaN.
    pub fn signum(self) -> Self {
        if self.is_nan() {
            Self::NAN
        } else if self.is_sign_negative() {
            Self::NEG_ONE
        } else {
            F16(0x3C00) // 1.0
        }
    }

    /// Returns a value with the magnitude of `self` and the sign of `sign`.
    pub fn copysign(self, sign: Self) -> Self {
        F16((self.0 & 0x7FFF) | (sign.0 & 0x8000))
    }

    /// Square root.
    pub fn sqrt(self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    /// Reciprocal (1 / self).
    pub fn recip(self) -> Self {
        Self::from_f32(self.to_f32().recip())
    }

    /// Largest integer ≤ self.
    pub fn floor(self) -> Self {
        Self::from_f32(self.to_f32().floor())
    }

    /// Smallest integer ≥ self.
    pub fn ceil(self) -> Self {
        Self::from_f32(self.to_f32().ceil())
    }

    /// Round to nearest integer (ties away from zero).
    pub fn round(self) -> Self {
        Self::from_f32(self.to_f32().round())
    }

    /// Truncate toward zero.
    pub fn trunc(self) -> Self {
        Self::from_f32(self.to_f32().trunc())
    }

    /// Fractional part (self - self.trunc()).
    pub fn fract(self) -> Self {
        Self::from_f32(self.to_f32().fract())
    }

    /// Raise to integer power.
    pub fn powi(self, n: i32) -> Self {
        Self::from_f32(self.to_f32().powi(n))
    }

    /// Raise to floating-point power.
    pub fn powf(self, n: Self) -> Self {
        Self::from_f32(self.to_f32().powf(n.to_f32()))
    }

    /// Minimum of two values. If one is NaN, returns the other.
    pub fn min(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().min(other.to_f32()))
    }

    /// Maximum of two values. If one is NaN, returns the other.
    pub fn max(self, other: Self) -> Self {
        Self::from_f32(self.to_f32().max(other.to_f32()))
    }

    /// Clamp to [min, max].
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self::from_f32(self.to_f32().clamp(min.to_f32(), max.to_f32()))
    }

    /// Convert radians to degrees.
    pub fn to_degrees(self) -> Self {
        Self::from_f32(self.to_f32().to_degrees())
    }

    /// Convert degrees to radians.
    pub fn to_radians(self) -> Self {
        Self::from_f32(self.to_f32().to_radians())
    }

    /// Total ordering: -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN.
    pub fn total_cmp(&self, other: &Self) -> Ordering {
        let mut a = self.0 as i16;
        let mut b = other.0 as i16;
        if a < 0 {
            a ^= 0x7FFF;
        }
        if b < 0 {
            b ^= 0x7FFF;
        }
        a.cmp(&b)
    }
}

// -- Comparison --

impl PartialEq for F16 {
    fn eq(&self, other: &Self) -> bool {
        // NaN != NaN, +0 == -0 (IEEE semantics via f32).
        self.to_f32() == other.to_f32()
    }
}

impl PartialOrd for F16 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.to_f32().partial_cmp(&other.to_f32())
    }
}

// -- Display / Debug --

impl Display for F16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.to_f32(), f)
    }
}

impl fmt::Debug for F16 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "F16({})", self.to_f32())
    }
}

// -- Arithmetic operators (all via f32 promotion) --

impl Neg for F16 {
    type Output = Self;
    fn neg(self) -> Self {
        F16(self.0 ^ 0x8000)
    }
}

impl Add for F16 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for F16 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for F16 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for F16 {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Rem for F16 {
    type Output = Self;
    fn rem(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() % rhs.to_f32())
    }
}

impl AddAssign for F16 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for F16 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for F16 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl DivAssign for F16 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl RemAssign for F16 {
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

// -- From conversions --

/// Lossless: F16 → f32.
impl From<F16> for f32 {
    fn from(value: F16) -> Self {
        value.to_f32()
    }
}

/// Lossless: F16 → f64.
impl From<F16> for f64 {
    fn from(value: F16) -> Self {
        value.to_f64()
    }
}

/// Lossless: u8 → F16 (all u8 values are exactly representable).
impl From<u8> for F16 {
    fn from(value: u8) -> Self {
        F16::from_f32(value as f32)
    }
}

/// Lossless: i8 → F16 (all i8 values are exactly representable).
impl From<i8> for F16 {
    fn from(value: i8) -> Self {
        F16::from_f32(value as f32)
    }
}

// -- Trait impls for crate integration --

impl Zero for F16 {
    const ZERO: Self = F16(0x0000);
}

impl One for F16 {
    const ONE: Self = F16(0x3C00);
}

// -- codec --

impl codec::Codec for F16 {
    fn encode(&self, buf: &mut Vec<u8>) {
        self.to_bits().encode(buf);
    }

    fn decode(buf: &[u8]) -> std::result::Result<(Self, usize), codec::CodecError> {
        let (bits, n) = u16::decode(buf)?;
        Ok((F16::from_bits(bits), n))
    }
}

// -- Tests --

#[cfg(test)]
mod tests {
    use {super::*, codec::Codec};

    // -- Codec --

    #[test]
    fn test_codec_f16_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0] {
            let val = F16::from_f32(v);
            let mut buf = Vec::new();
            val.encode(&mut buf);
            let (decoded, len) = F16::decode(&buf).unwrap();
            assert_eq!(buf.len(), len);
            assert_eq!(decoded, val);
        }
    }

    #[test]
    fn test_codec_f16_encoding() {
        let mut buf = Vec::new();
        F16::from_f32(1.0).encode(&mut buf);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf, F16::from_f32(1.0).to_bits().to_le_bytes());
    }

    #[test]
    fn test_codec_f16_truncated() {
        assert!(F16::decode(&[0x01]).is_err());
    }

    // -- Conversion: known exact values --

    #[test]
    fn test_from_f32_zero() {
        let z = F16::from_f32(0.0);
        assert_eq!(z.to_bits(), 0x0000);
        assert_eq!(z.to_f32(), 0.0);
    }

    #[test]
    fn test_from_f32_neg_zero() {
        let z = F16::from_f32(-0.0);
        assert_eq!(z.to_bits(), 0x8000);
        assert_eq!(z.to_f32(), 0.0); // +0 == -0
        assert!(z.is_sign_negative());
    }

    #[test]
    fn test_from_f32_one() {
        let one = F16::from_f32(1.0);
        assert_eq!(one.to_bits(), 0x3C00);
        assert_eq!(one.to_f32(), 1.0);
    }

    #[test]
    fn test_from_f32_neg_one() {
        let neg = F16::from_f32(-1.0);
        assert_eq!(neg.to_bits(), 0xBC00);
        assert_eq!(neg.to_f32(), -1.0);
    }

    #[test]
    fn test_from_f32_half() {
        let h = F16::from_f32(0.5);
        assert_eq!(h.to_bits(), 0x3800);
        assert_eq!(h.to_f32(), 0.5);
    }

    #[test]
    fn test_from_f32_two() {
        let two = F16::from_f32(2.0);
        assert_eq!(two.to_bits(), 0x4000);
        assert_eq!(two.to_f32(), 2.0);
    }

    #[test]
    fn test_from_f32_max() {
        let max = F16::from_f32(65504.0);
        assert_eq!(max.to_bits(), 0x7BFF);
        assert_eq!(max.to_f32(), 65504.0);
    }

    #[test]
    fn test_from_f32_min_positive_normal() {
        // 2^-14
        let min = F16::from_f32(6.103515625e-05);
        assert_eq!(min.to_bits(), 0x0400);
        assert_eq!(min.to_f32(), 6.103515625e-05);
    }

    // -- Conversion: special values --

    #[test]
    fn test_from_f32_infinity() {
        let inf = F16::from_f32(f32::INFINITY);
        assert_eq!(inf.to_bits(), 0x7C00);
        assert!(inf.is_infinite());
        assert!(inf.is_sign_positive());
    }

    #[test]
    fn test_from_f32_neg_infinity() {
        let ninf = F16::from_f32(f32::NEG_INFINITY);
        assert_eq!(ninf.to_bits(), 0xFC00);
        assert!(ninf.is_infinite());
        assert!(ninf.is_sign_negative());
    }

    #[test]
    fn test_from_f32_nan() {
        let nan = F16::from_f32(f32::NAN);
        assert!(nan.is_nan());
    }

    #[test]
    fn test_from_f32_overflow() {
        let big = F16::from_f32(100_000.0);
        assert!(big.is_infinite());
    }

    #[test]
    fn test_from_f32_underflow() {
        // Smaller than smallest f16 subnormal (2^-24 ≈ 5.96e-8).
        let tiny = F16::from_f32(1.0e-10);
        assert_eq!(tiny.to_bits() & 0x7FFF, 0); // flushed to zero
    }

    // -- Conversion: subnormals --

    #[test]
    fn test_subnormal_smallest() {
        // Smallest f16 subnormal: 2^-24.
        let val = F16::from_f32(5.960464477539063e-08);
        assert_eq!(val.to_bits(), 0x0001);
        assert!(val.is_subnormal());
        assert!(!val.is_normal());
    }

    #[test]
    fn test_subnormal_roundtrip() {
        // A mid-range subnormal.
        let bits: u16 = 0x0200; // mantissa = 512, value = 2^-14 * 512/1024 = 2^-15
        let h = F16::from_bits(bits);
        let f = h.to_f32();
        let back = F16::from_f32(f);
        assert_eq!(back.to_bits(), bits);
    }

    // -- Conversion: roundtrips --

    #[test]
    fn test_roundtrip_all_normal_patterns() {
        // Spot-check a spread of normal values.
        for bits in (0x0400u16..=0x7BFF).step_by(37) {
            let h = F16::from_bits(bits);
            let f = h.to_f32();
            let back = F16::from_f32(f);
            assert_eq!(
                back.to_bits(),
                bits,
                "roundtrip failed for bits 0x{:04X}, f32={}",
                bits,
                f
            );
        }
    }

    #[test]
    fn test_roundtrip_all_subnormals() {
        for bits in 0x0001u16..=0x03FF {
            let h = F16::from_bits(bits);
            let f = h.to_f32();
            let back = F16::from_f32(f);
            assert_eq!(
                back.to_bits(),
                bits,
                "subnormal roundtrip failed for bits 0x{:04X}",
                bits
            );
        }
    }

    // -- Classification --

    #[test]
    fn test_classification() {
        assert!(F16::NAN.is_nan());
        assert!(!F16::NAN.is_infinite());
        assert!(!F16::NAN.is_finite());
        assert!(!F16::NAN.is_normal());

        assert!(!F16::INFINITY.is_nan());
        assert!(F16::INFINITY.is_infinite());
        assert!(!F16::INFINITY.is_finite());

        assert!(F16::ONE.is_normal());
        assert!(F16::ONE.is_finite());
        assert!(!F16::ONE.is_nan());

        assert!(!F16::from_bits(0x0000).is_normal()); // zero
        assert!(!F16::from_bits(0x0001).is_normal()); // subnormal
        assert!(F16::from_bits(0x0001).is_subnormal());
    }

    #[test]
    fn test_sign() {
        assert!(F16::ONE.is_sign_positive());
        assert!(!F16::ONE.is_sign_negative());
        assert!(F16::NEG_ONE.is_sign_negative());
        assert!(!F16::NEG_ONE.is_sign_positive());
        assert!(F16::NEG_ZERO.is_sign_negative());
    }

    // -- Arithmetic --

    #[test]
    fn test_add() {
        let a = F16::from_f32(1.0);
        let b = F16::from_f32(2.0);
        assert_eq!((a + b).to_f32(), 3.0);
    }

    #[test]
    fn test_sub() {
        let a = F16::from_f32(5.0);
        let b = F16::from_f32(3.0);
        assert_eq!((a - b).to_f32(), 2.0);
    }

    #[test]
    fn test_mul() {
        let a = F16::from_f32(3.0);
        let b = F16::from_f32(4.0);
        assert_eq!((a * b).to_f32(), 12.0);
    }

    #[test]
    fn test_div() {
        let a = F16::from_f32(10.0);
        let b = F16::from_f32(4.0);
        assert_eq!((a / b).to_f32(), 2.5);
    }

    #[test]
    fn test_rem() {
        let a = F16::from_f32(7.0);
        let b = F16::from_f32(3.0);
        assert_eq!((a % b).to_f32(), 1.0);
    }

    #[test]
    fn test_neg() {
        let a = F16::from_f32(3.5);
        assert_eq!((-a).to_f32(), -3.5);
        assert_eq!((-(-a)).to_f32(), 3.5);
    }

    #[test]
    fn test_div_by_zero() {
        let a = F16::from_f32(1.0);
        let z = F16::from_f32(0.0);
        assert!((a / z).is_infinite());
        assert!((z / z).is_nan());
    }

    #[test]
    fn test_assign_ops() {
        let mut a = F16::from_f32(10.0);
        a += F16::from_f32(5.0);
        assert_eq!(a.to_f32(), 15.0);
        a -= F16::from_f32(3.0);
        assert_eq!(a.to_f32(), 12.0);
        a *= F16::from_f32(2.0);
        assert_eq!(a.to_f32(), 24.0);
        a /= F16::from_f32(4.0);
        assert_eq!(a.to_f32(), 6.0);
        a %= F16::from_f32(4.0);
        assert_eq!(a.to_f32(), 2.0);
    }

    // -- Comparison --

    #[test]
    fn test_eq() {
        assert_eq!(F16::from_f32(1.0), F16::from_f32(1.0));
        assert_ne!(F16::from_f32(1.0), F16::from_f32(2.0));
    }

    #[test]
    fn test_nan_ne() {
        assert_ne!(F16::NAN, F16::NAN);
    }

    #[test]
    fn test_zero_eq() {
        // +0 == -0 per IEEE.
        assert_eq!(F16::from_bits(0x0000), F16::from_bits(0x8000));
    }

    #[test]
    fn test_ord() {
        assert!(F16::from_f32(1.0) < F16::from_f32(2.0));
        assert!(F16::from_f32(-1.0) < F16::from_f32(1.0));
        assert_eq!(
            F16::NAN.partial_cmp(&F16::ONE),
            None,
            "NaN should be unordered"
        );
    }

    #[test]
    fn test_total_cmp() {
        assert_eq!(F16::NEG_ZERO.total_cmp(&F16::from_bits(0x0000)), Ordering::Less);
        assert_eq!(F16::ONE.total_cmp(&F16::TWO), Ordering::Less);
        assert_eq!(F16::NEG_ONE.total_cmp(&F16::ONE), Ordering::Less);
    }

    // -- Math methods --

    #[test]
    fn test_abs() {
        assert_eq!(F16::from_f32(-3.0).abs().to_f32(), 3.0);
        assert_eq!(F16::from_f32(3.0).abs().to_f32(), 3.0);
        assert_eq!(F16::NEG_ZERO.abs().to_bits(), 0x0000);
    }

    #[test]
    fn test_signum() {
        assert_eq!(F16::from_f32(5.0).signum().to_f32(), 1.0);
        assert_eq!(F16::from_f32(-5.0).signum().to_f32(), -1.0);
        assert!(F16::NAN.signum().is_nan());
    }

    #[test]
    fn test_copysign() {
        let a = F16::from_f32(3.0);
        assert_eq!(a.copysign(F16::NEG_ONE).to_f32(), -3.0);
        assert_eq!(a.copysign(F16::ONE).to_f32(), 3.0);
    }

    #[test]
    fn test_sqrt() {
        assert_eq!(F16::from_f32(4.0).sqrt().to_f32(), 2.0);
        assert_eq!(F16::from_f32(9.0).sqrt().to_f32(), 3.0);
        assert!(F16::from_f32(-1.0).sqrt().is_nan());
    }

    #[test]
    fn test_recip() {
        assert_eq!(F16::from_f32(2.0).recip().to_f32(), 0.5);
        assert_eq!(F16::from_f32(4.0).recip().to_f32(), 0.25);
    }

    #[test]
    fn test_floor_ceil_round_trunc() {
        let v = F16::from_f32(2.7);
        let f = v.to_f32();
        assert_eq!(v.floor().to_f32(), f.floor());
        assert_eq!(v.ceil().to_f32(), f.ceil());
        assert_eq!(v.round().to_f32(), f.round());
        assert_eq!(v.trunc().to_f32(), f.trunc());
    }

    #[test]
    fn test_fract() {
        let v = F16::from_f32(3.75);
        assert_eq!(v.fract().to_f32(), 0.75);
    }

    #[test]
    fn test_powi() {
        assert_eq!(F16::from_f32(2.0).powi(3).to_f32(), 8.0);
        assert_eq!(F16::from_f32(3.0).powi(2).to_f32(), 9.0);
    }

    #[test]
    fn test_min_max_clamp() {
        let a = F16::from_f32(3.0);
        let b = F16::from_f32(5.0);
        assert_eq!(a.min(b).to_f32(), 3.0);
        assert_eq!(a.max(b).to_f32(), 5.0);

        let v = F16::from_f32(10.0);
        assert_eq!(v.clamp(a, b).to_f32(), 5.0);
        assert_eq!(a.clamp(a, b).to_f32(), 3.0);
    }

    #[test]
    fn test_min_nan() {
        // f32::min returns the non-NaN argument.
        assert_eq!(F16::NAN.min(F16::ONE).to_f32(), 1.0);
        assert_eq!(F16::ONE.min(F16::NAN).to_f32(), 1.0);
    }

    // -- Display --

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", F16::from_f32(1.5)), "1.5");
    }

    #[test]
    fn test_debug() {
        assert_eq!(format!("{:?}", F16::from_f32(1.5)), "F16(1.5)");
    }

    // -- From conversions --

    #[test]
    fn test_from_u8() {
        let h: F16 = 255u8.into();
        assert_eq!(h.to_f32(), 255.0);
    }

    #[test]
    fn test_from_i8() {
        let h: F16 = (-128i8).into();
        assert_eq!(h.to_f32(), -128.0);
    }

    #[test]
    fn test_into_f32() {
        let h = F16::from_f32(42.0);
        let f: f32 = h.into();
        assert_eq!(f, 42.0);
    }

    #[test]
    fn test_into_f64() {
        let h = F16::from_f32(42.0);
        let d: f64 = h.into();
        assert_eq!(d, 42.0);
    }

    // -- Zero/One traits --

    #[test]
    fn test_zero_one_traits() {
        assert_eq!(F16::ZERO.to_f32(), 0.0);
        assert_eq!(<F16 as Zero>::ZERO.to_f32(), 0.0);
        assert_eq!(<F16 as One>::ONE.to_f32(), 1.0);
    }

    // -- Rounding behavior --

    #[test]
    fn test_round_to_nearest_even() {
        // 1.0 + epsilon/2 should round to 1.0 (tie → even, 1.0 mantissa=0 is even).
        // 1.0 in f16 = 0x3C00. epsilon = 2^-10.
        // 1.0 + 2^-11 is exactly between 1.0 (mant=0) and 1.0+eps (mant=1).
        // Round to even → mant=0 → 1.0.
        let val = 1.0f32 + f32::powi(2.0, -11);
        let h = F16::from_f32(val);
        assert_eq!(h.to_bits(), 0x3C00, "tie should round to even (1.0)");

        // 1.0+eps + epsilon/2 should round to 1.0+2*eps (tie → even, mant=1 is odd → round up).
        let one_plus_eps = F16::from_bits(0x3C01).to_f32(); // 1.0 + 2^-10
        let val2 = one_plus_eps + f32::powi(2.0, -11);
        let h2 = F16::from_f32(val2);
        assert_eq!(h2.to_bits(), 0x3C02, "tie should round to even (up)");
    }

    // -- Default --

    #[test]
    fn test_default() {
        let d = F16::default();
        assert_eq!(d.to_bits(), 0x0000);
        assert_eq!(d.to_f32(), 0.0);
    }

    // -- Edge: largest f32 that stays finite in f16 --

    #[test]
    fn test_boundary_max() {
        // 65504.0 should be MAX.
        assert_eq!(F16::from_f32(65504.0).to_bits(), 0x7BFF);
        // 65520.0 overflows to infinity (65504 + 16 = 65520, which is
        // the next rounding boundary above MAX).
        assert!(F16::from_f32(65520.0).is_infinite());
    }
}
