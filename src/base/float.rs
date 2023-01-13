use {
    crate::*,
    std::cmp::Ordering,
};

/// Floating-point template.
/// 
/// Floating-point numbers already exist in Rust, but a generic way to address them does not.
pub trait Float: Real {
    const DIGITS: u32;
    const MANTISSA_DIGITS: u32;
    const EPSILON: Self;
    const MIN_POSITIVE: Self;
    const MIN_EXP: i32;
    const MAX_EXP: i32;
    const MIN_10_EXP: i32;
    const MAX_10_EXP: i32;
    const NAN: Self;
    const INFINITY: Self;
    const NEG_INFINITY: Self;
    const RADIX: u32;
    fn is_nan(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_finite(self) -> bool;
    fn is_subnormal(self) -> bool;
    fn is_normal(self) -> bool;
    fn total_cmp(&self,other: &Self) -> Ordering;
}

macro_rules! float_impl {
    ($($t:ty)*) => ($(
        impl Float for $t {
            const DIGITS: u32 = <$t>::DIGITS;
            const MANTISSA_DIGITS: u32 = <$t>::MANTISSA_DIGITS;
            const EPSILON: Self = <$t>::EPSILON;
            const MIN_POSITIVE: Self = <$t>::MIN_POSITIVE;
            const MIN_EXP: i32 = <$t>::MIN_EXP;
            const MAX_EXP: i32 = <$t>::MAX_EXP;
            const MIN_10_EXP: i32 = <$t>::MIN_10_EXP;
            const MAX_10_EXP: i32 = <$t>::MAX_10_EXP;
            const NAN: Self = <$t>::NAN;
            const INFINITY: Self = <$t>::INFINITY;
            const NEG_INFINITY: Self = <$t>::NEG_INFINITY;
            const RADIX: u32 = <$t>::RADIX;
            fn is_nan(self) -> bool { self.is_nan() }
            fn is_infinite(self) -> bool { self.is_infinite() }
            fn is_finite(self) -> bool { self.is_finite() }
            fn is_subnormal(self) -> bool { self.is_subnormal() }
            fn is_normal(self) -> bool { self.is_normal() }
            fn total_cmp(&self,other: &Self) -> Ordering { self.total_cmp(other) }
        }
    )*)
}

float_impl! { f32 f64 }
