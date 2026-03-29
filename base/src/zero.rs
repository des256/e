/// Trait for types that have an additive identity (zero element).
///
/// Provides a `ZERO` constant representing the additive identity,
/// where `x + ZERO == x` for all `x`.
///
/// Implemented for all primitive numeric types.
pub trait Zero {
    /// The additive identity element.
    const ZERO: Self;
}

impl Zero for u8 {
    const ZERO: Self = 0;
}

impl Zero for u16 {
    const ZERO: Self = 0;
}

impl Zero for u32 {
    const ZERO: Self = 0;
}

impl Zero for u64 {
    const ZERO: Self = 0;
}

impl Zero for u128 {
    const ZERO: Self = 0;
}

impl Zero for usize {
    const ZERO: Self = 0;
}

impl Zero for i8 {
    const ZERO: Self = 0;
}

impl Zero for i16 {
    const ZERO: Self = 0;
}

impl Zero for i32 {
    const ZERO: Self = 0;
}

impl Zero for i64 {
    const ZERO: Self = 0;
}

impl Zero for i128 {
    const ZERO: Self = 0;
}

impl Zero for isize {
    const ZERO: Self = 0;
}

impl Zero for f32 {
    const ZERO: Self = 0.0;
}

impl Zero for f64 {
    const ZERO: Self = 0.0;
}
