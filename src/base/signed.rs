use crate::*;

/// Signed number trait.
/// 
/// Signed numbers already exist (`isize`, `i8`, `i16`, `i32`, `i64` and
/// `i128`), but there is no way to address them generically. `Signed` numbers
/// contain the `Unsigned` numbers.
/// 
/// `object: Signed` corresponds to the mathematical 'object E Z'
pub trait Signed : Unsigned {
}

macro_rules! impl_signed {
    ($($t:ty)*) => ($(
        impl Signed for $t {
        }
    )*)
}

impl_signed! { isize i8 i16 i32 i64 i128 }
impl_signed! { f32 f64 }
