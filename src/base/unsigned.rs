/// Unsigned integer number trait.
/// 
/// Unsigned numbers already exist (`usize`, `u8`, `u16`, `u32`, `u64` and
/// `u128`), but there is no way to address them generically.
/// 
/// `object: Unsigned` corresponds to the mathematical 'object E N'
pub trait Unsigned {
    /*
    const MIN: Self;
    const MAX: Self;
    const BITS: u32;
    */
    fn pow(self,exp: u32) -> Self;
    fn div_euclid(self,rhs: Self) -> Self;
    fn rem_euclid(self,rhs: Self) -> Self;
}

macro_rules! impl_unsigned {
    ($($t:ty)*) => ($(
        impl Unsigned for $t {

            /*
            const MIN: Self = <$t>::MIN;
            const MAX: Self = <$t>::MAX;
            const BITS: u32 = <$t>::BITS;
            */

            fn pow(self,exp: u32) -> Self {
                let mut r = self;
                for i in 0..(exp - 1) {
                    r *= self;
                }
                r
            }

            fn div_euclid(self,rhs: Self) -> Self {
                self / rhs
            }

            fn rem_euclid(self,rhs: Self) -> Self {
                self % rhs
            }
        }
    )*)
}

impl_unsigned! { usize u8 u16 u32 u64 u128 }
impl_unsigned! { isize i8 i16 i32 i64 i128 }
impl_unsigned! { f32 f64 }
