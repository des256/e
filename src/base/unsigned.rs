/// Unsigned number trait.
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
    fn div_euclid(self,rhs: Self) -> Self;
    fn rem_euclid(self,rhs: Self) -> Self;
}

macro_rules! impl_unsigned {
    ($($t:ty)+) => {
        $(
            impl Unsigned for $t {

                /*
                const MIN: Self = <$t>::MIN;
                const MAX: Self = <$t>::MAX;
                const BITS: u32 = <$t>::BITS;
                */

                fn div_euclid(self,rhs: Self) -> Self {
                    self / rhs
                }

                fn rem_euclid(self,rhs: Self) -> Self {
                    self % rhs
                }
            }
        )+
    }
}

impl_unsigned! { usize u8 u16 u32 u64 u128 }
impl_unsigned! { isize i8 i16 i32 i64 i128 }
impl_unsigned! { f32 f64 }

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn div_euclid() {
        assert_eq!(4f32.div_euclid(3f32),1f32);
    }

    #[test]
    fn rem_euclid() {
        assert_eq!(4f32.rem_euclid(3f32),1f32);
    }
}
