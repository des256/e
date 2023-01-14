use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Formatter,
            Result,
        },
        ops::{
            Add,
            Sub,
            Mul,
            Div,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
            Neg,
        },
    },
};

/// Generic vector.
#[derive(Copy,Clone,Debug)]
pub struct VecN<T,const S: usize>([T; S]);

impl<T: Copy + Mul<T,Output=T> + AddAssign<T> + DivAssign<T> + Real + Zero + PartialEq<T>,const S: usize> VecN<T,S> {

    /// Create vector from splatting  one value.
    pub fn splat(value: T) -> Self {
        VecN([value; S])
    }

    /// The dot-product. 
    pub fn dot(self,other: &Self) -> T {
        let mut result: T = self.0[0] * other.0[0];
        for i in 1..S {
            result += self.0[i] * other.0[i];
        }
        result
    }

    /// The norm/length of the vector.
    pub fn norm(&self) -> T {
        self.dot(&self).sqrt()
    }

    /// Normalize the vector (scale so norm becomes 1).
    pub fn normalize(&mut self) {
        let d = self.norm();
        if d != T::ZERO {
            for i in 0..S {
                self.0[i] /= d;
            }
        }
    }

    /// Component-wise scaling of the vector.
    pub fn scale(&self,other: &Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] * other.0[i];
        }
        VecN(result)
    }
}

/// Partial equality operators `==` and `!=`.
impl<T: PartialEq,const S: usize> PartialEq for VecN<T,S> {
    fn eq(&self,other: &Self) -> bool {
        for i in 0..S {
            if self.0[i] != other.0[i] {
                return false;
            }
        }
        true
    }
}

/// Prettyprint the vector.
impl<T: Display,const S: usize> Display for VecN<T,S> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"(")?;
        write!(f,"{}",self.0[0])?;
        if S > 1 {
            for i in 1..S {
                write!(f,",{}",self.0[i])?;
            }
        }
        write!(f,")")
    }
}

/// The Zero trait, adds ZERO constant to the vector.
impl<T: Zero,const S: usize> Zero for VecN<T,S> { const ZERO: Self = VecN([T::ZERO; S]); }

/// The addition operator `+`.
impl<T: Copy + Zero + Add<T,Output=T>,const S: usize> Add<VecN<T,S>> for VecN<T,S> {
    type Output = VecN<T,S>;
    fn add(self,other: Self) -> Self::Output {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] + other.0[i];
        }
        VecN(result)
    }
}

/// The addition-assignment operator `+=`.
impl<T: Copy + AddAssign<T>,const S: usize> AddAssign<VecN<T,S>> for VecN<T,S> {
    fn add_assign(&mut self,other: Self) {
        for i in 0..S {
            self.0[i] += other.0[i];
        }
    }
}

/// The subtraction operator `-`.
impl<T: Copy + Zero + Sub<T,Output=T>,const S: usize> Sub<VecN<T,S>> for VecN<T,S> {
    type Output = VecN<T,S>;
    fn sub(self,other: Self) -> Self::Output {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] - other.0[i];
        }
        VecN(result)
    }
}

/// The subtraction-assignment operator `-=`.
impl<T: Copy + SubAssign<T>,const S: usize> SubAssign<VecN<T,S>> for VecN<T,S> {
    fn sub_assign(&mut self,other: Self) {
        for i in 0..S {
            self.0[i] -= other.0[i];
        }
    }
}

macro_rules! scalar_vec_mul {
    ($($t:ty)+) => {
        $(
            /// Scalar-vector multiplication operator `*`.
            impl<const S: usize> Mul<VecN<$t,S>> for $t {
                type Output = VecN<$t,S>;
                fn mul(self, other: VecN<$t,S>) -> Self::Output {
                    let mut result = [<$t>::ZERO; S];
                    for i in 0..S {
                        result[i] = self * other.0[i];
                    }
                    VecN(result)
                }
            }
        )+
    }
}

scalar_vec_mul!(usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64);

/// Vector-scalar multiplication operator `*`.
impl<T: Copy + Zero + Mul<T,Output=T>,const S: usize> Mul<T> for VecN<T,S> {
    type Output = Self;
    fn mul(self,other: T) -> Self::Output {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] * other;
        }
        VecN(result)
    }
}

/// Vector-scalar multiply-assignment operator `*=`.
impl<T: Copy + MulAssign<T>,const S: usize> MulAssign<T> for VecN<T,S> {
    fn mul_assign(&mut self,other: T) {
        for i in 0..S {
            self.0[i] *= other;
        }
    }
}

/// Vector-scalar division operator `/`.
impl<T: Copy + Zero + Div<T,Output=T>,const S: usize> Div<T> for VecN<T,S> {
    type Output = Self;
    fn div(self,other: T) -> Self::Output {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] / other;
        }
        VecN(result)
    }
}

/// Vector-scalar division-assignment operator `/=`.
impl<T: Copy + DivAssign<T>,const S: usize> DivAssign<T> for VecN<T,S> {
    fn div_assign(&mut self,other: T) {
        for i in 0..S {
            self.0[i] /= other;
        }
    }
}

macro_rules! vec_signed {
    ($($t:ty)+) => {
        $(
            /// Negation operator `-`.
            impl<const S: usize> Neg for VecN<$t,S> {
                type Output = VecN<$t,S>;
                fn neg(self) -> Self::Output {
                    let mut result = [0 as $t; S];
                    for i in 0..S {
                        result[i] = -self.0[i];
                    }
                    VecN(result)
                }
            }
        )+
    }
}

vec_signed!(isize i8 i16 i32 i64 i128 f32 f64);

#[cfg(test)]
mod tests {

    use crate::*;

    #[test]
    fn dot_product() {
        let a: VecN<f32,5> = VecN::splat(2.0);
        let b: VecN<f32,5> = VecN::splat(3.0);
        assert_eq!(a.dot(&b),30.0);
    }

    #[test]
    fn norm() {
        let a: VecN<f32,5> = VecN::splat(2.0);
        assert_eq!(a.norm(),20.0.sqrt());
    }

    #[test]
    fn normalize() {
        let mut a: VecN<f32,5> = VecN::splat(2.0);
        a.normalize();
        assert_eq!(a,VecN::splat(2.0 / 20.0.sqrt()));
    }

    #[test]
    fn scale() {
        let a: VecN<f32,5> = VecN::splat(2.0);
        let b: VecN<f32,5> = VecN::splat(3.0);
        assert_eq!(a.scale(&b),VecN::splat(6.0));
    }

    #[test]
    fn add() {
        let a: VecN<f32,5> = VecN::splat(2.0);
        let b: VecN<f32,5> = VecN::splat(3.0);
        assert_eq!(a + b,VecN::splat(5.0));
    }

    #[test]
    fn add_assign() {
        let mut a: VecN<f32,5> = VecN::splat(2.0);
        a += VecN::splat(3.0);
        assert_eq!(a,VecN::splat(5.0));
    }

    #[test]
    fn sub() {
        let a: VecN<f32,5> = VecN::splat(3.0);
        let b: VecN<f32,5> = VecN::splat(2.0);
        assert_eq!(a - b,VecN::splat(1.0));
    }

    #[test]
    fn sub_assign() {
        let mut a: VecN<f32,5> = VecN::splat(3.0);
        a -= VecN::splat(1.0);
        assert_eq!(a,VecN::splat(2.0));
    }

    #[test]
    fn scalar_mul() {
        let a: VecN<f32,5> = VecN::splat(3.0);
        assert_eq!(2.0 * a,VecN::splat(6.0));
    }

    #[test]
    fn mul_scalar() {
        let a: VecN<f32,5> = VecN::splat(3.0);
        assert_eq!(a * 2.0,VecN::splat(6.0));
    }

    #[test]
    fn mul_assign() {
        let mut a: VecN<f32,5> = VecN::splat(3.0);
        a *= 2.0;
        assert_eq!(a,VecN::splat(6.0));
    }

    #[test]
    fn div() {
        let a: VecN<f32,5> = VecN::splat(6.0);
        assert_eq!(a / 2.0,VecN::splat(3.0));
    }

    #[test]
    fn div_assign() {
        let mut a: VecN<f32,5> = VecN::splat(6.0);
        a /= 2.0;
        assert_eq!(a,VecN::splat(3.0));
    }

    #[test]
    fn neg() {
        let a: VecN<f32,5> = VecN::splat(2.0);
        assert_eq!(-a,VecN::splat(-2.0));
    }
}
