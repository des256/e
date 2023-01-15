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

/// generic vector
#[derive(Copy,Clone,Debug)]
pub struct VecN<T,const S: usize>([T; S]);

impl<T: Copy,const S: usize> VecN<T,S> {

    pub fn splat(value: T) -> Self {
        VecN([value; S])
    }
}

impl<T: Copy + Mul<T,Output=T> + AddAssign<T>,const S: usize> VecN<T,S> {
    pub fn dot(self,other: &Self) -> T {
        let mut result: T = self.0[0] * other.0[0];
        for i in 1..S {
            result += self.0[i] * other.0[i];
        }
        result
    }
}

impl<T: Copy + Real + Mul<T,Output=T> + AddAssign<T>,const S: usize> VecN<T,S> {
    pub fn norm(&self) -> T {
        self.dot(&self).sqrt()
    }
}

impl<T: Copy + Real + Mul<T,Output=T> + AddAssign<T> + Zero + PartialEq<T> + DivAssign<T>,const S: usize> VecN<T,S> {
    pub fn normalize(&mut self) {
        let d = self.norm();
        if d != T::ZERO {
            for i in 0..S {
                self.0[i] /= d;
            }
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + AddAssign<T> + DivAssign<T> + Real + Zero + PartialEq<T>,const S: usize> VecN<T,S> {
    pub fn scale(&self,other: &Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = self.0[i] * other.0[i];
        }
        VecN(result)
    }
}

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

impl<T: Zero,const S: usize> Zero for VecN<T,S> { const ZERO: Self = VecN([T::ZERO; S]); }

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

impl<T: Copy + AddAssign<T>,const S: usize> AddAssign<VecN<T,S>> for VecN<T,S> {
    fn add_assign(&mut self,other: Self) {
        for i in 0..S {
            self.0[i] += other.0[i];
        }
    }
}

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

impl<T: Copy + MulAssign<T>,const S: usize> MulAssign<T> for VecN<T,S> {
    fn mul_assign(&mut self,other: T) {
        for i in 0..S {
            self.0[i] *= other;
        }
    }
}

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

impl<T: Copy + DivAssign<T>,const S: usize> DivAssign<T> for VecN<T,S> {
    fn div_assign(&mut self,other: T) {
        for i in 0..S {
            self.0[i] /= other;
        }
    }
}

impl<T: Copy + Neg<Output=T> + Zero,const S: usize> Neg for VecN<T,S> {
    type Output = VecN<T,S>;
    fn neg(self) -> Self::Output {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result[i] = -self.0[i];
        }
        VecN(result)
    }
}

impl<const S: usize> VecN<bool,S> {
    pub fn select<T: Copy>(&self,a: VecN<T,S>,b: VecN<T,S>) -> VecN<T,S> {
        let mut result = a;
        for i in 0..S {
            if self.0[i] {
                result.0[i] = b.0[i];
            }
        }
        result
    }

    pub fn all(&self) -> bool {
        for i in 0..S {
            if !self.0[i] {
                return false;
            }
        }
        true
    }

    pub fn any(&self) -> bool {
        for i in 0..S {
            if self.0[i] {
                return true;
            }
        }
        false
    }

    pub fn not(&self) -> Self {
        let mut result = VecN([false; S]);
        for i in 0..S {
            if !self.0[i] {
                result.0[i] = true;
            }
        }
        result
    }
}

impl<T: Copy + Zero + Neg<Output=T> + PartialOrd<T>,const S: usize> VecN<T,S> {
    pub fn abs(self) -> Self {
        let mut result = self;
        for i in 0..S {
            if self.0[i] < T::ZERO {
                result.0[i] = -self.0[i];
            }
        }
        result
    }
}

impl<T: Zero + One + PartialOrd<T> + Neg<Output=T>,const S: usize> VecN<T,S> {
    pub fn signum(self) -> Self {
        let mut result = [T::ONE; S];
        for i in 0..S {
            if self.0[i] < T::ZERO {
                result[i] = -T::ONE;
            }
        }
        VecN(result)
    }
}

impl<T: Copy + PartialOrd<T>,const S: usize> VecN<T,S> {
    pub fn min(self,other: Self) -> Self {
        let mut result = self;
        for i in 0..S {
            if other.0[i] < self.0[i] {
                result.0[i] = other.0[i];
            }
        }
        result
    }
}

impl<T: Copy + PartialOrd<T>,const S: usize> VecN<T,S> {
    pub fn max(self,other: Self) -> Self {
        let mut result = self;
        for i in 0..S {
            if other.0[i] > self.0[i] {
                result.0[i] = other.0[i];
            }
        }
        result
    }
}

impl<T: Copy + PartialOrd<T>,const S: usize> VecN<T,S> {
    pub fn clamp(self,min: Self,max: Self) -> Self {
        let mut result = self;
        for i in 0..S {
            if min.0[i] > result.0[i] {
                result.0[i] = min.0[i];
            }
            if max.0[i] < result.0[i] {
                result.0[i] = max.0[i];
            }
        }
        result
    }
}

impl<T: Copy + PartialOrd<T>,const S: usize> VecN<T,S> {
    pub fn sclamp(self,min: T,max: T) -> Self {
        let mut result = self;
        for i in 0..S {
            if min > result.0[i] {
                result.0[i] = min;
            }
            if max < result.0[i] {
                result.0[i] = max;
            }
        }
        result
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn to_radians(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].to_radians();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn to_degrees(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].to_degrees();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn sin(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].sin();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn cos(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].cos();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn tan(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].tan();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn sinh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].sinh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn cosh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].cosh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn tanh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].tanh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn asin(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].asin();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn acos(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].acos();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn atan(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].atan();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn atan2(self,other: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].atan2(other.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn asinh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].asinh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn acosh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].acosh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn atanh(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].atanh();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn powf(self,other: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].powf(other.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn spowf(self,other: T) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].spowf(other);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn exp(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].exp();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn ln(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].ln();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn exp2(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].exp2();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn log2(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].log2();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn sqrt(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].sqrt();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn div_euclid(self,other: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].div_euclid(other.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn rem_euclid(self,other: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].rem_euclid(other.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn invsqrt(self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].invsqrt();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn modf(self) -> (Self,Self) {
        let mut result0 = [T::ZERO; S];
        let mut result1 = [T::ZERO; S];
        for i in 0..S {
            let r = self.0[i].modf();
            result0.0[i] = r.0;
            result1.0[i] = r.1;
        }
        VecN((result0,result1))
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn mix(self,other: Self,a: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].mix(other.0[i],a.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn smix(self,other: Self,a: T) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].mix(other.0[i],a);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn step(self,edge: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].step(edge.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn sstep(self,edge: T) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].sstep(edge);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn smoothstep(self,edge0: Self,edge1: Self) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].smoothstep(edge0.0[i],edge1.0[i]);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn ssmoothstep(self,edge0: T,edge1: T) -> Self {
        let mut result = [T::ZERO; S];
        for i in 0..S {
            result.0[i] = self.0[i].ssmoothstep(edge0,edge1);
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn is_nan(self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i].is_nan();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn is_infinite(self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i].is_infinite();
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn fma(self,b: Self,c: Self) -> Self {
        let mut result = c;
        for i in 0..S {
            result.0[i] += self.0[i] * b.0[i];
        }
        result
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn distance(self,other: Self) -> T {
        (other - self).norm()
    }
}

/*
impl<T,const U: usize,const S: usize> VecN<T,S> {
    pub fn outer(self,other: VecN<T,U>) -> MatNxM<T,U,S> {
        let mut result = [[T::ZERO; S]; U];
        for u in 0..U {
            for s in 0..S {
                result[u][s] = self.0[s] * other.0[u];
            }
        }
        MatNxM(result)
    }
}
*/

impl<T,const S: usize> VecN<T,S> {
    pub fn less_than(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] < other.0[i];
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn less_than_equal(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] <= other.0[i];
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn greater_than(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] > other.0[i];
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn greater_than_equal(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] >= other.0[i];
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn equal(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] == other.0[i];
        }
        VecN(result)
    }
}

impl<T,const S: usize> VecN<T,S> {
    pub fn not_equal(self,other: Self) -> VecN<bool,S> {
        let mut result = [false; S];
        for i in 0..S {
            result.0[i] = self.0[i] != other.0[i];
        }
        VecN(result)
    }
}

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
