use {
    crate::*,
    std::{
        cmp::{
            PartialEq,
            PartialOrd,
            Ordering,
        },
        fmt::{
            Display,
            Debug,
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
            Shl,
            Shr,
        },
    },
};

/// Fixed point template.
/// 
/// Fixed point numbers are sometimes easier to use than floating point numbers. Also, some architectures do not support
/// floating point numbers at all.
/// 
/// `Fixed` implements `Real` as it is one implementation of real numbers. The onther one is IEEE floats `f32`/`f64`.

#[derive(Copy,Clone,Debug)]
pub struct Fixed<T,const B: usize>(T);

impl<T,const B: usize> Fixed<T,B> {
    const BITS: usize = B;
}

impl<T: Copy,const B: usize> Display for Fixed<T,B> where f64: From<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        let value = f64::from(self.0) / 2.0f64.powf(Self::BITS as f64);
        write!(f,"{}",value)
    }
}

/*
macro_rules! fixed_impl {
    ($($t:ty)*) => ($(

        // scalar == fixed
        impl<T: From<$t> + Shl<usize,Output=T> + PartialEq,const B: usize> PartialEq<Fixed<T,B>> for $t {
            fn eq(&self,other: &Fixed<T,B>) -> bool {
                let scalar = T::from(*self) << B;
                scalar == other.0
            }
        }

        // fixed == scalar
        impl<T: From<$t> + Shl<usize,Output=T> + PartialEq,const B: usize> PartialEq<$t> for Fixed<T,B> {
            fn eq(&self,other: &$t) -> bool {
                let scalar = T::from(*other) << B;
                self.0 == scalar
            }
        }

        // scalar ? fixed
        impl<T: From<$t> + Shl<usize,Output=T> + PartialOrd,const B: usize> PartialOrd<Fixed<T,B>> for $t {
            fn partial_cmp(&self,other: &Fixed<T,B>) -> Option<Ordering> {
                let scalar = T::from(*self) << B;
                scalar.partial_cmp(&other.0)
            }
        }

        // fixed ? scalar
        impl<T: From<$t> + Shl<usize,Output=T> + PartialOrd,const B: usize> PartialOrd<$t> for Fixed<T,B> {
            fn partial_cmp(&self,other: &$t) -> Option<Ordering> {
                let scalar = T::from(*other) << B;
                self.0.partial_cmp(&scalar)
            }
        }

        // scalar + fixed
        impl<T: From<$t> + Add<Output=T> + Shl<usize,Output=T>,const B: usize> Add<Fixed<T,B>> for $t {
            type Output = Fixed<T,B>;
            fn add(self,other: Fixed<T,B>) -> Self::Output {
                Fixed((T::from(self) << B) + other.0)
            }
        }

        // fixed + scalar
        impl<T: From<$t> + Add<Output=T> + Shl<usize,Output=T>,const B: usize> Add<$t> for Fixed<T,B> {
            type Output = Self;
            fn add(self,other: $t) -> Self::Output {
                Fixed(self.0 + (T::from(other) << B))
            }
        }

        // fixed += scalar
        impl<T: From<$t> + Add<Output=T> + AddAssign + Shl<usize,Output=T>,const B: usize> AddAssign<$t> for Fixed<T,B> {
            fn add_assign(&mut self,other: $t) {
                self.0 += (T::from(other) << B);
            }
        }

        // scalar - fixed
        impl<T: From<$t> + Sub<Output=T> + Shl<usize,Output=T>,const B: usize> Sub<Fixed<T,B>> for $t {
            type Output = Fixed<T,B>;
            fn sub(self,other: Fixed<T,B>) -> Self::Output {
                Fixed((T::from(self) << B) - other.0)
            }
        }

        // fixed - scalar
        impl<T: From<$t> + Sub<Output=T> + Shl<usize,Output=T>,const B: usize> Sub<$t> for Fixed<T,B> {
            type Output = Self;
            fn sub(self,other: $t) -> Self::Output {
                Fixed(self.0 - (T::from(other) << B))
            }
        }

        // fixed -= scalar
        impl<T: From<$t> + Sub<Output=T> + SubAssign + Shl<usize,Output=T>,const B: usize> SubAssign<$t> for Fixed<T,B> {
            fn sub_assign(&mut self,other: $t) {
                self.0 -= (T::from(other) << B);
            }
        }

        // scalar * fixed
        impl<T: From<$t> + Mul<Output=T>,const B: usize> Mul<Fixed<T,B>> for $t {
            type Output = Fixed<T,B>;
            fn mul(self,other: Fixed<T,B>) -> Self::Output {
                Fixed(T::from(self) * other.0)
            }
        }

        // fixed * scalar
        impl<T: From<$t> + Mul<Output=T>,const B: usize> Mul<$t> for Fixed<T,B> {
            type Output = Self;
            fn mul(self,other: $t) -> Self::Output {
                Fixed(self.0 * T::from(other))
            }
        }

        // fixed *= scalar
        impl<T: From<$t> + MulAssign,const B: usize> MulAssign<$t> for Fixed<T,B> {
            fn mul_assign(&mut self,other: $t) {
                self.0 *= T::from(other);
            }
        }

        // scalar / fixed
        impl<T: From<$t> + Div<Output=T> + Shl<usize,Output=T>,const B: usize> Div<Fixed<T,B>> for $t {
            type Output = Fixed<T,B>;
            fn div(self,other: Fixed<T,B>) -> Self::Output {
                Fixed((T::from(self) << (2 * B)) / other.0)
            }
        }

        // fixed / scalar
        impl<T: From<$t> + Div<Output=T>,const B: usize> Div<$t> for Fixed<T,B> {
            type Output = Self;
            fn div(self,other: $t) -> Self::Output {
                Fixed(self.0 / T::from(other))
            }
        }

        // fixed /= scalar
        impl<T: From<$t> + DivAssign,const B: usize> DivAssign<$t> for Fixed<T,B> {
            fn div_assign(&mut self,other: $t) {
                self.0 /= T::from(other);
            }
        }
    )*)
}

fixed_impl! { usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 }
*/

impl<T: Zero,const B: usize> Zero for Fixed<T,B> {
    const ZERO: Self = Fixed(T::ZERO);
}

impl<T: One + Shl<usize,Output=T>,const B: usize> One for Fixed<T,B> {
    const ONE: Self = Fixed(T::ONE << B);
}

// fixed == fixed
impl<T: PartialEq,const B: usize> PartialEq<Fixed<T,B>> for Fixed<T,B> {
    fn eq(&self,other: &Fixed<T,B>) -> bool {
        self.0 == other.0
    }
}

// fixed ? fixed
impl<T: PartialOrd,const B: usize> PartialOrd<Fixed<T,B>> for Fixed<T,B> {
    fn partial_cmp(&self,other: &Fixed<T,B>) -> Option<Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

// fixed + fixed
impl<T: Add<Output=T>,const B: usize> Add<Fixed<T,B>> for Fixed<T,B> {
    type Output = Self;
    fn add(self,other: Self) -> Self::Output {
        Fixed(self.0 + other.0)
    }
}

// fixed += fixed
impl<T: AddAssign,const B: usize> AddAssign<Fixed<T,B>> for Fixed<T,B> {
    fn add_assign(&mut self,other: Fixed<T,B>) {
        self.0 += other.0;
    }
}

// fixed - fixed
impl<T: Sub<Output=T>,const B: usize> Sub<Fixed<T,B>> for Fixed<T,B> {
    type Output = Self;
    fn sub(self,other: Self) -> Self::Output {
        Fixed(self.0 - other.0)
    }
}

// fixed -= fixed
impl<T: SubAssign,const B: usize> SubAssign<Fixed<T,B>> for Fixed<T,B> {
    fn sub_assign(&mut self,other: Fixed<T,B>) {
        self.0 -= other.0;
    }
}

// fixed * fixed
impl<T: Mul<Output=T> + Shr<usize,Output=T>,const B: usize> Mul<Fixed<T,B>> for Fixed<T,B> {
    type Output = Self;
    fn mul(self,other: Self) -> Self::Output {
        Fixed(self.0 * other.0 >> B)
    }
}

// fixed *= fixed
impl<T: Copy + Mul<Output=T> + Shr<usize,Output=T>,const B: usize> MulAssign<Fixed<T,B>> for Fixed<T,B> {
    fn mul_assign(&mut self,other: Fixed<T,B>) {
        self.0 = (self.0 * other.0) >> B;
    }
}

// fixed / fixed
impl<T: Div<Output=T> + Shl<usize,Output=T>,const B: usize> Div<Fixed<T,B>> for Fixed<T,B> {
    type Output = Self;
    fn div(self,other: Self) -> Self::Output {
        Fixed((self.0 << B) / other.0)
    }
}

// fixed /= fixed
impl<T: Copy + Div<Output=T> + Shl<usize,Output=T>,const B: usize> DivAssign<Fixed<T,B>> for Fixed<T,B> {
    fn div_assign(&mut self,other: Fixed<T,B>) {
        self.0 = (self.0 << B) / other.0;
    }
}

// -fixed
impl<T: Neg<Output=T>,const B: usize> Neg for Fixed<T,B> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Fixed(-self.0)
    }
}

macro_rules! fixed_real_impl {
    ($($t:ty)*) => ($(
        impl<const B: usize> Real for Fixed<$t,B> {

            const MIN: Self = Fixed(<$t>::MIN);
            const MAX: Self = Fixed(<$t>::MAX);
            const PI: Self = Fixed((f32::PI * ((1 << B) as f32)) as $t);

            fn floor(self) -> Self {
                Fixed(self.0 & !((1 << B) - 1))
            }
        
            fn ceil(self) -> Self {
                Fixed(self.0 & !((1 << B) - 1) + ((1 << B) - 1))
            }
        
            fn round(self) -> Self {
                Fixed((self.0 & !((1 << (B - 1)) - 1) - ((1 << B) - 1)) & !((1 << B) - 1))
            }
        
            fn trunc(self) -> Self {
                let mut result = Fixed(self.0 & !((1 << B) - 1));
                if (self.0 < 0) {
                    result.0 += (1 << B) - 1;
                }
                result
            }
        
            fn fract(self) -> Self {
                let mut result: Self = Fixed(self.0 & !((1 << B) - 1));
                if (self.0 < 0) {
                    result.0 += (1 << B) - 1;
                }
                Fixed(self.0 - result.0)
            }

            fn abs(self) -> Self {
                if (self.0 < 0) {
                    Fixed(-self.0)
                }
                else {
                    self
                }
            }

            fn signum(self) -> Self {
                if (self.0 < 0) {
                    Fixed(!((1 << B) - 1))
                }
                else {
                    Fixed((1 << B) - 1)
                }
            }

            fn copysign(self,sign: Self) -> Self {
                if (sign.0 < 0) {
                    if (self.0 < 0) {
                        self
                    }
                    else {
                        Fixed(-self.0)
                    }
                }
                else {
                    if (self.0 < 0) {
                        Fixed(-self.0)
                    }
                    else {
                        self
                    }
                }
            }

            fn mul_add(self,a: Self,b: Self) -> Self {
                self * a + b
            }

            fn div_euclid(self,rhs: Self) -> Self {
                Fixed(((self.0 >> B) / (rhs.0 >> B)) << B)
            }

            fn rem_euclid(self,rhs: Self) -> Self {
                self - self.div_euclid(rhs)
            }

            fn powi(self,n: i32) -> Self {
                let mut result = self;
                for _ in 0..(n - 1) {
                    result *= self;
                }
                result
            }

            fn powf(self,n: Self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let nf = (n.0 as f32) / f;
                let rf = sf.powf(nf);
                Fixed((rf * f) as $t)
            }

            fn sqrt(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.sqrt();
                Fixed((rf * f) as $t)
            }

            fn exp(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.exp();
                Fixed((rf * f) as $t)
            }

            fn exp2(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.exp2();
                Fixed((rf * f) as $t)
            }

            fn ln(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.ln();
                Fixed((rf * f) as $t)
            }

            fn log(self,base: Self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let bf = (base.0 as f32) / f;
                let rf = sf.log(bf);
                Fixed((rf * f) as $t)
            }

            fn log2(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.log2();
                Fixed((rf * f) as $t)
            }

            fn log10(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.log10();
                Fixed((rf * f) as $t)
            }

            fn cbrt(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.cbrt();
                Fixed((rf * f) as $t)
            }
            
            fn hypot(self,other: Self) -> Self {
                (self * self + other * other).sqrt()
            }

            fn sin(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.sin();
                Fixed((rf * f) as $t)
            }

            fn cos(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.cos();
                Fixed((rf * f) as $t)
            }

            fn tan(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.tan();
                Fixed((rf * f) as $t)
            }

            fn asin(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.asin();
                Fixed((rf * f) as $t)
            }

            fn acos(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.acos();
                Fixed((rf * f) as $t)
            }

            fn atan(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.atan();
                Fixed((rf * f) as $t)
            }

            fn atan2(self,other: Self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let of = (other.0 as f32) / f;
                let rf = sf.atan2(of);
                Fixed((rf * f) as $t)
            }

            fn sin_cos(self) -> (Self,Self) {
                (self.sin(),self.cos())
            }

            fn exp_m1(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.exp_m1();
                Fixed((rf * f) as $t)
            }

            fn ln_1p(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.ln_1p();
                Fixed((rf * f) as $t)
            }

            fn sinh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.sinh();
                Fixed((rf * f) as $t)
            }

            fn cosh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.cosh();
                Fixed((rf * f) as $t)
            }

            fn tanh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.tanh();
                Fixed((rf * f) as $t)
            }

            fn asinh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.asinh();
                Fixed((rf * f) as $t)
            }

            fn acosh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.acosh();
                Fixed((rf * f) as $t)
            }

            fn atanh(self) -> Self {
                let f = (1 << B) as f32;
                let sf = (self.0 as f32) / f;
                let rf = sf.atanh();
                Fixed((rf * f) as $t)
            }

            fn is_sign_positive(self) -> bool {
                self.0 >= 0
            }

            fn is_sign_negative(self) -> bool {
                self.0 < 0
            }

            fn inv(self) -> Self {
                Fixed(1 << B) / self
            }

            fn to_degrees(self) -> Self {
                Fixed(((self.0 * 180) << B) / Self::PI.0)
            }

            fn to_radians(self) -> Self {
                Fixed(((self.0 * Self::PI.0) >> B) / 180)
            }

            fn max(self,other: Self) -> Self {
                if (other > self) {
                    other
                }
                else {
                    self
                }
            }

            fn min(self,other: Self) -> Self {
                if (other < self) {
                    other
                }
                else {
                    self
                }
            }

            fn clamp(self,min: Self,max: Self) -> Self {
                self.max(min).min(max)
            }
        }
    )*)
}

fixed_real_impl! { isize i8 i16 i32 i64 i128 }
