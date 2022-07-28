use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Debug,
            Formatter,
            Result
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
            Neg
        },
    },
};

#[derive(Copy,Clone,Debug)]
pub struct Complex<T> {
    pub r: T,
    pub i: T,
}

impl<T> Complex<T> {
    pub fn new(r: T,i: T) -> Complex<T> {
        Complex {
            r: r,
            i: i,
        }
    }
}

impl<T: PartialEq> PartialEq for Complex<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) &&
        (self.i == other.i)
    }
}

impl<T: Zero> Zero for Complex<T> { const ZERO: Self = Complex { r: T::ZERO,i: T::ZERO, }; }

impl<T: Zero + Display + PartialOrd> Display for Complex<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        let si = if self.i < T::ZERO {
            format!("{}i",self.i)
        } else {
            format!("+{}i",self.i)
        };
        write!(f,"{}{}",self.r,si)
    }
}

macro_rules! scalar_complex_mul {
    ($t:ty) => {
        impl Add<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn add(self,other: Complex<$t>) -> Complex<$t> {
                Complex {
                    r: self + other.r,
                    i: other.i,
                }
            }
        }
    }
}

scalar_complex_mul!(f32);
scalar_complex_mul!(f64);

impl<T: Add<T,Output=T>> Add<T> for Complex<T> {
    type Output = Self;
    fn add(self,other: T) -> Self {
        Complex::new(self.r + other,self.i)
    }
}

impl<T: Add<T,Output=T>> Add<Complex<T>> for Complex<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Complex {
            r: self.r + other.r,
            i: self.i + other.i,
        }
    }
}

impl<T: AddAssign<T>> AddAssign<T> for Complex<T> {
    fn add_assign(&mut self,other: T) {
        self.r += other;
    }
}

impl<T: AddAssign<T>> AddAssign<Complex<T>> for Complex<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.i += other.i;
    }
}

macro_rules! scalar_complex_sub {
    ($t:ty) => {
        impl Sub<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn sub(self,other: Complex<$t>) -> Complex<$t> {
                Complex {
                    r: self - other.r,
                    i: -other.i,
                }
            }
        }        
    }
}

scalar_complex_sub!(f32);
scalar_complex_sub!(f64);

impl<T: Sub<T,Output=T>> Sub<T> for Complex<T> {
    type Output = Self;
    fn sub(self,other: T) -> Self {
        Complex {
            r: self.r - other,
            i: self.i,
        }
    }
}

impl<T: Sub<T,Output=T>> Sub<Complex<T>> for Complex<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Complex {
            r: self.r - other.r,
            i: self.i - other.i,
        }
    }
}

impl<T: SubAssign<T>> SubAssign<T> for Complex<T> {
    fn sub_assign(&mut self,other: T) {
        self.r -= other;
    }
}

impl<T: SubAssign<T>> SubAssign<Complex<T>> for Complex<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.i -= other.i;
    }
}

macro_rules! scalar_complex_mul {
    ($t:ty) => {
        impl Mul<Complex<$t>> for $t {
            type Output = Complex<$t>;
            fn mul(self,other: Complex<$t>) -> Complex<$t> {
                Complex {
                    r: self * other.r,
                    i: self * other.i,
                }
            }
        }        
    }
}

scalar_complex_mul!(f32);
scalar_complex_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Complex<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Complex {
            r: self.r * other,
            i: self.i * other,
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> Mul<Complex<T>> for Complex<T> {
    type Output = Self;
    fn mul(self,other: Self) -> Self {
        Complex {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
        }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Complex<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.i *= other;
    }
} 

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> MulAssign<Complex<T>> for Complex<T> {
    fn mul_assign(&mut self,other: Complex<T>) {
        let r = self.r * other.r - self.i * other.i;
        let i = self.r * other.i + self.i * other.r;
        self.r = r;
        self.i = i;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Complex<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Complex {
            r: self.r / other,
            i: self.i / other,
        }
    }
}

// TODO: Complex / Complex

impl<T: Copy + DivAssign<T>> DivAssign<T> for Complex<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.i /= other;
    }
}

// TODO: Complex /= Complex

impl<T: Neg<Output=T>> Neg for Complex<T> {
    type Output = Complex<T>;
    fn neg(self) -> Complex<T> {
        Complex {
            r: -self.r,
            i: -self.i,
        }
    }
}

#[macro_export]
macro_rules! complex {
    ($r:expr,$i:expr) => { Complex::new($r,$i) };
}
