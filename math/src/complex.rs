use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Debug, Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

#[derive(Copy, Clone, Debug)]
pub struct Complex<T> {
    pub r: T,
    pub i: T,
}

impl<T> Complex<T>
where
    T: Copy + Neg<Output = T>,
{
    /// Complex conjugate.
    pub fn conj(&self) -> Self {
        Complex {
            r: self.r,
            i: -self.i,
        }
    }
}

impl<T> Zero for Complex<T>
where
    T: Zero,
{
    const ZERO: Self = Complex {
        r: T::ZERO,
        i: T::ZERO,
    };
}

impl<T> One for Complex<T>
where
    T: Zero + One,
{
    const ONE: Self = Complex {
        r: T::ONE,
        i: T::ZERO,
    };
}

impl<T> Display for Complex<T>
where
    T: Zero + Display + PartialOrd,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        let si = if self.i < <T>::ZERO {
            format!("{}i", self.i)
        } else {
            format!("+{}i", self.i)
        };
        write!(f, "{}{}", self.r, si)
    }
}

impl<T> PartialEq<Complex<T>> for Complex<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        (self.r == other.r) && (self.i == other.i)
    }

    fn ne(&self, other: &Self) -> bool {
        (self.r != other.r) || (self.i != other.i)
    }
}

/// Complex + real.
impl<T> Add<T> for Complex<T>
where
    T: Add<Output = T>,
{
    type Output = Self;
    fn add(self, other: T) -> Self::Output {
        Complex {
            r: self.r + other,
            i: self.i,
        }
    }
}

/// Complex + complex.
impl<T> Add<Complex<T>> for Complex<T>
where
    T: Add<Output = T>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Complex {
            r: self.r + other.r,
            i: self.i + other.i,
        }
    }
}

/// Complex += real.
impl<T> AddAssign<T> for Complex<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: T) {
        self.r += other;
    }
}

/// Complex += complex.
impl<T> AddAssign<Complex<T>> for Complex<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.i += other.i;
    }
}

/// Complex - real.
impl<T> Sub<T> for Complex<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: T) -> Self::Output {
        Complex {
            r: self.r - other,
            i: self.i,
        }
    }
}

/// Complex - complex.
impl<T> Sub<Complex<T>> for Complex<T>
where
    T: Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Complex {
            r: self.r - other.r,
            i: self.i - other.i,
        }
    }
}

/// Complex -= real.
impl<T> SubAssign<T> for Complex<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: T) {
        self.r -= other;
    }
}

/// Complex -= complex.
impl<T> SubAssign<Complex<T>> for Complex<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.r -= other.r;
        self.i -= other.i;
    }
}

/// Complex * real.
impl<T> Mul<T> for Complex<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Self;
    fn mul(self, other: T) -> Self::Output {
        Complex {
            r: self.r * other,
            i: self.i * other,
        }
    }
}

/// Complex * complex.
impl<T> Mul<Complex<T>> for Complex<T>
where
    T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Complex {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
        }
    }
}

/// Complex *= real.
impl<T> MulAssign<T> for Complex<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, other: T) {
        self.r *= other;
        self.i *= other;
    }
}

/// Complex *= complex.
impl<T> MulAssign<Complex<T>> for Complex<T>
where
    T: Copy + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
    fn mul_assign(&mut self, other: Self) {
        let r = self.r * other.r - self.i * other.i;
        let i = self.r * other.i + self.i * other.r;
        self.r = r;
        self.i = i;
    }
}

/// Complex / real.
impl<T> Div<T> for Complex<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Self;
    fn div(self, other: T) -> Self::Output {
        Complex {
            r: self.r / other,
            i: self.i / other,
        }
    }
}

/// Complex / complex.
impl<T> Div<Complex<T>> for Complex<T>
where
    T: Copy + Div<Output = T> + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
    type Output = Self;
    fn div(self, other: Self) -> Self::Output {
        let f = other.r * other.r + other.i * other.i;
        Complex {
            r: (self.r * other.r + self.i * other.i) / f,
            i: (self.i * other.r - self.r * other.i) / f,
        }
    }
}

/// Complex /= real.
impl<T> DivAssign<T> for Complex<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, other: T) {
        self.r /= other;
        self.i /= other;
    }
}

/// Complex /= complex.
impl<T> DivAssign<Complex<T>> for Complex<T>
where
    T: Copy + Div<Output = T> + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
    fn div_assign(&mut self, other: Self) {
        let f = other.r * other.r + other.i * other.i;
        let r = (self.r * other.r + self.i * other.i) / f;
        let i = (self.i * other.r - self.r * other.i) / f;
        self.r = r;
        self.i = i;
    }
}

/// -Complex.
impl<T> Neg for Complex<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Complex {
            r: -self.r,
            i: -self.i,
        }
    }
}

macro_rules! complex_impl {
    ($($t:ty)+) => {
        $(
            impl Complex<$t> {

                /// Complex argument.
                pub fn arg(&self) -> $t {
                    self.r.atan2(self.i)
                }
            }

            /// Real + complex.
            impl Add<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn add(self,other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self + other.r,
                        i: other.i,
                    }
                }
            }

            /// Real - complex.
            impl Sub<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn sub(self,other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self - other.r,
                        i: -other.i,
                    }
                }
            }

            /// Real * complex.
            impl Mul<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn mul(self,other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self * other.r,
                        i: self * other.i,
                    }
                }
            }

            /// Real / complex.
            impl Div<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn div(self,other: Complex<$t>) -> Self::Output {
                    let f = other.r * other.r + other.i * other.i;
                    Complex {
                        r: self * other.r / f,
                        i: -self * other.i / f,
                    }
                }
            }
        )+
    }
}

complex_impl! { f32 f64 }

// if `T as U` exists, `Complex<U>::from(Complex<T>)` should also exist
impl From<Complex<f32>> for Complex<f64> {
    fn from(value: Complex<f32>) -> Self {
        Complex {
            r: value.r as f64,
            i: value.i as f64,
        }
    }
}

impl From<Complex<f64>> for Complex<f32> {
    fn from(value: Complex<f64>) -> Self {
        Complex {
            r: value.r as f32,
            i: value.i as f32,
        }
    }
}

/// Complex number built from `f32`s.
#[allow(non_camel_case_types)]
pub type c32 = Complex<f32>;

/// Complex number built from `f64`s.
#[allow(non_camel_case_types)]
pub type c64 = Complex<f64>;

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn add_rc() {
        let result = 4.0 + Complex::<f32> { r: 5.0, i: 2.0 };
        assert_eq!(result, Complex::<f32> { r: 9.0, i: 2.0 });
    }

    #[test]
    fn add_cr() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } + 4.0;
        assert_eq!(result, Complex::<f32> { r: 9.0, i: 2.0 });
    }

    #[test]
    fn add_cc() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } + Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 9.0, i: 5.0 });
    }

    #[test]
    fn add_assign_cr() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result += 4.0;
        assert_eq!(result, Complex::<f32> { r: 9.0, i: 2.0 });
    }

    #[test]
    fn add_assign_cc() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result += Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 9.0, i: 5.0 });
    }

    #[test]
    fn sub_rc() {
        let result = 4.0 - Complex::<f32> { r: 5.0, i: 2.0 };
        assert_eq!(result, Complex::<f32> { r: -1.0, i: -2.0 });
    }

    #[test]
    fn sub_cr() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } - 4.0;
        assert_eq!(result, Complex::<f32> { r: 1.0, i: 2.0 });
    }

    #[test]
    fn sub_cc() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } - Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 1.0, i: -1.0 });
    }

    #[test]
    fn sub_assign_cr() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result -= 4.0;
        assert_eq!(result, Complex::<f32> { r: 1.0, i: 2.0 });
    }

    #[test]
    fn sub_assign_cc() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result -= Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 1.0, i: -1.0 });
    }

    #[test]
    fn mul_rc() {
        let result = 4.0 * Complex::<f32> { r: 5.0, i: 2.0 };
        assert_eq!(result, Complex::<f32> { r: 20.0, i: 8.0 });
    }

    #[test]
    fn mul_cr() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } * 4.0;
        assert_eq!(result, Complex::<f32> { r: 20.0, i: 8.0 });
    }

    #[test]
    fn mul_cc() {
        let result = Complex::<f32> { r: 5.0, i: 2.0 } * Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 14.0, i: 23.0 });
    }

    #[test]
    fn mul_assign_cr() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result *= 4.0;
        assert_eq!(result, Complex::<f32> { r: 20.0, i: 8.0 });
    }

    #[test]
    fn mul_assign_cc() {
        let mut result = Complex::<f32> { r: 5.0, i: 2.0 };
        result *= Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 14.0, i: 23.0 });
    }

    #[test]
    fn div_rc() {
        let result = 25.0 / Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 4.0, i: -3.0 });
    }

    #[test]
    fn div_cr() {
        let result = Complex::<f32> { r: 25.0, i: 2.0 } / 5.0;
        assert_eq!(result, Complex::<f32> { r: 5.0, i: 0.4 });
    }

    #[test]
    fn div_cc() {
        let result = Complex::<f32> { r: 14.0, i: 23.0 } / Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 5.0, i: 2.0 });
    }

    #[test]
    fn div_assign_cr() {
        let mut result = Complex::<f32> { r: 25.0, i: 10.0 };
        result /= 5.0;
        assert_eq!(result, Complex::<f32> { r: 5.0, i: 2.0 });
    }

    #[test]
    fn div_assign_cc() {
        let mut result = Complex::<f32> { r: 14.0, i: 23.0 };
        result /= Complex::<f32> { r: 4.0, i: 3.0 };
        assert_eq!(result, Complex::<f32> { r: 5.0, i: 2.0 });
    }

    #[test]
    fn neg() {
        let result = -Complex::<f32> { r: 5.0, i: 4.0 };
        assert_eq!(result, Complex::<f32> { r: -5.0, i: -4.0 })
    }
}
