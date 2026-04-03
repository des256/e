use {
    crate::*,
    codec::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// Complex number, generic over the component type.
///
/// Supports full arithmetic with both scalars and other complex numbers,
/// including `+`, `-`, `*`, `/` and their assign variants. Scalar operands
/// may appear on either side (e.g. `2.0 * z` and `z * 2.0` both work).
///
/// For `f32`/`f64`, additional methods are available: [`norm`](Complex::norm)
/// (modulus) and [`arg`](Complex::arg) (argument / phase angle).
///
/// Type aliases [`c32`] and [`c64`] are provided for convenience.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let a = complex(3.0f32, 4.0);
/// assert_eq!(a.norm(), 5.0);
/// assert_eq!(a.conj(), complex(3.0, -4.0));
/// ```
#[derive(Copy, Clone, Debug, Codec)]
pub struct Complex<T> {
    /// Real part.
    pub r: T,
    /// Imaginary part.
    pub i: T,
}

/// Create a new complex number.
pub const fn complex<T>(r: T, i: T) -> Complex<T> {
    Complex { r, i }
}

macro_rules! complex_impl {
    ($($t:ty)+) => {
        $(
            impl Complex<$t> {
                /// Complex conjugate.
                pub fn conj(self) -> Self {
                    Complex {
                        r: self.r,
                        i: -self.i,
                    }
                }

                /// Complex modulus (absolute value).
                pub fn norm(&self) -> $t {
                    (self.r * self.r + self.i * self.i).sqrt()
                }

                /// Complex argument.
                pub fn arg(&self) -> $t {
                    self.i.atan2(self.r)
                }
            }

            impl Zero for Complex<$t> {
                const ZERO: Self = Complex {
                    r: 0.0,
                    i: 0.0,
                };
            }

            impl One for Complex<$t> {
                const ONE: Self = Complex {
                    r: 1.0,
                    i: 0.0,
                };
            }

            impl Display for Complex<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    let si = if self.i < 0.0 {
                        format!("{}i", self.i)
                    } else {
                        format!("+{}i", self.i)
                    };
                    write!(f, "{}{}", self.r, si)
                }
            }

            impl PartialEq<Complex<$t>> for Complex<$t> {
                fn eq(&self, other: &Self) -> bool {
                    (self.r == other.r) && (self.i == other.i)
                }

                fn ne(&self, other: &Self) -> bool {
                    (self.r != other.r) || (self.i != other.i)
                }
            }

            /// Complex + real.
            impl Add<$t> for Complex<$t> {
                type Output = Self;
                fn add(self, other: $t) -> Self::Output {
                    Complex {
                        r: self.r + other,
                        i: self.i,
                    }
                }
            }

            /// Complex + complex.
            impl Add<Complex<$t>> for Complex<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self::Output {
                    Complex {
                        r: self.r + other.r,
                        i: self.i + other.i,
                    }
                }
            }

            /// Complex += real.
            impl AddAssign<$t> for Complex<$t> {
                fn add_assign(&mut self, other: $t) {
                    self.r += other;
                }
            }

            /// Complex += complex.
            impl AddAssign<Complex<$t>> for Complex<$t> {
                fn add_assign(&mut self, other: Self) {
                    self.r += other.r;
                    self.i += other.i;
                }
            }

            /// Complex - real.
            impl Sub<$t> for Complex<$t> {
                type Output = Self;
                fn sub(self, other: $t) -> Self::Output {
                    Complex {
                        r: self.r - other,
                        i: self.i,
                    }
                }
            }

            /// Complex - complex.
            impl Sub<Complex<$t>> for Complex<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self::Output {
                    Complex {
                        r: self.r - other.r,
                        i: self.i - other.i,
                    }
                }
            }

            /// Complex -= real.
            impl SubAssign<$t> for Complex<$t> {
                fn sub_assign(&mut self, other: $t) {
                    self.r -= other;
                }
            }

            /// Complex -= complex.
            impl SubAssign<Complex<$t>> for Complex<$t> {
                fn sub_assign(&mut self, other: Self) {
                    self.r -= other.r;
                    self.i -= other.i;
                }
            }

            /// Complex * real.
            impl Mul<$t> for Complex<$t> {
                type Output = Self;
                fn mul(self, other: $t) -> Self::Output {
                    Complex {
                        r: self.r * other,
                        i: self.i * other,
                    }
                }
            }

            /// Complex * complex.
            impl Mul<Complex<$t>> for Complex<$t> {
                type Output = Self;
                fn mul(self, other: Self) -> Self::Output {
                    Complex {
                        r: self.r * other.r - self.i * other.i,
                        i: self.r * other.i + self.i * other.r,
                    }
                }
            }

            /// Complex *= real.
            impl MulAssign<$t> for Complex<$t> {
                fn mul_assign(&mut self, other: $t) {
                    self.r *= other;
                    self.i *= other;
                }
            }

            /// Complex *= complex.
            impl MulAssign<Complex<$t>> for Complex<$t> {
                fn mul_assign(&mut self, other: Self) {
                    let r = self.r * other.r - self.i * other.i;
                    let i = self.r * other.i + self.i * other.r;
                    self.r = r;
                    self.i = i;
                }
            }

            /// Complex / real.
            impl Div<$t> for Complex<$t> {
                type Output = Self;
                fn div(self, other: $t) -> Self::Output {
                    Complex {
                        r: self.r / other,
                        i: self.i / other,
                    }
                }
            }

            /// Complex / complex.
            impl Div<Complex<$t>> for Complex<$t> {
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
            impl DivAssign<$t> for Complex<$t> {
                fn div_assign(&mut self, other: $t) {
                    self.r /= other;
                    self.i /= other;
                }
            }

            /// Complex /= complex.
            impl DivAssign<Complex<$t>> for Complex<$t> {
                fn div_assign(&mut self, other: Self) {
                    let f = other.r * other.r + other.i * other.i;
                    let r = (self.r * other.r + self.i * other.i) / f;
                    let i = (self.i * other.r - self.r * other.i) / f;
                    self.r = r;
                    self.i = i;
                }
            }

            /// -Complex.
            impl Neg for Complex<$t> {
                type Output = Self;
                fn neg(self) -> Self {
                    Complex {
                        r: -self.r,
                        i: -self.i,
                    }
                }
            }

            /// Real + complex.
            impl Add<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn add(self, other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self + other.r,
                        i: other.i,
                    }
                }
            }

            /// Real - complex.
            impl Sub<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn sub(self, other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self - other.r,
                        i: -other.i,
                    }
                }
            }

            /// Real * complex.
            impl Mul<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn mul(self, other: Complex<$t>) -> Self::Output {
                    Complex {
                        r: self * other.r,
                        i: self * other.i,
                    }
                }
            }

            /// Real / complex.
            impl Div<Complex<$t>> for $t {
                type Output = Complex<$t>;
                fn div(self, other: Complex<$t>) -> Self::Output {
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

    #[test]
    fn test_codec_complex_roundtrip() {
        let val = Complex { r: 3.0f32, i: 4.0 };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Complex::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded.r, val.r);
        assert_eq!(decoded.i, val.i);
    }
}
