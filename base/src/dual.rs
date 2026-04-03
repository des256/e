use {
    crate::*,
    codec::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// Dual number for automatic differentiation.
///
/// A dual number has the form `a + bε`, where `ε² = 0`. Arithmetic on dual numbers
/// automatically computes derivatives: if `f(a + ε) = f(a) + f'(a)ε`, then the
/// dual part carries the derivative.
///
/// Type aliases [`d32`] and [`d64`] are provided for convenience.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// // Compute f(x) = x² and f'(x) = 2x at x = 3
/// let x = dual(3.0f32, 1.0);  // seed with dual = 1
/// let f = x * x;
/// assert_eq!(f.real, 9.0);   // f(3) = 9
/// assert_eq!(f.dual, 6.0);   // f'(3) = 6
/// ```
#[derive(Copy, Clone, Debug, Codec)]
pub struct Dual<T> {
    /// Real part (function value).
    pub real: T,
    /// Dual part (derivative).
    pub dual: T,
}

/// Create a new dual number.
pub const fn dual<T>(real: T, d: T) -> Dual<T> {
    Dual { real, dual: d }
}

macro_rules! dual_impl {
    ($($t:ty)+) => {
        $(
            impl PartialEq for Dual<$t> {
                fn eq(&self, other: &Self) -> bool {
                    self.real == other.real && self.dual == other.dual
                }
            }

            impl Zero for Dual<$t> {
                const ZERO: Self = Dual { real: 0.0, dual: 0.0 };
            }

            impl One for Dual<$t> {
                const ONE: Self = Dual { real: 1.0, dual: 0.0 };
            }

            impl Display for Dual<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    if self.dual < 0.0 {
                        write!(f, "{}{}ε", self.real, self.dual)
                    } else {
                        write!(f, "{}+{}ε", self.real, self.dual)
                    }
                }
            }

            /// -Dual.
            impl Neg for Dual<$t> {
                type Output = Self;
                fn neg(self) -> Self { Dual { real: -self.real, dual: -self.dual } }
            }

            /// Dual + scalar.
            impl Add<$t> for Dual<$t> {
                type Output = Self;
                fn add(self, other: $t) -> Self { Dual { real: self.real + other, dual: self.dual } }
            }

            /// Scalar + dual.
            impl Add<Dual<$t>> for $t {
                type Output = Dual<$t>;
                fn add(self, other: Dual<$t>) -> Dual<$t> { Dual { real: self + other.real, dual: other.dual } }
            }

            /// Dual + dual.
            impl Add for Dual<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self { Dual { real: self.real + other.real, dual: self.dual + other.dual } }
            }

            /// Dual += scalar.
            impl AddAssign<$t> for Dual<$t> {
                fn add_assign(&mut self, other: $t) { self.real += other; }
            }

            /// Dual += dual.
            impl AddAssign for Dual<$t> {
                fn add_assign(&mut self, other: Self) { self.real += other.real; self.dual += other.dual; }
            }

            /// Dual - scalar.
            impl Sub<$t> for Dual<$t> {
                type Output = Self;
                fn sub(self, other: $t) -> Self { Dual { real: self.real - other, dual: self.dual } }
            }

            /// Scalar - dual.
            impl Sub<Dual<$t>> for $t {
                type Output = Dual<$t>;
                fn sub(self, other: Dual<$t>) -> Dual<$t> { Dual { real: self - other.real, dual: -other.dual } }
            }

            /// Dual - dual.
            impl Sub for Dual<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self { Dual { real: self.real - other.real, dual: self.dual - other.dual } }
            }

            /// Dual -= scalar.
            impl SubAssign<$t> for Dual<$t> {
                fn sub_assign(&mut self, other: $t) { self.real -= other; }
            }

            /// Dual -= dual.
            impl SubAssign for Dual<$t> {
                fn sub_assign(&mut self, other: Self) { self.real -= other.real; self.dual -= other.dual; }
            }

            /// Dual * scalar.
            impl Mul<$t> for Dual<$t> {
                type Output = Self;
                fn mul(self, other: $t) -> Self { Dual { real: self.real * other, dual: self.dual * other } }
            }

            /// Scalar * dual.
            impl Mul<Dual<$t>> for $t {
                type Output = Dual<$t>;
                fn mul(self, other: Dual<$t>) -> Dual<$t> { Dual { real: self * other.real, dual: self * other.dual } }
            }

            /// Dual * dual: (a + bε)(c + dε) = ac + (ad + bc)ε.
            impl Mul for Dual<$t> {
                type Output = Self;
                fn mul(self, other: Self) -> Self {
                    Dual {
                        real: self.real * other.real,
                        dual: self.real * other.dual + self.dual * other.real,
                    }
                }
            }

            /// Dual *= scalar.
            impl MulAssign<$t> for Dual<$t> {
                fn mul_assign(&mut self, other: $t) { self.real *= other; self.dual *= other; }
            }

            /// Dual *= dual.
            impl MulAssign for Dual<$t> {
                fn mul_assign(&mut self, other: Self) {
                    let real = self.real * other.real;
                    let dual = self.real * other.dual + self.dual * other.real;
                    self.real = real;
                    self.dual = dual;
                }
            }

            /// Dual / scalar.
            impl Div<$t> for Dual<$t> {
                type Output = Self;
                fn div(self, other: $t) -> Self { Dual { real: self.real / other, dual: self.dual / other } }
            }

            /// Scalar / dual.
            impl Div<Dual<$t>> for $t {
                type Output = Dual<$t>;
                fn div(self, other: Dual<$t>) -> Dual<$t> {
                    let c2 = other.real * other.real;
                    Dual { real: self / other.real, dual: -self * other.dual / c2 }
                }
            }

            /// Dual / dual: (a + bε)/(c + dε) = a/c + (bc - ad)/c²ε.
            impl Div for Dual<$t> {
                type Output = Self;
                fn div(self, other: Self) -> Self {
                    let c2 = other.real * other.real;
                    Dual {
                        real: self.real / other.real,
                        dual: (self.dual * other.real - self.real * other.dual) / c2,
                    }
                }
            }

            /// Dual /= scalar.
            impl DivAssign<$t> for Dual<$t> {
                fn div_assign(&mut self, other: $t) { self.real /= other; self.dual /= other; }
            }

            /// Dual /= dual.
            impl DivAssign for Dual<$t> {
                fn div_assign(&mut self, other: Self) {
                    let c2 = other.real * other.real;
                    let real = self.real / other.real;
                    let dual = (self.dual * other.real - self.real * other.dual) / c2;
                    self.real = real;
                    self.dual = dual;
                }
            }
        )+
    }
}

dual_impl! { f32 f64 }

impl From<Dual<f32>> for Dual<f64> {
    fn from(value: Dual<f32>) -> Self {
        Dual {
            real: value.real as f64,
            dual: value.dual as f64,
        }
    }
}

impl From<Dual<f64>> for Dual<f32> {
    fn from(value: Dual<f64>) -> Self {
        Dual {
            real: value.real as f32,
            dual: value.dual as f32,
        }
    }
}

/// Dual number built from `f32`s.
#[allow(non_camel_case_types)]
pub type d32 = Dual<f32>;

/// Dual number built from `f64`s.
#[allow(non_camel_case_types)]
pub type d64 = Dual<f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_add() {
        let a = dual(3.0f32, 1.0);
        let b = dual(2.0, 4.0);
        assert_eq!(a + b, dual(5.0, 5.0));
    }

    #[test]
    fn test_dual_mul() {
        let x = dual(3.0f32, 1.0);
        let result = x * x;
        assert_eq!(result.real, 9.0);
        assert_eq!(result.dual, 6.0);
    }

    #[test]
    fn test_dual_div() {
        let a = dual(6.0f32, 1.0);
        let b = dual(3.0, 0.0);
        let result = a / b;
        assert_eq!(result.real, 2.0);
        assert!((result.dual - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_dual_neg() {
        let a = dual(3.0f32, 1.0);
        assert_eq!(-a, dual(-3.0, -1.0));
    }

    #[test]
    fn test_dual_chain_rule() {
        let x = dual(2.0f32, 1.0);
        let xp1 = x + 1.0;
        let result = xp1 * xp1;
        assert_eq!(result.real, 9.0);
        assert_eq!(result.dual, 6.0);
    }

    #[test]
    fn test_codec_dual_roundtrip() {
        let val = Dual {
            real: 1.0f64,
            dual: 0.5,
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Dual::<f64>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded.real, val.real);
        assert_eq!(decoded.dual, val.dual);
    }
}
