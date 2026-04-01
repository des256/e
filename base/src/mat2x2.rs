use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 2x2 column-major matrix, generic over the component type.
///
/// Columns are stored as [`Vec2`] fields `x` and `y`. Supports matrix
/// arithmetic (`+`, `-`, `*`, `/`), scalar scaling, matrix-vector
/// multiplication, transpose, and for `f32`/`f64`: [`det`](Mat2x2::det)
/// and [`inv`](Mat2x2::inv).
///
/// Can be constructed from a [`Complex`] number (rotation matrix).
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let m = Mat2x2::<f32>::ONE;
/// let v = m * vec2(3.0, 4.0);
/// assert_eq!(v, vec2(3.0, 4.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Mat2x2<T> {
    /// First column.
    pub x: Vec2<T>,
    /// Second column.
    pub y: Vec2<T>,
}

/// Create a new 2x2 matrix from column vectors.
pub const fn mat2x2<T>(x: Vec2<T>, y: Vec2<T>) -> Mat2x2<T> {
    Mat2x2 { x, y }
}

macro_rules! mat2x2_impl {
    ($($t:ty)+) => {
        $(
            impl Mat2x2<$t> {
                /// Transpose the matrix.
                pub fn transpose(self) -> Self {
                    Mat2x2 {
                        x: Vec2 {
                            x: self.x.x,
                            y: self.y.x,
                        },
                        y: Vec2 {
                            x: self.x.y,
                            y: self.y.y,
                        },
                    }
                }

                /// Calculate determinant of matrix.
                pub fn det(self) -> $t {
                    self.x.x * self.y.y - self.y.x * self.x.y
                }

                /// Calculate inverse of matrix.
                pub fn inv(self) -> Self {
                    let a = self.x.x;
                    let b = self.x.y;
                    let c = self.y.x;
                    let d = self.y.y;
                    let det = a * d - c * b;
                    if det == 0.0 {
                        return self;
                    }
                    Mat2x2 {
                        x: Vec2 { x: d, y: -b },
                        y: Vec2 { x: -c, y: a },
                    } / det
                }
            }

            impl Zero for Mat2x2<$t> {
                const ZERO: Mat2x2<$t> = Mat2x2 {
                    x: Vec2 { x: 0.0, y: 0.0 },
                    y: Vec2 { x: 0.0, y: 0.0 },
                };
            }

            impl One for Mat2x2<$t> {
                const ONE: Mat2x2<$t> = Mat2x2 {
                    x: Vec2 { x: 1.0, y: 0.0 },
                    y: Vec2 { x: 0.0, y: 1.0 },
                };
            }

            impl Display for Mat2x2<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "[{},{}]", self.x, self.y)
                }
            }

            /// Matrix + matrix.
            impl Add<Mat2x2<$t>> for Mat2x2<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self::Output {
                    Mat2x2 {
                        x: self.x + other.x,
                        y: self.y + other.y,
                    }
                }
            }

            /// Matrix += matrix.
            impl AddAssign<Mat2x2<$t>> for Mat2x2<$t> {
                fn add_assign(&mut self, other: Self) {
                    self.x += other.x;
                    self.y += other.y;
                }
            }

            /// Matrix - matrix.
            impl Sub<Mat2x2<$t>> for Mat2x2<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self::Output {
                    Mat2x2 {
                        x: self.x - other.x,
                        y: self.y - other.y,
                    }
                }
            }

            /// Matrix -= matrix.
            impl SubAssign<Mat2x2<$t>> for Mat2x2<$t> {
                fn sub_assign(&mut self, other: Self) {
                    self.x -= other.x;
                    self.y -= other.y;
                }
            }

            /// Matrix * scalar.
            impl Mul<$t> for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn mul(self, other: $t) -> Self::Output {
                    Mat2x2 {
                        x: self.x * other,
                        y: self.y * other,
                    }
                }
            }

            /// Scalar * matrix.
            impl Mul<Mat2x2<$t>> for $t {
                type Output = Mat2x2<$t>;
                fn mul(self, other: Mat2x2<$t>) -> Self::Output {
                    Mat2x2 {
                        x: self * other.x,
                        y: self * other.y,
                    }
                }
            }

            /// Matrix * vector.
            impl Mul<Vec2<$t>> for Mat2x2<$t> {
                type Output = Vec2<$t>;
                fn mul(self, other: Vec2<$t>) -> Self::Output {
                    Vec2 {
                        x: self.x.x * other.x + self.y.x * other.y,
                        y: self.x.y * other.x + self.y.y * other.y,
                    }
                }
            }

            /// Matrix * matrix.
            impl Mul<Mat2x2<$t>> for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn mul(self, other: Mat2x2<$t>) -> Self::Output {
                    Mat2x2 {
                        x: self * other.x,
                        y: self * other.y,
                    }
                }
            }

            /// Matrix *= scalar.
            impl MulAssign<$t> for Mat2x2<$t> {
                fn mul_assign(&mut self, other: $t) {
                    self.x *= other;
                    self.y *= other;
                }
            }

            /// Matrix *= matrix.
            impl MulAssign<Mat2x2<$t>> for Mat2x2<$t> {
                fn mul_assign(&mut self, other: Mat2x2<$t>) {
                    let m = *self * other;
                    *self = m;
                }
            }

            /// Matrix / scalar.
            impl Div<$t> for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn div(self, other: $t) -> Self::Output {
                    Mat2x2 {
                        x: self.x / other,
                        y: self.y / other,
                    }
                }
            }

            /// Scalar / matrix.
            impl Div<Mat2x2<$t>> for $t {
                type Output = Mat2x2<$t>;
                fn div(self, other: Mat2x2<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix / matrix.
            impl Div<Mat2x2<$t>> for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn div(self, other: Mat2x2<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix /= scalar.
            impl DivAssign<$t> for Mat2x2<$t> {
                fn div_assign(&mut self, other: $t) {
                    self.x /= other;
                    self.y /= other;
                }
            }

            /// Matrix /= matrix.
            impl DivAssign<Mat2x2<$t>> for Mat2x2<$t> {
                fn div_assign(&mut self, other: Mat2x2<$t>) {
                    *self *= other.inv()
                }
            }

            /// -Matrix
            impl Neg for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn neg(self) -> Self::Output {
                    Mat2x2 {
                        x: -self.x,
                        y: -self.y,
                    }
                }
            }

            /// Convert complex number into rotation matrix (TODO: needs testing).
            impl From<Complex<$t>> for Mat2x2<$t> {
                fn from(value: Complex<$t>) -> Self {
                    let x2 = value.i + value.i;
                    let a = 1.0 - value.i * x2;
                    let b = value.r * x2;
                    Mat2x2 {
                        x: Vec2 { x: a, y: b },
                        y: Vec2 { x: -b, y: a },
                    }
                }
            }
        )+
    }
}

mat2x2_impl! { f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! mat2x2_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Mat2x2<$t>> for Mat2x2<$u> {
                fn from(value: Mat2x2<$t>) -> Self { Mat2x2 { x: value.x.into(),y: value.y.into(), } }
            }
        )+
    }
}

mat2x2_from_impl! { (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,isize) }
mat2x2_from_impl! { (i16,i32) (i16,i64) (i16,i128) (i16,isize) }
mat2x2_from_impl! { (i32,i64) (i32,i128) }
mat2x2_from_impl! { (i64,i128) }
mat2x2_from_impl! { (f32,f64) (f64,f32) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        }
        .transpose();
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 2.0, y: 4.0 },
                y: Vec2 { x: 3.0, y: 5.0 },
            }
        );
    }

    #[test]
    fn det() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        }
        .det();
        assert_eq!(result, -2.0);
    }

    #[test]
    fn add() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        } + Mat2x2::<f32> {
            x: Vec2 { x: 6.0, y: 7.0 },
            y: Vec2 { x: 8.0, y: 9.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 8.0, y: 10.0 },
                y: Vec2 { x: 12.0, y: 14.0 },
            }
        );
    }

    #[test]
    fn add_assign() {
        let mut result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        result += Mat2x2::<f32> {
            x: Vec2 { x: 6.0, y: 7.0 },
            y: Vec2 { x: 8.0, y: 9.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 8.0, y: 10.0 },
                y: Vec2 { x: 12.0, y: 14.0 },
            }
        );
    }

    #[test]
    fn sub() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 8.0, y: 7.0 },
            y: Vec2 { x: 6.0, y: 5.0 },
        } - Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 6.0, y: 4.0 },
                y: Vec2 { x: 2.0, y: 0.0 },
            }
        );
    }

    #[test]
    fn sub_assign() {
        let mut result = Mat2x2::<f32> {
            x: Vec2 { x: 8.0, y: 7.0 },
            y: Vec2 { x: 6.0, y: 5.0 },
        };
        result -= Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 6.0, y: 4.0 },
                y: Vec2 { x: 2.0, y: 0.0 },
            }
        );
    }

    #[test]
    fn mul_sm() {
        let result = 2.0
            * Mat2x2::<f32> {
                x: Vec2 { x: 2.0, y: 3.0 },
                y: Vec2 { x: 4.0, y: 5.0 },
            };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 4.0, y: 6.0 },
                y: Vec2 { x: 8.0, y: 10.0 },
            }
        );
    }

    #[test]
    fn mul_ms() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        } * 2.0;
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 4.0, y: 6.0 },
                y: Vec2 { x: 8.0, y: 10.0 },
            }
        );
    }

    #[test]
    fn mul_mv() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 4.0, y: 5.0 },
            y: Vec2 { x: 6.0, y: 7.0 },
        } * Vec2::<f32> { x: 2.0, y: 3.0 };
        assert_eq!(result, Vec2::<f32> { x: 26.0, y: 31.0 });
    }

    #[test]
    fn mul_mm() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        } * Mat2x2::<f32> {
            x: Vec2 { x: 6.0, y: 7.0 },
            y: Vec2 { x: 8.0, y: 9.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 40.0, y: 53.0 },
                y: Vec2 { x: 52.0, y: 69.0 },
            }
        );
    }

    #[test]
    fn mul_assign_ms() {
        let mut result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        result *= 2.0;
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 4.0, y: 6.0 },
                y: Vec2 { x: 8.0, y: 10.0 },
            }
        );
    }

    #[test]
    fn mul_assign_mm() {
        let mut result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        result *= Mat2x2::<f32> {
            x: Vec2 { x: 6.0, y: 7.0 },
            y: Vec2 { x: 8.0, y: 9.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 40.0, y: 53.0 },
                y: Vec2 { x: 52.0, y: 69.0 },
            }
        );
    }

    #[test]
    fn div_ms() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 4.0 },
            y: Vec2 { x: 6.0, y: 8.0 },
        } / 2.0;
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 1.0, y: 2.0 },
                y: Vec2 { x: 3.0, y: 4.0 },
            }
        );
    }

    #[test]
    fn div_assign_ms() {
        let mut result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 4.0 },
            y: Vec2 { x: 6.0, y: 8.0 },
        };
        result /= 2.0;
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: 1.0, y: 2.0 },
                y: Vec2 { x: 3.0, y: 4.0 },
            }
        );
    }

    #[test]
    fn neg() {
        let result = -Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        };
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: -2.0, y: -3.0 },
                y: Vec2 { x: -4.0, y: -5.0 },
            }
        );
    }

    #[test]
    fn inv() {
        let result = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        }
        .inv();
        assert_eq!(
            result,
            Mat2x2::<f32> {
                x: Vec2 { x: -2.5, y: 1.5 },
                y: Vec2 { x: 2.0, y: -1.0 },
            }
        );
    }

    #[test]
    fn test_codec_mat2x2_roundtrip() {
        let val = Mat2x2 { x: vec2(1.0f32, 2.0), y: vec2(3.0, 4.0) };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Mat2x2::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
