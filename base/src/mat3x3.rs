use {
    crate::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 3x3 column-major matrix, generic over the component type.
///
/// Columns are stored as [`Vec3`] fields `x`, `y`, and `z`. Supports matrix
/// arithmetic, scalar scaling, matrix-vector and matrix-matrix multiplication,
/// transpose, and for `f32`/`f64`: [`det`](Mat3x3::det) and [`inv`](Mat3x3::inv).
///
/// Can be constructed from [`Mat2x2`] (embed), [`Quat`] (rotation), or
/// [`Euler`] angles.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let q = Quat::<f32>::from_axis_angle(vec3(0.0, 0.0, 1.0), std::f32::consts::FRAC_PI_2);
/// let m: Mat3x3<f32> = q.into();
/// let v = m * vec3(1.0, 0.0, 0.0);
/// assert!((v.y - 1.0).abs() < 1e-6);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Mat3x3<T> {
    /// First column.
    pub x: Vec3<T>,
    /// Second column.
    pub y: Vec3<T>,
    /// Third column.
    pub z: Vec3<T>,
}

/// Create a new 3x3 matrix from column vectors.
pub const fn mat3x3<T>(x: Vec3<T>, y: Vec3<T>, z: Vec3<T>) -> Mat3x3<T> {
    Mat3x3 { x, y, z }
}

macro_rules! mat3x3_impl {
    ($($t:ty)+) => {
        $(
            impl Mat3x3<$t> {
                /// Create translation matrix from vector.
                pub fn from_vec2(value: Vec2<$t>) -> Mat3x3<$t> {
                    Mat3x3 {
                        x: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
                        y: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
                        z: Vec3 {
                            x: value.x,
                            y: value.y,
                            z: 1.0,
                        },
                    }
                }

                /// Compose matrix from 2x2 matrix and vector.
                pub fn from_mv(m: Mat2x2<$t>, v: Vec2<$t>) -> Mat3x3<$t> {
                    Mat3x3 {
                        x: Vec3 {
                            x: m.x.x,
                            y: m.x.y,
                            z: 0.0,
                        },
                        y: Vec3 {
                            x: m.y.x,
                            y: m.y.y,
                            z: 0.0,
                        },
                        z: Vec3 {
                            x: v.x,
                            y: v.y,
                            z: 1.0,
                        },
                    }
                }

                /// Transpose the matrix.
                pub fn transpose(self) -> Mat3x3<$t> {
                    Mat3x3 {
                        x: Vec3 {
                            x: self.x.x,
                            y: self.y.x,
                            z: self.z.x,
                        },
                        y: Vec3 {
                            x: self.x.y,
                            y: self.y.y,
                            z: self.z.y,
                        },
                        z: Vec3 {
                            x: self.x.z,
                            y: self.y.z,
                            z: self.z.z,
                        },
                    }
                }

                /// Calculate determinant of matrix.
                pub fn det(self) -> $t {
                    let a = self.x.x;
                    let d = self.x.y;
                    let g = self.x.z;
                    let b = self.y.x;
                    let e = self.y.y;
                    let h = self.y.z;
                    let c = self.z.x;
                    let f = self.z.y;
                    let i = self.z.z;
                    let eifh = e * i - f * h;
                    let fgdi = f * g - d * i;
                    let dheg = d * h - e * g;
                    a * eifh + b * fgdi + c * dheg
                }

                /// Calculate inverse of matrix.
                pub fn inv(self) -> Self {
                    let a = self.x.x;
                    let b = self.x.y;
                    let c = self.x.z;
                    let d = self.y.x;
                    let e = self.y.y;
                    let f = self.y.z;
                    let g = self.z.x;
                    let h = self.z.y;
                    let i = self.z.z;
                    let eifh = e * i - h * f;
                    let hcbi = h * c - b * i;
                    let bfec = b * f - e * c;
                    let det = a * eifh + d * hcbi + g * bfec;
                    if det == 0.0 {
                        return self;
                    }
                    let gfdi = g * f - d * i;
                    let aigc = a * i - g * c;
                    let dcaf = d * c - a * f;
                    let dhge = d * h - g * e;
                    let gbah = g * b - a * h;
                    let aedb = a * e - d * b;
                    Mat3x3 {
                        x: Vec3 { x: eifh, y: hcbi, z: bfec },
                        y: Vec3 { x: gfdi, y: aigc, z: dcaf },
                        z: Vec3 { x: dhge, y: gbah, z: aedb },
                    } / det
                }
            }

            impl Zero for Mat3x3<$t> {
                const ZERO: Mat3x3<$t> = Mat3x3 {
                    x: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
                    y: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
                    z: Vec3 { x: 0.0, y: 0.0, z: 0.0 },
                };
            }

            impl One for Mat3x3<$t> {
                const ONE: Mat3x3<$t> = Mat3x3 {
                    x: Vec3 { x: 1.0, y: 0.0, z: 0.0 },
                    y: Vec3 { x: 0.0, y: 1.0, z: 0.0 },
                    z: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
                };
            }

            impl Display for Mat3x3<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "[{},{},{}]", self.x, self.y, self.z)
                }
            }

            /// Matrix + matrix.
            impl Add<Mat3x3<$t>> for Mat3x3<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self::Output {
                    Mat3x3 {
                        x: self.x + other.x,
                        y: self.y + other.y,
                        z: self.z + other.z,
                    }
                }
            }

            /// Matrix += matrix.
            impl AddAssign<Mat3x3<$t>> for Mat3x3<$t> {
                fn add_assign(&mut self, other: Self) {
                    self.x += other.x;
                    self.y += other.y;
                    self.z += other.z;
                }
            }

            /// Matrix - matrix.
            impl Sub<Mat3x3<$t>> for Mat3x3<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self::Output {
                    Mat3x3 {
                        x: self.x - other.x,
                        y: self.y - other.y,
                        z: self.z - other.z,
                    }
                }
            }

            /// Matrix -= matrix.
            impl SubAssign<Mat3x3<$t>> for Mat3x3<$t> {
                fn sub_assign(&mut self, other: Self) {
                    self.x -= other.x;
                    self.y -= other.y;
                    self.z -= other.z;
                }
            }

            /// Scalar * matrix.
            impl Mul<Mat3x3<$t>> for $t {
                type Output = Mat3x3<$t>;
                fn mul(self, other: Mat3x3<$t>) -> Self::Output {
                    Mat3x3 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                    }
                }
            }

            /// Matrix * scalar.
            impl Mul<$t> for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn mul(self, other: $t) -> Self::Output {
                    Mat3x3 {
                        x: self.x * other,
                        y: self.y * other,
                        z: self.z * other,
                    }
                }
            }

            /// Matrix * vector.
            impl Mul<Vec3<$t>> for Mat3x3<$t> {
                type Output = Vec3<$t>;
                fn mul(self, other: Vec3<$t>) -> Self::Output {
                    Vec3 {
                        x: self.x.x * other.x + self.y.x * other.y + self.z.x * other.z,
                        y: self.x.y * other.x + self.y.y * other.y + self.z.y * other.z,
                        z: self.x.z * other.x + self.y.z * other.y + self.z.z * other.z,
                    }
                }
            }

            // Vector * matrix is not defined.

            /// Matrix * matrix.
            impl Mul<Mat3x3<$t>> for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn mul(self, other: Mat3x3<$t>) -> Self::Output {
                    Mat3x3 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                    }
                }
            }

            /// Matrix *= scalar.
            impl MulAssign<$t> for Mat3x3<$t> {
                fn mul_assign(&mut self, other: $t) {
                    self.x *= other;
                    self.y *= other;
                    self.z *= other;
                }
            }

            /// Matrix *= matrix.
            impl MulAssign<Mat3x3<$t>> for Mat3x3<$t> {
                fn mul_assign(&mut self, other: Mat3x3<$t>) {
                    let m = *self * other;
                    *self = m;
                }
            }

            /// Matrix / scalar.
            impl Div<$t> for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn div(self, other: $t) -> Self::Output {
                    Mat3x3 {
                        x: self.x / other,
                        y: self.y / other,
                        z: self.z / other,
                    }
                }
            }

            /// Scalar / matrix.
            impl Div<Mat3x3<$t>> for $t {
                type Output = Mat3x3<$t>;
                fn div(self, other: Mat3x3<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix / matrix.
            impl Div<Mat3x3<$t>> for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn div(self, other: Mat3x3<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix /= scalar.
            impl DivAssign<$t> for Mat3x3<$t> {
                fn div_assign(&mut self, other: $t) {
                    self.x /= other;
                    self.y /= other;
                    self.z /= other;
                }
            }

            /// Matrix /= matrix.
            impl DivAssign<Mat3x3<$t>> for Mat3x3<$t> {
                fn div_assign(&mut self, other: Mat3x3<$t>) {
                    *self *= other.inv()
                }
            }

            /// -Matrix.
            impl Neg for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn neg(self) -> Self {
                    Mat3x3 {
                        x: -self.x,
                        y: -self.y,
                        z: -self.z,
                    }
                }
            }

            /// Extend 2x2 matrix into 3x3 matrix.
            impl From<Mat2x2<$t>> for Mat3x3<$t> {
                fn from(value: Mat2x2<$t>) -> Mat3x3<$t> {
                    Mat3x3 {
                        x: Vec3 {
                            x: value.x.x,
                            y: value.x.y,
                            z: 0.0,
                        },
                        y: Vec3 {
                            x: value.y.x,
                            y: value.y.y,
                            z: 0.0,
                        },
                        z: Vec3 { x: 0.0, y: 0.0, z: 1.0 },
                    }
                }
            }

            /// Create rotation matrix from quaternion.
            impl From<Quat<$t>> for Mat3x3<$t> {
                fn from(value: Quat<$t>) -> Mat3x3<$t> {
                    let x2 = value.i + value.i;
                    let y2 = value.j + value.j;
                    let z2 = value.k + value.k;
                    let xx2 = value.i * x2;
                    let yy2 = value.j * y2;
                    let zz2 = value.k * z2;
                    let yz2 = value.j * z2;
                    let wx2 = value.r * x2;
                    let xy2 = value.i * y2;
                    let wz2 = value.r * z2;
                    let xz2 = value.i * z2;
                    let wy2 = value.r * y2;
                    Mat3x3 {
                        x: Vec3 {
                            x: 1.0 - yy2 - zz2,
                            y: xy2 + wz2,
                            z: xz2 - wy2,
                        },
                        y: Vec3 {
                            x: xy2 - wz2,
                            y: 1.0 - xx2 - zz2,
                            z: yz2 + wx2,
                        },
                        z: Vec3 {
                            x: xz2 + wy2,
                            y: yz2 - wx2,
                            z: 1.0 - xx2 - yy2,
                        },
                    }
                }
            }
        )+
    }
}

mat3x3_impl! { f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! mat3x3_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Mat3x3<$t>> for Mat3x3<$u> {
                fn from(value: Mat3x3<$t>) -> Self { Mat3x3 { x: value.x.into(),y: value.y.into(),z: value.z.into(), } }
            }
        )+
    }
}

mat3x3_from_impl! { (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,isize) }
mat3x3_from_impl! { (i16,i32) (i16,i64) (i16,i128) (i16,isize) }
mat3x3_from_impl! { (i32,i64) (i32,i128) }
mat3x3_from_impl! { (i64,i128) }
mat3x3_from_impl! { (f32,f64) (f64,f32) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec2() {
        let result = Mat3x3::<f32>::from_vec2(Vec2::<f32> { x: 2.0, y: 3.0 });
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                y: Vec3 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                },
                z: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 1.0,
                },
            }
        );
    }

    #[test]
    fn from_mat2x2() {
        let result: Mat3x3<f32> = Mat2x2::<f32> {
            x: Vec2 { x: 2.0, y: 3.0 },
            y: Vec2 { x: 4.0, y: 5.0 },
        }
        .into();
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 0.0,
                },
                y: Vec3 {
                    x: 4.0,
                    y: 5.0,
                    z: 0.0,
                },
                z: Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            }
        );
    }

    #[test]
    fn from_mv() {
        let result = Mat3x3::<f32>::from_mv(
            Mat2x2::<f32> {
                x: Vec2 { x: 2.0, y: 3.0 },
                y: Vec2 { x: 4.0, y: 5.0 },
            },
            Vec2::<f32> { x: 6.0, y: 7.0 },
        );
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 0.0,
                },
                y: Vec3 {
                    x: 4.0,
                    y: 5.0,
                    z: 0.0,
                },
                z: Vec3 {
                    x: 6.0,
                    y: 7.0,
                    z: 1.0,
                },
            }
        );
    }

    #[test]
    fn from_quat() {
        let result: Mat3x3<f32> = Quat::<f32>::from_axis_angle(
            Vec3 {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            },
            0.785398163,
        )
        .into();
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                y: Vec3 {
                    x: 0.0,
                    y: 0.7071067,
                    z: 0.7071068,
                },
                z: Vec3 {
                    x: 0.0,
                    y: -0.7071068,
                    z: 0.7071067,
                },
            }
        );
    }

    #[test]
    fn transpose() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        }
        .transpose();
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 5.0,
                    z: 8.0,
                },
                y: Vec3 {
                    x: 3.0,
                    y: 6.0,
                    z: 9.0,
                },
                z: Vec3 {
                    x: 4.0,
                    y: 7.0,
                    z: 10.0,
                },
            }
        );
    }

    #[test]
    fn det() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        }
        .det();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn add() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        } + Mat3x3::<f32> {
            x: Vec3 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
            },
            y: Vec3 {
                x: 0.0,
                y: -1.0,
                z: -2.0,
            },
            z: Vec3 {
                x: -3.0,
                y: -4.0,
                z: -5.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
                z: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
            }
        );
    }

    #[test]
    fn add_assign() {
        let mut result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        };
        result += Mat3x3::<f32> {
            x: Vec3 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
            },
            y: Vec3 {
                x: 0.0,
                y: -1.0,
                z: -2.0,
            },
            z: Vec3 {
                x: -3.0,
                y: -4.0,
                z: -5.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
                z: Vec3 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                },
            }
        );
    }

    #[test]
    fn sub() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        } - Mat3x3::<f32> {
            x: Vec3 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
            },
            y: Vec3 {
                x: 0.0,
                y: -1.0,
                z: -2.0,
            },
            z: Vec3 {
                x: -3.0,
                y: -4.0,
                z: -5.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: -1.0,
                    y: 1.0,
                    z: 3.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 7.0,
                    z: 9.0,
                },
                z: Vec3 {
                    x: 11.0,
                    y: 13.0,
                    z: 15.0,
                },
            }
        );
    }

    #[test]
    fn sub_assign() {
        let mut result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        };
        result -= Mat3x3::<f32> {
            x: Vec3 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
            },
            y: Vec3 {
                x: 0.0,
                y: -1.0,
                z: -2.0,
            },
            z: Vec3 {
                x: -3.0,
                y: -4.0,
                z: -5.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: -1.0,
                    y: 1.0,
                    z: 3.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 7.0,
                    z: 9.0,
                },
                z: Vec3 {
                    x: 11.0,
                    y: 13.0,
                    z: 15.0,
                },
            }
        );
    }

    #[test]
    fn mul_sm() {
        let result = 2.0
            * Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 6.0,
                    z: 7.0,
                },
                z: Vec3 {
                    x: 8.0,
                    y: 9.0,
                    z: 10.0,
                },
            };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                },
                y: Vec3 {
                    x: 10.0,
                    y: 12.0,
                    z: 14.0,
                },
                z: Vec3 {
                    x: 16.0,
                    y: 18.0,
                    z: 20.0,
                },
            }
        );
    }

    #[test]
    fn mul_ms() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        } * 2.0;
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                },
                y: Vec3 {
                    x: 10.0,
                    y: 12.0,
                    z: 14.0,
                },
                z: Vec3 {
                    x: 16.0,
                    y: 18.0,
                    z: 20.0,
                },
            }
        );
    }

    #[test]
    fn mul_mv() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        } * Vec3::<f32> {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        };
        assert_eq!(
            result,
            Vec3::<f32> {
                x: 51.0,
                y: 60.0,
                z: 69.0,
            }
        );
    }

    #[test]
    fn mul_mm() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        } * Mat3x3::<f32> {
            x: Vec3 {
                x: 11.0,
                y: 12.0,
                z: 13.0,
            },
            y: Vec3 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
            },
            z: Vec3 {
                x: 17.0,
                y: 18.0,
                z: 19.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 186.0,
                    y: 222.0,
                    z: 258.0,
                },
                y: Vec3 {
                    x: 231.0,
                    y: 276.0,
                    z: 321.0,
                },
                z: Vec3 {
                    x: 276.0,
                    y: 330.0,
                    z: 384.0,
                },
            }
        );
    }

    #[test]
    fn mul_assign_ms() {
        let mut result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        };
        result *= 2.0;
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                },
                y: Vec3 {
                    x: 10.0,
                    y: 12.0,
                    z: 14.0,
                },
                z: Vec3 {
                    x: 16.0,
                    y: 18.0,
                    z: 20.0,
                },
            }
        );
    }

    #[test]
    fn mul_assign_mm() {
        let mut result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        };
        result *= Mat3x3::<f32> {
            x: Vec3 {
                x: 11.0,
                y: 12.0,
                z: 13.0,
            },
            y: Vec3 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
            },
            z: Vec3 {
                x: 17.0,
                y: 18.0,
                z: 19.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 186.0,
                    y: 222.0,
                    z: 258.0,
                },
                y: Vec3 {
                    x: 231.0,
                    y: 276.0,
                    z: 321.0,
                },
                z: Vec3 {
                    x: 276.0,
                    y: 330.0,
                    z: 384.0,
                },
            }
        );
    }

    #[test]
    fn div_ms() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 4.0,
                y: 6.0,
                z: 8.0,
            },
            y: Vec3 {
                x: 10.0,
                y: 12.0,
                z: 14.0,
            },
            z: Vec3 {
                x: 16.0,
                y: 18.0,
                z: 20.0,
            },
        } / 2.0;
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 6.0,
                    z: 7.0,
                },
                z: Vec3 {
                    x: 8.0,
                    y: 9.0,
                    z: 10.0,
                },
            }
        );
    }

    #[test]
    fn div_assign_ms() {
        let mut result = Mat3x3::<f32> {
            x: Vec3 {
                x: 4.0,
                y: 6.0,
                z: 8.0,
            },
            y: Vec3 {
                x: 10.0,
                y: 12.0,
                z: 14.0,
            },
            z: Vec3 {
                x: 16.0,
                y: 18.0,
                z: 20.0,
            },
        };
        result /= 2.0;
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                },
                y: Vec3 {
                    x: 5.0,
                    y: 6.0,
                    z: 7.0,
                },
                z: Vec3 {
                    x: 8.0,
                    y: 9.0,
                    z: 10.0,
                },
            }
        );
    }

    #[test]
    fn neg() {
        let result = -Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            y: Vec3 {
                x: 5.0,
                y: 6.0,
                z: 7.0,
            },
            z: Vec3 {
                x: 8.0,
                y: 9.0,
                z: 10.0,
            },
        };
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: -2.0,
                    y: -3.0,
                    z: -4.0,
                },
                y: Vec3 {
                    x: -5.0,
                    y: -6.0,
                    z: -7.0,
                },
                z: Vec3 {
                    x: -8.0,
                    y: -9.0,
                    z: -10.0,
                },
            }
        );
    }

    #[test]
    fn inv() {
        let result = Mat3x3::<f32> {
            x: Vec3 {
                x: 2.0,
                y: 5.0,
                z: 2.0,
            },
            y: Vec3 {
                x: 3.0,
                y: 8.0,
                z: 6.0,
            },
            z: Vec3 {
                x: 2.0,
                y: 1.0,
                z: 4.0,
            },
        }
        .inv();
        assert_eq!(
            result,
            Mat3x3::<f32> {
                x: Vec3 {
                    x: 1.0,
                    y: -9.0 / 13.0,
                    z: 7.0 / 13.0,
                },
                y: Vec3 {
                    x: 0.0,
                    y: 2.0 / 13.0,
                    z: -3.0 / 13.0,
                },
                z: Vec3 {
                    x: -0.5,
                    y: 4.0 / 13.0,
                    z: 1.0 / 26.0,
                },
            }
        );
    }

    #[test]
    fn test_codec_mat3x3_roundtrip() {
        let val = Mat3x3 {
            x: vec3(1.0f32, 2.0, 3.0),
            y: vec3(4.0, 5.0, 6.0),
            z: vec3(7.0, 8.0, 9.0),
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Mat3x3::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
