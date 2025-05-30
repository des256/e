use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 3x3 matrix of numbers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat3x3<T> {
    pub x: Vec3<T>,
    pub y: Vec3<T>,
    pub z: Vec3<T>,
}

impl<T> Mat3x3<T>
where
    T: Copy
        + Zero
        + One
        + PartialEq
        + Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>
        + Add<Output = T>,
{
    /// Create translation matrix from vector.
    pub fn from_vec2(value: Vec2<T>) -> Mat3x3<T> {
        Mat3x3 {
            x: Vec3::<T>::UNIT_X,
            y: Vec3::<T>::UNIT_Y,
            z: Vec3 {
                x: value.x,
                y: value.y,
                z: T::ONE,
            },
        }
    }

    /// Compose matrix from 2x2 matrix and vector.
    pub fn from_mv(m: Mat2x2<T>, v: Vec2<T>) -> Mat3x3<T> {
        Mat3x3 {
            x: Vec3 {
                x: m.x.x,
                y: m.x.y,
                z: T::ZERO,
            },
            y: Vec3 {
                x: m.y.x,
                y: m.y.y,
                z: T::ZERO,
            },
            z: Vec3 {
                x: v.x,
                y: v.y,
                z: T::ONE,
            },
        }
    }

    /// Transpose the matrix.
    pub fn transpose(self) -> Mat3x3<T> {
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
}

impl<T> Zero for Mat3x3<T>
where
    Vec3<T>: Zero,
{
    const ZERO: Mat3x3<T> = Mat3x3 {
        x: Vec3::ZERO,
        y: Vec3::ZERO,
        z: Vec3::ZERO,
    };
}

impl<T> One for Mat3x3<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    const ONE: Mat3x3<T> = Mat3x3 {
        x: Vec3::UNIT_X,
        y: Vec3::UNIT_Y,
        z: Vec3::UNIT_Z,
    };
}

impl<T> Display for Mat3x3<T>
where
    Vec3<T>: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "[{},{},{}]", self.x, self.y, self.z)
    }
}

/// Matrix + matrix.
impl<T> Add<Mat3x3<T>> for Mat3x3<T>
where
    Vec3<T>: Add<Vec3<T>, Output = Vec3<T>>,
{
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
impl<T> AddAssign<Mat3x3<T>> for Mat3x3<T>
where
    Vec3<T>: AddAssign<Vec3<T>>,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

/// Matrix - matrix.
impl<T> Sub<Mat3x3<T>> for Mat3x3<T>
where
    Vec3<T>: Sub<Vec3<T>, Output = Vec3<T>>,
{
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
impl<T> SubAssign<Mat3x3<T>> for Mat3x3<T>
where
    Vec3<T>: SubAssign<Vec3<T>>,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

/// Matrix * scalar.
impl<T> Mul<T> for Mat3x3<T>
where
    T: Copy,
    Vec3<T>: Mul<T, Output = Vec3<T>>,
{
    type Output = Mat3x3<T>;
    fn mul(self, other: T) -> Self::Output {
        Mat3x3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

/// Matrix * vector.
impl<T> Mul<Vec3<T>> for Mat3x3<T>
where
    T: Copy + Mul<Output = T> + Add<Output = T>,
{
    type Output = Vec3<T>;
    fn mul(self, other: Vec3<T>) -> Self::Output {
        Vec3 {
            x: self.x.x * other.x + self.y.x * other.y + self.z.x * other.z,
            y: self.x.y * other.x + self.y.y * other.y + self.z.y * other.z,
            z: self.x.z * other.x + self.y.z * other.y + self.z.z * other.z,
        }
    }
}

// Vector * matrix is not defined.

/// Matrix * matrix.
impl<T> Mul<Mat3x3<T>> for Mat3x3<T>
where
    Mat3x3<T>: Copy + Mul<Vec3<T>, Output = Vec3<T>>,
{
    type Output = Mat3x3<T>;
    fn mul(self, other: Mat3x3<T>) -> Self::Output {
        Mat3x3 {
            x: self * other.x,
            y: self * other.y,
            z: self * other.z,
        }
    }
}

/// Matrix *= scalar.
impl<T> MulAssign<T> for Mat3x3<T>
where
    T: Copy,
    Vec3<T>: MulAssign<T>,
{
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

/// Matrix *= matrix.
impl<T> MulAssign<Mat3x3<T>> for Mat3x3<T>
where
    Mat3x3<T>: Copy + Mul<Mat3x3<T>, Output = Mat3x3<T>>,
{
    fn mul_assign(&mut self, other: Mat3x3<T>) {
        let m = *self * other;
        *self = m;
    }
}

/// Matrix / scalar.
impl<T> Div<T> for Mat3x3<T>
where
    T: Copy,
    Vec3<T>: Div<T, Output = Vec3<T>>,
{
    type Output = Mat3x3<T>;
    fn div(self, other: T) -> Self::Output {
        Mat3x3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

/// Matrix /= scalar.
impl<T> DivAssign<T> for Mat3x3<T>
where
    T: Copy,
    Vec3<T>: DivAssign<T>,
{
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

/// -Matrix.
impl<T> Neg for Mat3x3<T>
where
    Vec3<T>: Neg<Output = Vec3<T>>,
{
    type Output = Mat3x3<T>;
    fn neg(self) -> Self {
        Mat3x3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

macro_rules! mat3x3_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * matrix.
            impl Mul<Mat3x3<$t>> for $t {
                type Output = Mat3x3<$t>;
                fn mul(self,other: Mat3x3<$t>) -> Self::Output {
                    Mat3x3 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                    }
                }
            }
        )+
    }
}

mat3x3_impl! { isize i8 i16 i32 i64 i128 f32 f64 }

macro_rules! mat3x3_real_impl {
    ($($t:ty)+) => {
        $(
            impl Mat3x3<$t> {

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
                        x: Vec3 { x: eifh,y: hcbi,z: bfec, },
                        y: Vec3 { x: gfdi,y: aigc,z: dcaf, },
                        z: Vec3 { x: dhge,y: gbah,z: aedb, },
                    } / det
                }
            }

            /// Scalar / matrix.
            impl Div<Mat3x3<$t>> for $t {
                type Output = Mat3x3<$t>;
                fn div(self,other: Mat3x3<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix / matrix.
            impl Div<Mat3x3<$t>> for Mat3x3<$t> {
                type Output = Mat3x3<$t>;
                fn div(self,other: Mat3x3<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix /= matrix.
            impl DivAssign<Mat3x3<$t>> for Mat3x3<$t> {
                fn div_assign(&mut self,other: Mat3x3<$t>) {
                    *self *= other.inv()
                }
            }
        )+
    }
}

mat3x3_real_impl! { f32 f64 }

impl<T> From<Mat2x2<T>> for Mat3x3<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    /// Extend 2x2 matrix into 3x3 matrix.
    fn from(value: Mat2x2<T>) -> Mat3x3<T> {
        Mat3x3 {
            x: Vec3 {
                x: value.x.x,
                y: value.x.y,
                z: T::ZERO,
            },
            y: Vec3 {
                x: value.y.x,
                y: value.y.y,
                z: T::ZERO,
            },
            z: Vec3::<T>::UNIT_Z,
        }
    }
}

impl<T> From<Quat<T>> for Mat3x3<T>
where
    T: Copy + One + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    /// Create rotation matrix from quaternion.
    fn from(value: Quat<T>) -> Mat3x3<T> {
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
                x: T::ONE - yy2 - zz2,
                y: xy2 + wz2,
                z: xz2 - wy2,
            },
            y: Vec3 {
                x: xy2 - wz2,
                y: T::ONE - xx2 - zz2,
                z: yz2 + wx2,
            },
            z: Vec3 {
                x: xz2 + wy2,
                y: yz2 - wx2,
                z: T::ONE - xx2 - yy2,
            },
        }
    }
}

// if `T as U` exists, `Mat3x3<U>::from(Mat3x3<T>)` should also exist
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

mat3x3_from_impl! { (isize,i8) (isize,i16) (isize,i32) (isize,i64) (isize,i128) (isize,f32) (isize,f64) }
mat3x3_from_impl! { (i8,isize) (i8,u16) (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,f32) (i8,f64) }
mat3x3_from_impl! { (i16,isize) (i16,i8) (i16,i32) (i16,i64) (i16,i128) (i16,f32) (i16,f64) }
mat3x3_from_impl! { (i32,isize) (i32,i8) (i32,i16) (i32,i64) (i32,i128) (i32,f32) (i32,f64) }
mat3x3_from_impl! { (i64,isize) (i64,i8) (i64,i16) (i64,i32) (i64,i128) (i64,f32) (i64,f64) }
mat3x3_from_impl! { (i128,isize) (i128,i8) (i128,i16) (i128,i32) (i128,i64) (i128,f32) (i128,f64) }
mat3x3_from_impl! { (f32,isize) (f32,i8) (f32,i16) (f32,i32) (f32,i64) (f32,i128) (f32,f64) }
mat3x3_from_impl! { (f64,isize) (f64,i8) (f64,i16) (f64,i32) (f64,i64)(f64,i128) (f64,f32) }

mod tests {
    #[allow(unused_imports)]
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
}
