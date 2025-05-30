use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 2x2 matrix of numbers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat2x2<T> {
    pub x: Vec2<T>,
    pub y: Vec2<T>,
}

impl<T> Mat2x2<T>
where
    T: Copy
        + Zero
        + PartialEq
        + Div<Output = T>
        + Mul<Output = T>
        + Sub<Output = T>
        + Neg<Output = T>,
{
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
}

impl<T> Zero for Mat2x2<T>
where
    Vec2<T>: Zero,
{
    const ZERO: Mat2x2<T> = Mat2x2 {
        x: Vec2::ZERO,
        y: Vec2::ZERO,
    };
}

impl<T> One for Mat2x2<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Add<Output = T>,
{
    const ONE: Mat2x2<T> = Mat2x2 {
        x: Vec2::UNIT_X,
        y: Vec2::UNIT_Y,
    };
}

impl<T> Display for Mat2x2<T>
where
    Vec2<T>: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "[{},{}]", self.x, self.y)
    }
}

/// Matrix + matrix.
impl<T> Add<Mat2x2<T>> for Mat2x2<T>
where
    Vec2<T>: Add<Vec2<T>, Output = Vec2<T>>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Mat2x2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// Matrix += matrix.
impl<T> AddAssign<Mat2x2<T>> for Mat2x2<T>
where
    Vec2<T>: AddAssign<Vec2<T>>,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// Matrix - matrix.
impl<T> Sub<Mat2x2<T>> for Mat2x2<T>
where
    Vec2<T>: Sub<Vec2<T>, Output = Vec2<T>>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Mat2x2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

/// Matrix -= matrix.
impl<T> SubAssign<Mat2x2<T>> for Mat2x2<T>
where
    Vec2<T>: SubAssign<Vec2<T>>,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

/// Matrix * scalar.
impl<T> Mul<T> for Mat2x2<T>
where
    Vec2<T>: Mul<T, Output = Vec2<T>>,
    T: Copy,
{
    type Output = Mat2x2<T>;
    fn mul(self, other: T) -> Self::Output {
        Mat2x2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

/// Matrix * vector.
impl<T> Mul<Vec2<T>> for Mat2x2<T>
where
    T: Copy + Mul<Output = T> + Add<Output = T>,
{
    type Output = Vec2<T>;
    fn mul(self, other: Vec2<T>) -> Self::Output {
        Vec2 {
            x: self.x.x * other.x + self.y.x * other.y,
            y: self.x.y * other.x + self.y.y * other.y,
        }
    }
}

/// Matrix * matrix.
impl<T> Mul<Mat2x2<T>> for Mat2x2<T>
where
    Mat2x2<T>: Copy + Mul<Vec2<T>, Output = Vec2<T>>,
{
    type Output = Mat2x2<T>;
    fn mul(self, other: Mat2x2<T>) -> Self::Output {
        Mat2x2 {
            x: self * other.x,
            y: self * other.y,
        }
    }
}

/// Matrix *= scalar.
impl<T> MulAssign<T> for Mat2x2<T>
where
    Vec2<T>: MulAssign<T>,
    T: Copy,
{
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
    }
}

/// Matrix *= matrix.
impl<T> MulAssign<Mat2x2<T>> for Mat2x2<T>
where
    Mat2x2<T>: Copy + Mul<Mat2x2<T>, Output = Mat2x2<T>>,
{
    fn mul_assign(&mut self, other: Mat2x2<T>) {
        let m = *self * other;
        *self = m;
    }
}

/// Matrix / scalar.
impl<T> Div<T> for Mat2x2<T>
where
    Vec2<T>: Div<T, Output = Vec2<T>>,
    T: Copy,
{
    type Output = Mat2x2<T>;
    fn div(self, other: T) -> Self::Output {
        Mat2x2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

/// Matrix /= scalar.
impl<T> DivAssign<T> for Mat2x2<T>
where
    Vec2<T>: DivAssign<T>,
    T: Copy,
{
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
    }
}

/// -Matrix
impl<T> Neg for Mat2x2<T>
where
    Vec2<T>: Neg<Output = Vec2<T>>,
{
    type Output = Mat2x2<T>;
    fn neg(self) -> Self::Output {
        Mat2x2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

macro_rules! mat2x2_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * matrix.
            impl Mul<Mat2x2<$t>> for $t {
                type Output = Mat2x2<$t>;
                fn mul(self,other: Mat2x2<$t>) -> Self::Output {
                    Mat2x2 {
                        x: self * other.x,
                        y: self * other.y,
                    }
                }
            }
        )+
    }
}

mat2x2_impl! { isize i8 i16 i32 i64 i128 f32 f64 }

macro_rules! mat2x2_real_impl {
    ($($t:ty)+) => {
        $(
            impl Mat2x2<$t> {

                /// Calculate determinant of matrix.
                pub fn det(self) -> $t {
                    let a = self.x.x;
                    let b = self.y.x;
                    let c = self.x.y;
                    let d = self.y.y;
                    let aa = d;
                    let ab = c;
                    a * aa - b * ab
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
                        x: Vec2 { x: d,y: -b, },
                        y: Vec2 { x: -c,y: a, },
                    } / det
                }
            }

            /// Scalar / matrix.
            impl Div<Mat2x2<$t>> for $t {
                type Output = Mat2x2<$t>;
                fn div(self,other: Mat2x2<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix / matrix.
            impl Div<Mat2x2<$t>> for Mat2x2<$t> {
                type Output = Mat2x2<$t>;
                fn div(self,other: Mat2x2<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix /= matrix.
            impl DivAssign<Mat2x2<$t>> for Mat2x2<$t> {
                fn div_assign(&mut self,other: Mat2x2<$t>) {
                    *self *= other.inv()
                }
            }
        )+
    }
}

mat2x2_real_impl! { f32 f64 }

impl<T> From<Complex<T>> for Mat2x2<T>
where
    T: Copy + One + Mul<Output = T> + Neg<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    /// Convert complex number into rotation matrix (TODO: needs testing).
    fn from(value: Complex<T>) -> Self {
        let x2 = value.i + value.i;
        let a = T::ONE - value.i * x2;
        let b = value.r * x2;
        Mat2x2 {
            x: Vec2 { x: a, y: b },
            y: Vec2 { x: -b, y: a },
        }
    }
}

// if `T as U` exists, `Mat2x2<U>::from(Mat2x2<T>)` should also exist
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

mat2x2_from_impl! { (isize,i8) (isize,i16) (isize,i32) (isize,i64) (isize,i128) (isize,f32) (isize,f64) }
mat2x2_from_impl! { (i8,isize) (i8,u16) (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,f32) (i8,f64) }
mat2x2_from_impl! { (i16,isize) (i16,i8) (i16,i32) (i16,i64) (i16,i128) (i16,f32) (i16,f64) }
mat2x2_from_impl! { (i32,isize) (i32,i8) (i32,i16) (i32,i64) (i32,i128) (i32,f32) (i32,f64) }
mat2x2_from_impl! { (i64,isize) (i64,i8) (i64,i16) (i64,i32) (i64,i128) (i64,f32) (i64,f64) }
mat2x2_from_impl! { (i128,isize) (i128,i8) (i128,i16) (i128,i32) (i128,i64) (i128,f32) (i128,f64) }
mat2x2_from_impl! { (f32,isize) (f32,i8) (f32,i16) (f32,i32) (f32,i64) (f32,i128) (f32,f64) }
mat2x2_from_impl! { (f64,isize) (f64,i8) (f64,i16) (f64,i32) (f64,i64)(f64,i128) (f64,f32) }

mod tests {
    #[allow(unused_imports)]
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
}
