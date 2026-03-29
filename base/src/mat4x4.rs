use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 4x4 column-major matrix, generic over the component type.
///
/// Columns are stored as [`Vec4`] fields `x`, `y`, `z`, and `w`. Supports matrix
/// arithmetic, scalar scaling, matrix-vector and matrix-matrix multiplication,
/// transpose, and for `f32`/`f64`: [`det`](Mat4x4::det), [`inv`](Mat4x4::inv),
/// [`perspective`](Mat4x4::perspective), [`ortho`](Mat4x4::ortho),
/// [`look_at`](Mat4x4::look_at), [`frustum`](Mat4x4::frustum), and
/// decomposition methods [`translation`](Mat4x4::translation),
/// [`rotation`](Mat4x4::rotation), [`scale`](Mat4x4::scale).
///
/// Can be constructed from [`Quat`], [`Pose`], [`Vec3`] (translation),
/// or [`Mat3x3`] (embed).
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let view = Mat4x4::<f32>::look_at(
///     vec3(0.0, 0.0, 5.0),
///     vec3(0.0, 0.0, 0.0),
///     vec3(0.0, 1.0, 0.0),
/// );
/// let origin = view * vec4(0.0, 0.0, 0.0, 1.0);
/// assert!((origin.z + 5.0).abs() < 1e-6);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mat4x4<T> {
    /// First column.
    pub x: Vec4<T>,
    /// Second column.
    pub y: Vec4<T>,
    /// Third column.
    pub z: Vec4<T>,
    /// Fourth column.
    pub w: Vec4<T>,
}

/// Create a new 4x4 matrix from column vectors.
pub const fn mat4x4<T>(x: Vec4<T>, y: Vec4<T>, z: Vec4<T>, w: Vec4<T>) -> Mat4x4<T> {
    Mat4x4 { x, y, z, w }
}

macro_rules! mat4x4_impl {
    ($($t:ty)+) => {
        $(
            impl Mat4x4<$t> {

                /// Transpose the matrix.
                pub fn transpose(self) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 {
                            x: self.x.x,
                            y: self.y.x,
                            z: self.z.x,
                            w: self.w.x,
                        },
                        y: Vec4 {
                            x: self.x.y,
                            y: self.y.y,
                            z: self.z.y,
                            w: self.w.y,
                        },
                        z: Vec4 {
                            x: self.x.z,
                            y: self.y.z,
                            z: self.z.z,
                            w: self.w.z,
                        },
                        w: Vec4 {
                            x: self.x.w,
                            y: self.y.w,
                            z: self.z.w,
                            w: self.w.w,
                        },
                    }
                }

                /// Create perspective projection matrix.
                pub fn perspective(fovy: $t, aspect: $t, n: $t, f: $t) -> Mat4x4<$t> {
                    let ymax = n * (0.5 * fovy).tan();
                    let xmax = ymax * aspect;
                    Self::frustum(-xmax, xmax, -ymax, ymax, n, f)
                }

                /// Create frustum projection matrix based on FOV and aspect ratio.
                pub fn perspective_from_fov(fov: Fov<$t>, n: $t, f: $t) -> Mat4x4<$t> {
                    let l = n * fov.l.tan();
                    let r = n * fov.r.tan();
                    let b = n * fov.b.tan();
                    let t = n * fov.t.tan();
                    Self::frustum(l, r, b, t, n, f)
                }

                /// Create frustum projection matrix.
                pub fn frustum(l: $t, r: $t, b: $t, t: $t, n: $t, f: $t) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 {
                            x: 2.0 * n / (r - l),
                            y: 0.0,
                            z: 0.0,
                            w: 0.0,
                        },
                        y: Vec4 {
                            x: 0.0,
                            y: 2.0 * n / (t - b),
                            z: 0.0,
                            w: 0.0,
                        },
                        z: Vec4 {
                            x: (r + l) / (r - l),
                            y: (t + b) / (t - b),
                            z: -(f + n) / (f - n),
                            w: -1.0,
                        },
                        w: Vec4 {
                            x: 0.0,
                            y: 0.0,
                            z: -(f * n + f * n) / (f - n),
                            w: 0.0,
                        }
                    }
                }

                /// Calculate inverse of matrix.
                pub fn inv(self) -> Self {

                    let a = self.x.x;
                    let b = self.x.y;
                    let c = self.x.z;
                    let d = self.x.w;
                    let e = self.y.x;
                    let f = self.y.y;
                    let g = self.y.z;
                    let h = self.y.w;
                    let i = self.z.x;
                    let j = self.z.y;
                    let k = self.z.z;
                    let l = self.z.w;
                    let m = self.w.x;
                    let n = self.w.y;
                    let o = self.w.z;
                    let p = self.w.w;

                    let kpol = k * p - o * l;
                    let jpnl = j * p - n * l;
                    let jonk = j * o - n * k;
                    let ipml = i * p - m * l;
                    let iomk = i * o - m * k;
                    let inmj = i * n - m * j;

                    let ba = f * kpol - g * jpnl + h * jonk;
                    let bb = -e * kpol + g * ipml - h * iomk;
                    let bc = e * jpnl - f * ipml + h * inmj;
                    let bd = -e * jonk + f * iomk - g * inmj;

                    let det = a * ba + b * bb + c * bc + d * bd;
                    if det == 0.0 {
                        return self;
                    }

                    let be = -b * kpol + c * jpnl - d * jonk;
                    let bf = a * kpol - c * ipml + d * iomk;
                    let bg = -a * jpnl + b * ipml - d * inmj;
                    let bh = a * jonk - b * iomk + c * inmj;

                    let chgd = c * h - g * d;
                    let bhfd = b * h - f * d;
                    let bgfc = b * g - f * c;
                    let ahed = a * h - e * d;
                    let agec = a * g - e * c;
                    let afeb = a * f - e * b;

                    let bi = n * chgd - o * bhfd + p * bgfc;
                    let bj = -m * chgd + o * ahed - p * agec;
                    let bk = m * bhfd - n * ahed + p * afeb;
                    let bl = -m * bgfc + n * agec - o * afeb;

                    let bm = -j * chgd + k * bhfd - l * bgfc;
                    let bn = i * chgd - k * ahed + l * agec;
                    let bo = -i * bhfd + j * ahed - l * afeb;
                    let bp = i * bgfc - j * agec + k * afeb;

                    Mat4x4 {
                        x: Vec4 { x: ba, y: be, z: bi, w: bm, },
                        y: Vec4 { x: bb, y: bf, z: bj, w: bn, },
                        z: Vec4 { x: bc, y: bg, z: bk, w: bo, },
                        w: Vec4 { x: bd, y: bh, z: bl, w: bp, },
                    } / det
                }

                /// Create an orthographic projection matrix.
                ///
                /// Parameters define the view volume: `l`/`r` (left/right),
                /// `b`/`t` (bottom/top), `n`/`f` (near/far).
                pub fn ortho(l: $t, r: $t, b: $t, t: $t, n: $t, f: $t) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 { x: 2.0 / (r - l), y: 0.0, z: 0.0, w: 0.0 },
                        y: Vec4 { x: 0.0, y: 2.0 / (t - b), z: 0.0, w: 0.0 },
                        z: Vec4 { x: 0.0, y: 0.0, z: -2.0 / (f - n), w: 0.0 },
                        w: Vec4 { x: -(r + l) / (r - l), y: -(t + b) / (t - b), z: -(f + n) / (f - n), w: 1.0 },
                    }
                }

                /// Create a right-handed look-at view matrix.
                ///
                /// Builds a view matrix that positions the camera at `eye`,
                /// looking towards `target`, with the given `up` direction.
                pub fn look_at(eye: Vec3<$t>, target: Vec3<$t>, up: Vec3<$t>) -> Mat4x4<$t> {
                    let f = vec3(
                        target.x - eye.x,
                        target.y - eye.y,
                        target.z - eye.z,
                    ).normalized();
                    let s = f.cross(up).normalized();
                    let u = s.cross(f);
                    Mat4x4 {
                        x: Vec4 { x: s.x, y: u.x, z: -f.x, w: 0.0 },
                        y: Vec4 { x: s.y, y: u.y, z: -f.y, w: 0.0 },
                        z: Vec4 { x: s.z, y: u.z, z: -f.z, w: 0.0 },
                        w: Vec4 {
                            x: -(s.x * eye.x + s.y * eye.y + s.z * eye.z),
                            y: -(u.x * eye.x + u.y * eye.y + u.z * eye.z),
                            z: f.x * eye.x + f.y * eye.y + f.z * eye.z,
                            w: 1.0,
                        },
                    }
                }

                /// Create uniform scaling matrix.
                pub fn from_scale(s: Vec3<$t>) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 { x: s.x, y: 0.0, z: 0.0, w: 0.0 },
                        y: Vec4 { x: 0.0, y: s.y, z: 0.0, w: 0.0 },
                        z: Vec4 { x: 0.0, y: 0.0, z: s.z, w: 0.0 },
                        w: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                    }
                }

                /// Extract translation component.
                pub fn translation(&self) -> Vec3<$t> {
                    Vec3 { x: self.w.x, y: self.w.y, z: self.w.z }
                }

                /// Extract scale factors (column lengths).
                pub fn scale(&self) -> Vec3<$t> {
                    Vec3 {
                        x: (self.x.x * self.x.x + self.x.y * self.x.y + self.x.z * self.x.z).sqrt(),
                        y: (self.y.x * self.y.x + self.y.y * self.y.y + self.y.z * self.y.z).sqrt(),
                        z: (self.z.x * self.z.x + self.z.y * self.z.y + self.z.z * self.z.z).sqrt(),
                    }
                }

                /// Extract the upper-left 3x3 rotation matrix (scale removed).
                ///
                /// Divides each column by its length to strip scaling. Returns a
                /// zero column if the corresponding scale factor is zero.
                pub fn rotation(&self) -> Mat3x3<$t> {
                    let s = self.scale();
                    let sx = if s.x != 0.0 { 1.0 / s.x } else { 0.0 };
                    let sy = if s.y != 0.0 { 1.0 / s.y } else { 0.0 };
                    let sz = if s.z != 0.0 { 1.0 / s.z } else { 0.0 };
                    Mat3x3 {
                        x: Vec3 { x: self.x.x * sx, y: self.x.y * sx, z: self.x.z * sx },
                        y: Vec3 { x: self.y.x * sy, y: self.y.y * sy, z: self.y.z * sy },
                        z: Vec3 { x: self.z.x * sz, y: self.z.y * sz, z: self.z.z * sz },
                    }
                }

                /// Calculate determinant of matrix.
                pub fn det(self) -> $t {

                    let a = self.x.x;
                    let b = self.x.y;
                    let c = self.x.z;
                    let d = self.x.w;
                    let e = self.y.x;
                    let f = self.y.y;
                    let g = self.y.z;
                    let h = self.y.w;
                    let i = self.z.x;
                    let j = self.z.y;
                    let k = self.z.z;
                    let l = self.z.w;
                    let m = self.w.x;
                    let n = self.w.y;
                    let o = self.w.z;
                    let p = self.w.w;

                    let kpol = k * p - o * l;
                    let jpnl = j * p - n * l;
                    let jonk = j * o - n * k;
                    let ipml = i * p - m * l;
                    let iomk = i * o - m * k;
                    let inmj = i * n - m * j;

                    let ba = f * kpol - g * jpnl + h * jonk;
                    let bb = -e * kpol + g * ipml - h * iomk;
                    let bc = e * jpnl - f * ipml + h * inmj;
                    let bd = -e * jonk + f * iomk - g * inmj;

                    a * ba + b * bb + c * bc + d * bd
                }
            }

            impl Zero for Mat4x4<$t> {
                const ZERO: Mat4x4<$t> = Mat4x4 {
                    x: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
                    y: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
                    z: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
                    w: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
                };
            }

            impl One for Mat4x4<$t> {
                const ONE: Mat4x4<$t> = Mat4x4 {
                    x: Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                    y: Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
                    z: Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
                    w: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                };
            }

            impl Display for Mat4x4<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "[{},{},{},{}]", self.x, self.y, self.z, self.w)
                }
            }

            /// Matrix + matrix.
            impl Add<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self::Output {
                    Mat4x4 {
                        x: self.x + other.x,
                        y: self.y + other.y,
                        z: self.z + other.z,
                        w: self.w + other.w,
                    }
                }
            }

            /// Matrix += matrix.
            impl AddAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn add_assign(&mut self, other: Self) {
                    self.x += other.x;
                    self.y += other.y;
                    self.z += other.z;
                    self.w += other.w;
                }
            }

            /// Matrix - matrix.
            impl Sub<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self::Output {
                    Mat4x4 {
                        x: self.x - other.x,
                        y: self.y - other.y,
                        z: self.z - other.z,
                        w: self.w - other.w,
                    }
                }
            }

            /// Matrix -= matrix.
            impl SubAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn sub_assign(&mut self, other: Self) {
                    self.x -= other.x;
                    self.y -= other.y;
                    self.z -= other.z;
                    self.w -= other.w;
                }
            }

            /// Matrix * scalar.
            impl Mul<$t> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn mul(self, other: $t) -> Self::Output {
                    Mat4x4 {
                        x: self.x * other,
                        y: self.y * other,
                        z: self.z * other,
                        w: self.w * other,
                    }
                }
            }

            /// Scalar * matrix.
            impl Mul<Mat4x4<$t>> for $t {
                type Output = Mat4x4<$t>;
                fn mul(self, other: Mat4x4<$t>) -> Self::Output {
                    Mat4x4 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                        w: self * other.w,
                    }
                }
            }

            /// Matrix * vector.
            impl Mul<Vec4<$t>> for Mat4x4<$t> {
                type Output = Vec4<$t>;
                fn mul(self, other: Vec4<$t>) -> Self::Output {
                    Vec4 {
                        x: self.x.x * other.x + self.y.x * other.y + self.z.x * other.z + self.w.x * other.w,
                        y: self.x.y * other.x + self.y.y * other.y + self.z.y * other.z + self.w.y * other.w,
                        z: self.x.z * other.x + self.y.z * other.y + self.z.z * other.z + self.w.z * other.w,
                        w: self.x.w * other.x + self.y.w * other.y + self.z.w * other.z + self.w.w * other.w,
                    }
                }
            }

            /// Matrix * matrix.
            impl Mul<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn mul(self, other: Mat4x4<$t>) -> Self::Output {
                    Mat4x4 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                        w: self * other.w,
                    }
                }
            }

            /// Matrix *= scalar.
            impl MulAssign<$t> for Mat4x4<$t> {
                fn mul_assign(&mut self, other: $t) {
                    self.x *= other;
                    self.y *= other;
                    self.z *= other;
                    self.w *= other;
                }
            }

            /// Matrix *= matrix.
            impl MulAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn mul_assign(&mut self, other: Mat4x4<$t>) {
                    let m = *self * other;
                    *self = m;
                }
            }

            /// Matrix / scalar.
            impl Div<$t> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn div(self, other: $t) -> Self::Output {
                    Mat4x4 {
                        x: self.x / other,
                        y: self.y / other,
                        z: self.z / other,
                        w: self.w / other,
                    }
                }
            }

            /// Scalar / matrix.
            impl Div<Mat4x4<$t>> for $t {
                type Output = Mat4x4<$t>;
                fn div(self, other: Mat4x4<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix / matrix.
            impl Div<Mat4x4<$t>> for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn div(self, other: Mat4x4<$t>) -> Self::Output {
                    self * other.inv()
                }
            }

            /// Matrix /= scalar.
            impl DivAssign<$t> for Mat4x4<$t> {
                fn div_assign(&mut self, other: $t) {
                    self.x /= other;
                    self.y /= other;
                    self.z /= other;
                    self.w /= other;
                }
            }

            /// Matrix /= matrix.
            impl DivAssign<Mat4x4<$t>> for Mat4x4<$t> {
                fn div_assign(&mut self, other: Mat4x4<$t>) {
                    *self *= other.inv()
                }
            }

            /// -Matrix.
            impl Neg for Mat4x4<$t> {
                type Output = Mat4x4<$t>;
                fn neg(self) -> Self::Output {
                    Mat4x4 {
                        x: -self.x,
                        y: -self.y,
                        z: -self.z,
                        w: -self.w,
                    }
                }
            }

            /// Convert quaternion to 4x4 matrix.
            impl From<Quat<$t>> for Mat4x4<$t> {
                fn from(value: Quat<$t>) -> Mat4x4<$t> {
                    let m: Mat3x3<$t> = value.into();
                    m.into()
                }
            }

            /// Compose matrix from pose.
            impl From<Pose<$t>> for Mat4x4<$t> {
                fn from(value: Pose<$t>) -> Mat4x4<$t> {
                    let rotation: Mat3x3<$t> = value.o.into();
                    Mat4x4 {
                        x: Vec4 {
                            x: rotation.x.x,
                            y: rotation.x.y,
                            z: rotation.x.z,
                            w: 0.0,
                        },
                        y: Vec4 {
                            x: rotation.y.x,
                            y: rotation.y.y,
                            z: rotation.y.z,
                            w: 0.0,
                        },
                        z: Vec4 {
                            x: rotation.z.x,
                            y: rotation.z.y,
                            z: rotation.z.z,
                            w: 0.0,
                        },
                        w: Vec4 {
                            x: value.p.x,
                            y: value.p.y,
                            z: value.p.z,
                            w: 1.0,
                        },
                    }
                }
            }

            /// Create translation matrix from vector.
            impl From<Vec3<$t>> for Mat4x4<$t> {
                fn from(value: Vec3<$t>) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 { x: 1.0, y: 0.0, z: 0.0, w: 0.0 },
                        y: Vec4 { x: 0.0, y: 1.0, z: 0.0, w: 0.0 },
                        z: Vec4 { x: 0.0, y: 0.0, z: 1.0, w: 0.0 },
                        w: Vec4 {
                            x: value.x,
                            y: value.y,
                            z: value.z,
                            w: 1.0,
                        },
                    }
                }
            }

            /// Extend 3x3 matrix to 4x4 matrix.
            impl From<Mat3x3<$t>> for Mat4x4<$t> {
                fn from(value: Mat3x3<$t>) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 {
                            x: value.x.x,
                            y: value.x.y,
                            z: value.x.z,
                            w: 0.0,
                        },
                        y: Vec4 {
                            x: value.y.x,
                            y: value.y.y,
                            z: value.y.z,
                            w: 0.0,
                        },
                        z: Vec4 {
                            x: value.z.x,
                            y: value.z.y,
                            z: value.z.z,
                            w: 0.0,
                        },
                        w: Vec4 { x: 0.0, y: 0.0, z: 0.0, w: 1.0 },
                    }
                }
            }
        )+
    }
}

mat4x4_impl! { f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! mat4x4_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Mat4x4<$t>> for Mat4x4<$u> {
                fn from(value: Mat4x4<$t>) -> Self { Mat4x4 { x: value.x.into(),y: value.y.into(),z: value.z.into(),w: value.w.into(), } }
            }
        )+
    }
}

mat4x4_from_impl! { (f32,f64) (f64,f32) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_vec3() {
        let result: Mat4x4<f32> = Vec3::<f32> {
            x: 2.0,
            y: 3.0,
            z: 4.0,
        }
        .into();
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                    w: 0.0,
                },
                w: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 1.0,
                },
            }
        );
    }

    #[test]
    fn from_mat3x3() {
        let result: Mat4x4<f32> = Mat3x3::<f32> {
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
        .into();
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 5.0,
                    y: 6.0,
                    z: 7.0,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 8.0,
                    y: 9.0,
                    z: 10.0,
                    w: 0.0,
                },
                w: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
            }
        );
    }

    #[test]
    fn from_quat() {
        let result: Mat4x4<f32> = Quat::<f32>::from_axis_angle(
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
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 0.7071067,
                    z: 0.7071068,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: -0.7071068,
                    z: 0.7071067,
                    w: 0.0,
                },
                w: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                },
            }
        );
    }

    #[test]
    fn perspective() {
        let result = Mat4x4::<f32>::perspective(1.570796327, 1.0, 0.1, 100.0);
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -100.1 / 99.9,
                    w: -1.0,
                },
                w: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -20.0 / 99.9,
                    w: 0.0,
                },
            }
        );
    }

    #[test]
    fn frustum() {
        let result = Mat4x4::<f32>::frustum(-0.1, 0.1, -0.1, 0.1, 0.1, 100.0);
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -100.1 / 99.9,
                    w: -1.0,
                },
                w: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -20.0 / 99.9,
                    w: 0.0,
                },
            }
        );
    }

    #[test]
    fn from_fov() {
        let result = Mat4x4::<f32>::perspective_from_fov(
            Fov {
                l: -0.785398163,
                r: 0.785398163,
                b: -0.785398163,
                t: 0.785398163,
            },
            0.1,
            100.0,
        );
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 1.0,
                    z: 0.0,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -100.1 / 99.9,
                    w: -1.0,
                },
                w: Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: -20.0 / 99.9,
                    w: 0.0,
                },
            }
        );
    }

    #[test]
    fn from_pose() {
        let result: Mat4x4<f32> = Pose::<f32> {
            p: Vec3 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
            },
            o: Quat::<f32>::from_axis_angle(
                Vec3 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
                0.785398163,
            ),
        }
        .into();
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                    w: 0.0,
                },
                y: Vec4 {
                    x: 0.0,
                    y: 0.7071067,
                    z: 0.7071068,
                    w: 0.0,
                },
                z: Vec4 {
                    x: 0.0,
                    y: -0.7071068,
                    z: 0.7071067,
                    w: 0.0,
                },
                w: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 1.0,
                },
            }
        );
    }

    #[test]
    fn transpose() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        }
        .transpose();
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 2.0,
                    y: 6.0,
                    z: 10.0,
                    w: 14.0,
                },
                y: Vec4 {
                    x: 3.0,
                    y: 7.0,
                    z: 11.0,
                    w: 15.0,
                },
                z: Vec4 {
                    x: 4.0,
                    y: 8.0,
                    z: 12.0,
                    w: 16.0,
                },
                w: Vec4 {
                    x: 5.0,
                    y: 9.0,
                    z: 13.0,
                    w: 17.0,
                },
            }
        );
    }

    #[test]
    fn det() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        }
        .det();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn add() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        } + Mat4x4::<f32> {
            x: Vec4 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
                w: 0.0,
            },
            y: Vec4 {
                x: -1.0,
                y: -2.0,
                z: -3.0,
                w: -4.0,
            },
            z: Vec4 {
                x: -5.0,
                y: -6.0,
                z: -7.0,
                w: -8.0,
            },
            w: Vec4 {
                x: -9.0,
                y: -10.0,
                z: -11.0,
                w: -12.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                z: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                w: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
            }
        );
    }

    #[test]
    fn add_assign() {
        let mut result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        };
        result += Mat4x4::<f32> {
            x: Vec4 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
                w: 0.0,
            },
            y: Vec4 {
                x: -1.0,
                y: -2.0,
                z: -3.0,
                w: -4.0,
            },
            z: Vec4 {
                x: -5.0,
                y: -6.0,
                z: -7.0,
                w: -8.0,
            },
            w: Vec4 {
                x: -9.0,
                y: -10.0,
                z: -11.0,
                w: -12.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                z: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
                w: Vec4 {
                    x: 5.0,
                    y: 5.0,
                    z: 5.0,
                    w: 5.0,
                },
            }
        );
    }

    #[test]
    fn sub() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        } - Mat4x4::<f32> {
            x: Vec4 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
                w: 0.0,
            },
            y: Vec4 {
                x: -1.0,
                y: -2.0,
                z: -3.0,
                w: -4.0,
            },
            z: Vec4 {
                x: -5.0,
                y: -6.0,
                z: -7.0,
                w: -8.0,
            },
            w: Vec4 {
                x: -9.0,
                y: -10.0,
                z: -11.0,
                w: -12.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: -1.0,
                    y: 1.0,
                    z: 3.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 7.0,
                    y: 9.0,
                    z: 11.0,
                    w: 13.0,
                },
                z: Vec4 {
                    x: 15.0,
                    y: 17.0,
                    z: 19.0,
                    w: 21.0,
                },
                w: Vec4 {
                    x: 23.0,
                    y: 25.0,
                    z: 27.0,
                    w: 29.0,
                },
            }
        );
    }

    #[test]
    fn sub_assign() {
        let mut result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        };
        result -= Mat4x4::<f32> {
            x: Vec4 {
                x: 3.0,
                y: 2.0,
                z: 1.0,
                w: 0.0,
            },
            y: Vec4 {
                x: -1.0,
                y: -2.0,
                z: -3.0,
                w: -4.0,
            },
            z: Vec4 {
                x: -5.0,
                y: -6.0,
                z: -7.0,
                w: -8.0,
            },
            w: Vec4 {
                x: -9.0,
                y: -10.0,
                z: -11.0,
                w: -12.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: -1.0,
                    y: 1.0,
                    z: 3.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 7.0,
                    y: 9.0,
                    z: 11.0,
                    w: 13.0,
                },
                z: Vec4 {
                    x: 15.0,
                    y: 17.0,
                    z: 19.0,
                    w: 21.0,
                },
                w: Vec4 {
                    x: 23.0,
                    y: 25.0,
                    z: 27.0,
                    w: 29.0,
                },
            }
        );
    }

    #[test]
    fn mul_sm() {
        let result = 2.0
            * Mat4x4::<f32> {
                x: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 6.0,
                    y: 7.0,
                    z: 8.0,
                    w: 9.0,
                },
                z: Vec4 {
                    x: 10.0,
                    y: 11.0,
                    z: 12.0,
                    w: 13.0,
                },
                w: Vec4 {
                    x: 14.0,
                    y: 15.0,
                    z: 16.0,
                    w: 17.0,
                },
            };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                    w: 10.0,
                },
                y: Vec4 {
                    x: 12.0,
                    y: 14.0,
                    z: 16.0,
                    w: 18.0,
                },
                z: Vec4 {
                    x: 20.0,
                    y: 22.0,
                    z: 24.0,
                    w: 26.0,
                },
                w: Vec4 {
                    x: 28.0,
                    y: 30.0,
                    z: 32.0,
                    w: 34.0,
                },
            }
        );
    }

    #[test]
    fn mul_ms() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        } * 2.0;
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                    w: 10.0,
                },
                y: Vec4 {
                    x: 12.0,
                    y: 14.0,
                    z: 16.0,
                    w: 18.0,
                },
                z: Vec4 {
                    x: 20.0,
                    y: 22.0,
                    z: 24.0,
                    w: 26.0,
                },
                w: Vec4 {
                    x: 28.0,
                    y: 30.0,
                    z: 32.0,
                    w: 34.0,
                },
            }
        );
    }

    #[test]
    fn mul_mv() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        } * Vec4 {
            x: 2.0,
            y: 3.0,
            z: 4.0,
            w: 5.0,
        };
        assert_eq!(
            result,
            Vec4::<f32> {
                x: 132.0,
                y: 146.0,
                z: 160.0,
                w: 174.0,
            }
        );
    }

    #[test]
    fn mul_mm() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        } * Mat4x4::<f32> {
            x: Vec4 {
                x: 18.0,
                y: 19.0,
                z: 20.0,
                w: 21.0,
            },
            y: Vec4 {
                x: 22.0,
                y: 23.0,
                z: 24.0,
                w: 25.0,
            },
            z: Vec4 {
                x: 26.0,
                y: 27.0,
                z: 28.0,
                w: 29.0,
            },
            w: Vec4 {
                x: 30.0,
                y: 31.0,
                z: 32.0,
                w: 33.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 644.0,
                    y: 722.0,
                    z: 800.0,
                    w: 878.0,
                },
                y: Vec4 {
                    x: 772.0,
                    y: 866.0,
                    z: 960.0,
                    w: 1054.0,
                },
                z: Vec4 {
                    x: 900.0,
                    y: 1010.0,
                    z: 1120.0,
                    w: 1230.0,
                },
                w: Vec4 {
                    x: 1028.0,
                    y: 1154.0,
                    z: 1280.0,
                    w: 1406.0,
                },
            }
        );
    }

    #[test]
    fn mul_assign_ms() {
        let mut result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        };
        result *= 2.0;
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 4.0,
                    y: 6.0,
                    z: 8.0,
                    w: 10.0,
                },
                y: Vec4 {
                    x: 12.0,
                    y: 14.0,
                    z: 16.0,
                    w: 18.0,
                },
                z: Vec4 {
                    x: 20.0,
                    y: 22.0,
                    z: 24.0,
                    w: 26.0,
                },
                w: Vec4 {
                    x: 28.0,
                    y: 30.0,
                    z: 32.0,
                    w: 34.0,
                },
            }
        );
    }

    #[test]
    fn mul_assign_mm() {
        let mut result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        };
        result *= Mat4x4::<f32> {
            x: Vec4 {
                x: 18.0,
                y: 19.0,
                z: 20.0,
                w: 21.0,
            },
            y: Vec4 {
                x: 22.0,
                y: 23.0,
                z: 24.0,
                w: 25.0,
            },
            z: Vec4 {
                x: 26.0,
                y: 27.0,
                z: 28.0,
                w: 29.0,
            },
            w: Vec4 {
                x: 30.0,
                y: 31.0,
                z: 32.0,
                w: 33.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 644.0,
                    y: 722.0,
                    z: 800.0,
                    w: 878.0,
                },
                y: Vec4 {
                    x: 772.0,
                    y: 866.0,
                    z: 960.0,
                    w: 1054.0,
                },
                z: Vec4 {
                    x: 900.0,
                    y: 1010.0,
                    z: 1120.0,
                    w: 1230.0,
                },
                w: Vec4 {
                    x: 1028.0,
                    y: 1154.0,
                    z: 1280.0,
                    w: 1406.0,
                },
            }
        );
    }

    #[test]
    fn div_ms() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 4.0,
                y: 6.0,
                z: 8.0,
                w: 10.0,
            },
            y: Vec4 {
                x: 12.0,
                y: 14.0,
                z: 16.0,
                w: 18.0,
            },
            z: Vec4 {
                x: 20.0,
                y: 22.0,
                z: 24.0,
                w: 26.0,
            },
            w: Vec4 {
                x: 28.0,
                y: 30.0,
                z: 32.0,
                w: 34.0,
            },
        } / 2.0;
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 6.0,
                    y: 7.0,
                    z: 8.0,
                    w: 9.0,
                },
                z: Vec4 {
                    x: 10.0,
                    y: 11.0,
                    z: 12.0,
                    w: 13.0,
                },
                w: Vec4 {
                    x: 14.0,
                    y: 15.0,
                    z: 16.0,
                    w: 17.0,
                },
            }
        );
    }

    #[test]
    fn div_assign_ms() {
        let mut result = Mat4x4::<f32> {
            x: Vec4 {
                x: 4.0,
                y: 6.0,
                z: 8.0,
                w: 10.0,
            },
            y: Vec4 {
                x: 12.0,
                y: 14.0,
                z: 16.0,
                w: 18.0,
            },
            z: Vec4 {
                x: 20.0,
                y: 22.0,
                z: 24.0,
                w: 26.0,
            },
            w: Vec4 {
                x: 28.0,
                y: 30.0,
                z: 32.0,
                w: 34.0,
            },
        };
        result /= 2.0;
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: 2.0,
                    y: 3.0,
                    z: 4.0,
                    w: 5.0,
                },
                y: Vec4 {
                    x: 6.0,
                    y: 7.0,
                    z: 8.0,
                    w: 9.0,
                },
                z: Vec4 {
                    x: 10.0,
                    y: 11.0,
                    z: 12.0,
                    w: 13.0,
                },
                w: Vec4 {
                    x: 14.0,
                    y: 15.0,
                    z: 16.0,
                    w: 17.0,
                },
            }
        );
    }

    #[test]
    fn neg() {
        let result = -Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 3.0,
                z: 4.0,
                w: 5.0,
            },
            y: Vec4 {
                x: 6.0,
                y: 7.0,
                z: 8.0,
                w: 9.0,
            },
            z: Vec4 {
                x: 10.0,
                y: 11.0,
                z: 12.0,
                w: 13.0,
            },
            w: Vec4 {
                x: 14.0,
                y: 15.0,
                z: 16.0,
                w: 17.0,
            },
        };
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: -2.0,
                    y: -3.0,
                    z: -4.0,
                    w: -5.0,
                },
                y: Vec4 {
                    x: -6.0,
                    y: -7.0,
                    z: -8.0,
                    w: -9.0,
                },
                z: Vec4 {
                    x: -10.0,
                    y: -11.0,
                    z: -12.0,
                    w: -13.0,
                },
                w: Vec4 {
                    x: -14.0,
                    y: -15.0,
                    z: -16.0,
                    w: -17.0,
                },
            }
        );
    }

    #[test]
    fn inv() {
        let result = Mat4x4::<f32> {
            x: Vec4 {
                x: 2.0,
                y: 7.0,
                z: 0.0,
                w: 3.0,
            },
            y: Vec4 {
                x: 3.0,
                y: 9.0,
                z: 1.0,
                w: 3.0,
            },
            z: Vec4 {
                x: 4.0,
                y: 4.0,
                z: 2.0,
                w: 9.0,
            },
            w: Vec4 {
                x: 5.0,
                y: 2.0,
                z: 7.0,
                w: 3.0,
            },
        }
        .inv();
        assert_eq!(
            result,
            Mat4x4::<f32> {
                x: Vec4 {
                    x: -145.0 / 8.0,
                    y: 109.0 / 8.0,
                    z: 19.0 / 8.0,
                    w: -21.0 / 8.0,
                },
                y: Vec4 {
                    x: 7.0 / 2.0,
                    y: -5.0 / 2.0,
                    z: -1.0 / 2.0,
                    w: 1.0 / 2.0,
                },
                z: Vec4 {
                    x: 81.0 / 8.0,
                    y: -61.0 / 8.0,
                    z: -11.0 / 8.0,
                    w: 13.0 / 8.0,
                },
                w: Vec4 {
                    x: 17.0 / 4.0,
                    y: -13.0 / 4.0,
                    z: -5.0 / 12.0,
                    w: 7.0 / 12.0,
                },
            }
        );
    }
}
