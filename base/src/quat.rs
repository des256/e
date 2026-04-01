use {
    crate::*,
    std::{
        cmp::PartialEq,
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
        },
    },
};

/// Quaternion, generic over the component type.
///
/// Quaternions extend complex numbers with three imaginary units (`i`, `j`, `k`)
/// and are the standard way to represent 3D rotations without gimbal lock.
///
/// Supports full arithmetic with scalars, other quaternions, and [`Vec3`]
/// (rotation via `Quat * Vec3`). For `f32`/`f64`, construction helpers
/// [`from_axis_angle`](Quat::from_axis_angle) and [`from_euler`](Quat::from_euler)
/// are available, along with [`conj`](Quat::conj), [`inv`](Quat::inv),
/// [`slerp`](Quat::slerp), and [`nlerp`](Quat::nlerp).
///
/// Convert to rotation matrix via `Mat3x3::from(quat)` or `Mat4x4::from(quat)`.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// // 90-degree rotation around the X axis
/// let q = Quat::<f32>::from_axis_angle(vec3(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2);
/// let v = q * vec3(0.0, 1.0, 0.0);
/// assert!((v.z - 1.0).abs() < 1e-6);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Quat<T> {
    /// Real (scalar) part.
    pub r: T,
    /// First imaginary component (i).
    pub i: T,
    /// Second imaginary component (j).
    pub j: T,
    /// Third imaginary component (k).
    pub k: T,
}

/// Create a new quaternion.
pub const fn quat<T>(r: T, i: T, j: T, k: T) -> Quat<T> {
    Quat { r, i, j, k }
}

macro_rules! quat_impl {
    ($($t:ty)+) => {
        $(
            impl Quat<$t> {
                /// Unit quaternion along the I axis (`0 + 1i + 0j + 0k`).
                pub const UNIT_I: Self = Quat { r: 0.0,i: 1.0,j: 0.0,k: 0.0, };
                /// Unit quaternion along the J axis (`0 + 0i + 1j + 0k`).
                pub const UNIT_J: Self = Quat { r: 0.0,i: 0.0,j: 1.0,k: 0.0, };
                /// Unit quaternion along the K axis (`0 + 0i + 0j + 1k`).
                pub const UNIT_K: Self = Quat { r: 0.0,i: 0.0,j: 0.0,k: 1.0, };

                /// Convert Axis-angle to quaternion.
                pub fn from_axis_angle(axis: Vec3<$t>,angle: $t) -> Self {
                    let ha = 0.5 * angle;
                    let d = axis.length();
                    let ch = ha.cos();
                    let shd = ha.sin() / d;
                    Quat {
                        r: ch,
                        i: axis.x * shd,
                        j: axis.y * shd,
                        k: axis.z * shd,
                    }
                }

                /// Convert Euler angles (order XYZ) to quaternion.
                pub fn from_euler(x: $t,y: $t,z: $t) -> Self {
                    let hx = 0.5 * x;
                    let hy = 0.5 * y;
                    let hz = 0.5 * z;
                    let cx = hx.cos();
                    let sx = hx.sin();
                    let cy = hy.cos();
                    let sy = hy.sin();
                    let cz = hz.cos();
                    let sz = hz.sin();
                    Quat {
                        r: cx * cy * cz - sx * sy * sz,
                        i: sx * cy * cz + cx * sy * sz,
                        j: cx * sy * cz - sx * cy * sz,
                        k: cx * cy * sz + sx * sy * cz,
                    }
                }

                /// Calculate quaternion conjugate.
                pub fn conj(self) -> Self {
                    Quat {
                        r: self.r,
                        i: -self.i,
                        j: -self.j,
                        k: -self.k,
                    }
                }

                /// Invert the quaternion.
                pub fn inv(self) -> Self {
                    let f = self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k;
                    Quat {
                        r: self.r / f,
                        i: -self.i / f,
                        j: -self.j / f,
                        k: -self.k / f,
                    }
                }
            }

            impl Zero for Quat<$t> {
                const ZERO: Self = Quat { r: 0.0,i: 0.0,j: 0.0,k: 0.0, };
            }

            impl One for Quat<$t> {
                const ONE: Self = Quat { r: 1.0,i: 0.0,j: 0.0,k: 0.0, };
            }

            impl Display for Quat<$t> {
                fn fmt(&self,f: &mut Formatter) -> Result {
                    let si = if self.i < 0.0 {
                        format!("{}i",self.i)
                    } else {
                        format!("+{}i",self.i)
                    };
                    let sj = if self.j < 0.0 {
                        format!("{}j",self.j)
                    } else {
                        format!("+{}j",self.j)
                    };
                    let sk = if self.k < 0.0 {
                        format!("{}k",self.k)
                    } else {
                        format!("+{}k",self.k)
                    };
                    write!(f,"{}{}{}{}",self.r,si,sj,sk)
                }
            }

            /// Quaternion + scalar.
            impl Add<$t> for Quat<$t> {
                type Output = Quat<$t>;
                fn add(self,other: $t) -> Self::Output {
                    Quat {
                        r: self.r + other,
                        i: self.i,
                        j: self.j,
                        k: self.k,
                    }
                }
            }

            /// Quaternion += scalar.
            impl AddAssign<$t> for Quat<$t> {
                fn add_assign(&mut self,other: $t) {
                    self.r += other;
                }
            }

            /// Quaternion - scalar.
            impl Sub<$t> for Quat<$t> {
                type Output = Quat<$t>;
                fn sub(self,other: $t) -> Self::Output {
                    Quat {
                        r: self.r - other,
                        i: self.i,
                        j: self.j,
                        k: self.k,
                    }
                }
            }

            /// Quaternion -= scalar.
            impl SubAssign<$t> for Quat<$t> {
                fn sub_assign(&mut self,other: $t) {
                    self.r -= other;
                }
            }

            /// Quaternion * scalar.
            impl Mul<$t> for Quat<$t> {
                type Output = Quat<$t>;
                fn mul(self,other: $t) -> Self::Output {
                    Quat {
                        r: self.r * other,
                        i: self.i * other,
                        j: self.j * other,
                        k: self.k * other,
                    }
                }
            }

            /// Quaternion * quaternion.
            impl Mul<Quat<$t>> for Quat<$t> {
                type Output = Quat<$t>;
                fn mul(self,other: Quat<$t>) -> Self::Output {
                    Quat {
                        r: self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k,
                        i: self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j,
                        j: self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i,
                        k: self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r,
                    }
                }
            }

            /// Quaternion * vector.
            impl Mul<Vec3<$t>> for Quat<$t> {
                type Output = Vec3<$t>;
                fn mul(self,other: Vec3<$t>) -> Self::Output {
                    let rr = self.r * self.r;
                    let ri = self.r * self.i;
                    let rj = self.r * self.j;
                    let rk = self.r * self.k;
                    let ii = self.i * self.i;
                    let ij = self.i * self.j;
                    let ik = self.i * self.k;
                    let jj = self.j * self.j;
                    let jk = self.j * self.k;
                    let kk = self.k * self.k;
                    let ijprk = ij + rk;
                    let ijprk2 = ijprk + ijprk;
                    let ijmrk = ij - rk;
                    let ijmrk2 = ijmrk + ijmrk;
                    let jkpri = jk + ri;
                    let jkpri2 = jkpri + jkpri;
                    let jkmri = jk - ri;
                    let jkmri2 = jkmri + jkmri;
                    let ikprj = ik + rj;
                    let ikprj2 = ikprj + ikprj;
                    let ikmrj = ik - rj;
                    let ikmrj2 = ikmrj + ikmrj;
                    Vec3 {
                        x: (rr + ii - jj - kk) * other.x + ijmrk2 * other.y + ikprj2 * other.z,
                        y: (rr - ii + jj - kk) * other.y + jkmri2 * other.z + ijprk2 * other.x,
                        z: (rr - ii - jj + kk) * other.z + ikmrj2 * other.x + jkpri2 * other.y,
                    }
                }
            }

            /// Quaternion *= scalar.
            impl MulAssign<$t> for Quat<$t> {
                fn mul_assign(&mut self,other: $t) {
                    self.r *= other;
                    self.i *= other;
                    self.j *= other;
                    self.k *= other;
                }
            }

            /// Quaternion *= quaternion.
            impl MulAssign<Quat<$t>> for Quat<$t> {
                fn mul_assign(&mut self,other: Quat<$t>) {
                    let r = self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k;
                    let i = self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j;
                    let j = self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i;
                    let k = self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r;
                    self.r = r;
                    self.i = i;
                    self.j = j;
                    self.k = k;
                }
            }

            /// Quaternion / scalar.
            impl Div<$t> for Quat<$t> {
                type Output = Quat<$t>;
                fn div(self,other: $t) -> Self::Output {
                    Quat {
                        r: self.r / other,
                        i: self.i / other,
                        j: self.j / other,
                        k: self.k / other,
                    }
                }
            }

            /// Quaternion / quaternion.
            impl Div<Quat<$t>> for Quat<$t> {
                type Output = Quat<$t>;
                fn div(self,other: Quat<$t>) -> Self::Output {
                    let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
                    Quat {
                        r: (self.r * other.r + self.i * other.i + self.j * other.j + self.k * other.k) / f,
                        i: (self.i * other.r - self.j * other.k + self.k * other.j - self.r * other.i) / f,
                        j: (self.j * other.r - self.k * other.i - self.r * other.j + self.i * other.k) / f,
                        k: (self.k * other.r - self.r * other.k - self.i * other.j + self.j * other.i) / f,
                    }
                }
            }

            /// Quaternion /= scalar.
            impl DivAssign<$t> for Quat<$t> {
                fn div_assign(&mut self,other: $t) {
                    self.r /= other;
                    self.i /= other;
                    self.j /= other;
                    self.k /= other;
                }
            }

            /// Quaternion /= quaternion.
            impl DivAssign<Quat<$t>> for Quat<$t> {
                fn div_assign(&mut self,other: Quat<$t>) {
                    let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
                    let r = (self.r * other.r + self.i * other.i + self.j * other.j + self.k * other.k) / f;
                    let i = (self.i * other.r - self.j * other.k + self.k * other.j - self.r * other.i) / f;
                    let j = (self.j * other.r - self.k * other.i - self.r * other.j + self.i * other.k) / f;
                    let k = (self.k * other.r - self.r * other.k - self.i * other.j + self.j * other.i) / f;
                    self.r = r;
                    self.i = i;
                    self.j = j;
                    self.k = k;
                }
            }

            /// -Quaternion.
            impl Neg for Quat<$t> {
                type Output = Self;
                fn neg(self) -> Self::Output {
                    Quat {
                        r: -self.r,
                        i: -self.i,
                        j: -self.j,
                        k: -self.k,
                    }
                }
            }

            /// Scalar + quaternion.
            impl Add<Quat<$t>> for $t {
                type Output = Quat<$t>;
                fn add(self,other: Quat<$t>) -> Self::Output {
                    Quat {
                        r: self + other.r,
                        i: other.i,
                        j: other.j,
                        k: other.k,
                    }
                }
            }

            /// Scalar - quaternion.
            impl Sub<Quat<$t>> for $t {
                type Output = Quat<$t>;
                fn sub(self,other: Quat<$t>) -> Self::Output {
                    Quat {
                        r: self - other.r,
                        i: -other.i,
                        j: -other.j,
                        k: -other.k,
                    }
                }
            }

            /// Scalar * quaternion.
            impl Mul<Quat<$t>> for $t {
                type Output = Quat<$t>;
                fn mul(self,other: Quat<$t>) -> Self::Output {
                    Quat {
                        r: self * other.r,
                        i: self * other.i,
                        j: self * other.j,
                        k: self * other.k,
                    }
                }
            }

            /// Scalar / quaternion.
            impl Div<Quat<$t>> for $t {
                type Output = Quat<$t>;
                fn div(self,other: Quat<$t>) -> Self::Output {
                    let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
                    Quat {
                        r: (self * other.r) / f,
                        i: (-self * other.i) / f,
                        j: (-self * other.j) / f,
                        k: (-self * other.k) / f,
                    }
                }
            }
        )+
    }
}

quat_impl! { f32 f64 }

// if `T as U` exists, `Quat<U>::from(Quat<T>)` should also exist
impl From<Quat<f32>> for Quat<f64> { fn from(value: Quat<f32>) -> Self { Quat { r: value.r as f64,i: value.i as f64,j: value.j as f64,k: value.k as f64, } } }
impl From<Quat<f64>> for Quat<f32> { fn from(value: Quat<f64>) -> Self { Quat { r: value.r as f32,i: value.i as f32,j: value.j as f32,k: value.k as f32, } } }


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_euler() {
        // 30 degrees around +X, then 45 degrees around +Y, then 60 degrees around +Z
        // verified from: https://www.andre-gaschler.com/rotationconverter
        let result = Quat::<f32>::from_euler(0.523598776,0.785398163,1.047197551);
        assert_eq!(result,Quat::<f32> { r: 0.7233174,i: 0.39190382,j: 0.20056215,k: 0.5319757, });
    }

    #[test]
    fn from_axis_angle() {
        // 45 degrees around +X
        let result = Quat::<f32>::from_axis_angle(Vec3 { x: 1.0,y: 0.0,z: 0.0, },0.785398163);
        assert_eq!(result,Quat::<f32> { r: 0.9238795,i: 0.38268346,j: 0.0,k: 0.0, });
    }

    #[test]
    fn conj() {
        let result = Quat::<f32> { r: 1.0,i: 2.0,j: 3.0,k: 4.0, }.conj();
        assert_eq!(result,Quat::<f32> { r: 1.0,i: -2.0,j: -3.0,k: -4.0, });
    }

    #[test]
    fn inv() {
        let result = Quat::<f32> { r: 1.0,i: 0.1,j: 0.2,k: 0.3, }.inv();
        assert_eq!(result,Quat::<f32> { r: 0.877193,i: -0.0877193,j: -0.1754386,k: -0.2631579, });
    }

    #[test]
    fn add_sq() {
        let result = 2.0 + Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn add_qs() {
        let result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, } + 2.0;
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn add_assign_qs() {
        let mut result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        result += 2.0;
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn sub_sq() {
        let result = 4.0 - Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 2.0,i: -3.0,j: -4.0,k: -5.0, });
    }

    #[test]
    fn sub_qs() {
        let result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, } - 2.0;
        assert_eq!(result,Quat::<f32> { r: 0.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn sub_assign_qs() {
        let mut result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        result -= 2.0;
        assert_eq!(result,Quat::<f32> { r: 0.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn mul_sq() {
        let result = 2.0 * Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, });
    }

    #[test]
    fn mul_qs() {
        let result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, } * 2.0;
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, });
    }

    #[test]
    fn mul_qq() {
        let result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, } * Quat::<f32> { r: 6.0,i: 7.0,j: 8.0,k: 9.0, };
        assert_eq!(result,Quat::<f32> { r: -86.0,i: 28.0,j: 48.0,k: 44.0, });
    }

    #[test]
    fn mul_qv() {
        // rotate +Y 90 degrees around the X-axis
        let result = Quat::<f32>::from_euler(1.570796327,0.0,0.0) * Vec3::<f32> { x: 0.0,y: 1.0,z: 0.0, };
        assert_eq!(result,Vec3::<f32> { x: 0.0,y: 0.0,z: 0.99999994, });
    }

    #[test]
    fn mul_assign_qs() {
        let mut result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        result *= 2.0;
        assert_eq!(result,Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, });
    }

    #[test]
    fn mul_assign_qq() {
        let mut result = Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        result *= Quat::<f32> { r: 6.0,i: 7.0,j: 8.0,k: 9.0, };
        assert_eq!(result,Quat::<f32> { r: -86.0,i: 28.0,j: 48.0,k: 44.0, });
    }

    #[test]
    fn div_sq() {
        let result = 10.0 / Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 0.37037036,i: -0.5555556,j: -0.7407407,k: -0.9259259, })
    }

    #[test]
    fn div_qs() {
        let result = Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, } / 2.0;
        assert_eq!(result,Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn div_qq() {
        let result = Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, } / Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 2.0,i: 0.0,j: 0.0,k: 0.0, });
    }

    #[test]
    fn div_assign_qs() {
        let mut result = Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, };
        result /= 2.0;
        assert_eq!(result,Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, });
    }

    #[test]
    fn div_assign_qq() {
        let mut result = Quat::<f32> { r: 4.0,i: 6.0,j: 8.0,k: 10.0, };
        result /= Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: 2.0,i: 0.0,j: 0.0,k: 0.0, });
    }

    #[test]
    fn neg() {
        let result = -Quat::<f32> { r: 2.0,i: 3.0,j: 4.0,k: 5.0, };
        assert_eq!(result,Quat::<f32> { r: -2.0,i: -3.0,j: -4.0,k: -5.0, });
    }

    #[test]
    fn test_codec_quat_roundtrip() {
        let val = quat(1.0f32, 0.0, 0.0, 0.0);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Quat::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
