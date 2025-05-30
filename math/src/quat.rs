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

/// Quaternion of real numbers.
/// 
/// A quaternion is a way to represent 3D orientation and allow for correct rotations without gimbal lock. The concept is
/// similar to [`Complex`], where imaginary numbers are combined with scalars. The [`Quat`] adds three separate
/// imaginary numbers, allowing rotations around 3 orthogonal axes.
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Quat<T> {
    pub r: T,
    pub i: T,
    pub j: T,
    pub k: T,
}

impl<T> Quat<T> where T: Zero + One {
    pub const UNIT_I: Self = Quat { r: T::ZERO,i: T::ONE,j: T::ZERO,k: T::ZERO, };
    pub const UNIT_J: Self = Quat { r: T::ZERO,i: T::ONE,j: T::ZERO,k: T::ZERO, };
    pub const UNIT_K: Self = Quat { r: T::ZERO,i: T::ONE,j: T::ZERO,k: T::ZERO, };
}

impl<T> Zero for Quat<T> where T: Zero {
    const ZERO: Self = Quat { r: T::ZERO,i: T::ZERO,j: T::ZERO,k: T::ZERO, };
}

impl<T> One for Quat<T> where T: Zero + One {
    const ONE: Self = Quat { r: T::ONE,i: T::ZERO,j: T::ZERO,k: T::ZERO, };
}

impl<T> Display for Quat<T> where T: Display + Zero + PartialOrd {
    fn fmt(&self,f: &mut Formatter) -> Result {
        let si = if self.i < T::ZERO {
            format!("{}i",self.i)
        } else {
            format!("+{}i",self.i)
        };
        let sj = if self.j < T::ZERO {
            format!("{}j",self.j)
        } else {
            format!("+{}j",self.j)
        };
        let sk = if self.k < T::ZERO {
            format!("{}k",self.k)
        } else {
            format!("+{}k",self.k)
        };
        write!(f,"{}{}{}{}",self.r,si,sj,sk)        
    }            
}

/// Quaternion + scalar.
impl<T> Add<T> for Quat<T> where T: Add<Output=T> {
    type Output = Quat<T>;
    fn add(self,other: T) -> Self::Output {
        Quat {
            r: self.r + other,
            i: self.i,
            j: self.j,
            k: self.k,
        }
    }
}

/// Quaternion += scalar.
impl<T> AddAssign<T> for Quat<T> where T: AddAssign {
    fn add_assign(&mut self,other: T) {
        self.r += other;
    }
}

/// Quaternion - scalar.
impl<T> Sub<T> for Quat<T> where T: Sub<Output=T> {
    type Output = Quat<T>;
    fn sub(self,other: T) -> Self::Output {
        Quat {
            r: self.r - other,
            i: self.i,
            j: self.j,
            k: self.k,
        }
    }
}

/// Quaternion -= scalar.
impl<T> SubAssign<T> for Quat<T> where T: SubAssign {
    fn sub_assign(&mut self,other: T) {
        self.r -= other;
    }
}

/// Quaternion * scalar.
impl<T> Mul<T> for Quat<T> where T: Copy + Mul<Output=T> {
    type Output = Quat<T>;
    fn mul(self,other: T) -> Self::Output {
        Quat {
            r: self.r * other,
            i: self.i * other,
            j: self.j * other,
            k: self.k * other,
        }
    }
}

/// Quaternion * quaternion.
impl<T> Mul<Quat<T>> for Quat<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = Quat<T>;
    fn mul(self,other: Quat<T>) -> Self::Output {
        Quat {
            r: self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k,
            i: self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j,
            j: self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i,
            k: self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r,
        }
    }
}

/// Quaternion * vector.
impl<T> Mul<Vec3<T>> for Quat<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = Vec3<T>;
    fn mul(self,other: Vec3<T>) -> Self::Output {
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
impl<T> MulAssign<T> for Quat<T> where T: Copy + MulAssign {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.i *= other;
        self.j *= other;
        self.k *= other;
    }
}

/// Quaternion *= quaternion.
impl<T> MulAssign<Quat<T>> for Quat<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    fn mul_assign(&mut self,other: Quat<T>) {
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
impl<T> Div<T> for Quat<T> where T: Copy + Div<Output=T> {
    type Output = Quat<T>;
    fn div(self,other: T) -> Self::Output {
        Quat {
            r: self.r / other,
            i: self.i / other,
            j: self.j / other,
            k: self.k / other,
        }
    }
}

/// Quaternion / quaternion.
impl<T> Div<Quat<T>> for Quat<T> where T: Copy + Div<Output=T> + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = Quat<T>;
    fn div(self,other: Quat<T>) -> Self::Output {
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
impl<T> DivAssign<T> for Quat<T> where T: Copy + DivAssign {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.i /= other;
        self.j /= other;
        self.k /= other;
    }
}

/// Quaternion /= quaternion.
impl<T> DivAssign<Quat<T>> for Quat<T> where T: Copy + Div<Output=T> + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    fn div_assign(&mut self,other: Quat<T>) {
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
impl<T> Neg for Quat<T> where T: Neg<Output=T> {
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

macro_rules! quat_real_impl {
    ($($t:ty)+) => {
        $(
            impl Quat<$t> {
            
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
                pub fn conj(&self) -> Self {
                    Quat {
                        r: self.r,
                        i: -self.i,
                        j: -self.j,
                        k: -self.k,
                    }
                }
            
                /// Invert the quaternion.
                pub fn inv(&self) -> Self {
                    let f = self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k;
                    Quat {
                        r: self.r / f,
                        i: -self.i / f,
                        j: -self.j / f,
                        k: -self.k / f,
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

quat_real_impl! { f32 f64 }

// if `T as U` exists, `Quat<U>::from(Quat<T>)` should also exist
impl From<Quat<f32>> for Quat<f64> { fn from(value: Quat<f32>) -> Self { Quat { r: value.r as f64,i: value.i as f64,j: value.j as f64,k: value.k as f64, } } }
impl From<Quat<f64>> for Quat<f32> { fn from(value: Quat<f64>) -> Self { Quat { r: value.r as f32,i: value.i as f32,j: value.j as f32,k: value.k as f32, } } }


mod tests {
#[allow(unused_imports)]
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
}