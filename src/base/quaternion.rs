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

/// Quaternion template.
/// 
/// A quaternion is a way to represent 3D orientation and allow for correct rotations without gimbal lock. The concept is
/// similar to [`Complex`], where imaginary numbers are combined with scalars. The [`Quaternion`] adds three separate
/// imaginary numbers, allowing rotations around 3 orthogonal axes.
/// 
/// Can use any scalar underneath (typically [`f32`] or [`f64`]), as well as [`Rational`] and [`Fixed`] types.
#[derive(Copy,Clone,Debug)]
pub struct Quaternion<T> {
    pub r: T,
    pub i: T,
    pub j: T,
    pub k: T,
}

impl<T: Zero + Display + PartialOrd> Display for Quaternion<T> {
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

impl<T: Copy + Neg<Output=T>> Quaternion<T> {

    // quaternion conjugate
    pub fn conj(&self) -> Self {
        Quaternion {
            r: self.r,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        }
    }
}

impl<T: Copy + Add<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T> + Float> Quaternion<T> {

    // |quaternion|
    pub fn norm(&self) -> T {
        (self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k).sqrt()
    }

    // quaternion inverse
    pub fn inverse(&self) -> Quaternion<T> {
        let f = self.r * self.r + self.i * self.i + self.j * self.j + self.k * self.k;
        Quaternion {
            r: self.r / f,
            i: -self.i / f,
            j: -self.j / f,
            k: -self.k / f,
        }
    }
}

// quaternion == scalar
impl<T: Zero + PartialEq> PartialEq<T> for Quaternion<T> {
    fn eq(&self,other: &T) -> bool {
        (self.r == *other) &&
        (self.i == T::ZERO) &&
        (self.j == T::ZERO) &&
        (self.k == T::ZERO)
    }
}

// complex == quaternion
impl<T: Zero + PartialEq> PartialEq<Quaternion<T>> for Complex<T> {
    fn eq(&self,other: &Quaternion<T>) -> bool {
        (self.r == other.r) &&
        (self.i == other.i) &&
        (other.j == T::ZERO) &&
        (other.k == T::ZERO)
    }
}

// quaternion == complex
impl<T: Zero + PartialEq> PartialEq<Complex<T>> for Quaternion<T> {
    fn eq(&self,other: &Complex<T>) -> bool {
        (self.r == other.r) &&
        (self.i == other.i) &&
        (self.j == T::ZERO) &&
        (self.k == T::ZERO)
    }
}

// quaternion == quaternion
impl<T: PartialEq> PartialEq<Quaternion<T>> for Quaternion<T> {
    fn eq(&self,other: &Quaternion<T>) -> bool {
        (self.r == other.r) &&
        (self.i == other.i) &&
        (self.j == other.j) &&
        (self.k == other.k)
    }
}

// scalar ... quaternion
macro_rules! scalar_quaternion {
    ($($t:ty)*) => ($(

        // scalar == quaternion
        impl PartialEq<Quaternion<$t>> for $t {
            fn eq(&self,other: &Quaternion<$t>) -> bool {
                (self == &other.r) &&
                (other.i == <$t>::ZERO) &&
                (other.j == <$t>::ZERO) &&
                (other.k == <$t>::ZERO)
            }
        }

        // scalar + quaternion
        impl Add<Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            fn add(self,other: Quaternion<$t>) -> Quaternion<$t> {
                Quaternion {
                    r: self + other.r,
                    i: other.i,
                    j: other.j,
                    k: other.k,
                }
            }
        }

        // scalar - quaternion
        impl Sub<Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            fn sub(self,other: Quaternion<$t>) -> Quaternion<$t> {
                Quaternion {
                    r: self - other.r,
                    i: -other.i,
                    j: -other.j,
                    k: -other.k,
                }
            }
        }

        // scalar * quaternion
        impl Mul<Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            fn mul(self,other: Quaternion<$t>) -> Quaternion<$t> {
                Quaternion {
                    r: self * other.r,
                    i: self * other.i,
                    j: self * other.j,
                    k: self * other.k,
                }
            }
        }

        // scalar / quaternion
        impl Div<Quaternion<$t>> for $t {
            type Output = Quaternion<$t>;
            fn div(self,other: Quaternion<$t>) -> Quaternion<$t> {
                let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
                Quaternion {
                    r: self * other.r / f,
                    i: self * -other.i / f,
                    j: self * -other.j / f,
                    k: self * -other.k / f,
                }
            }
        }
    )*)
}

scalar_quaternion! { f32 f64 }

// quaternion + scalar
impl<T: Add<Output=T>> Add<T> for Quaternion<T> {
    type Output = Quaternion<T>;
    fn add(self,other: T) -> Self::Output {
        Quaternion {
            r: self.r + other,
            i: self.i,
            j: self.j,
            k: self.k,
        }
    }
}

// complex + quaternion
impl<T: Add<Output=T>> Add<Quaternion<T>> for Complex<T> {
    type Output = Quaternion<T>;
    fn add(self,other: Quaternion<T>) -> Self::Output {
        Quaternion {
            r: self.r + other.r,
            i: self.i + other.i,
            j: other.j,
            k: other.k,
        }
    }
}

// quaternion + complex
impl<T: Add<Output=T>> Add<Complex<T>> for Quaternion<T> {
    type Output = Self;
    fn add(self,other: Complex<T>) -> Self::Output {
        Quaternion {
            r: self.r + other.r,
            i: self.i + other.i,
            j: self.j,
            k: self.k,
        }
    }
}

// quaternion + quaternion
impl<T: Add<Output=T>> Add<Quaternion<T>> for Quaternion<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self::Output {
        Quaternion {
            r: self.r + other.r,
            i: self.i + other.i,
            j: self.j + other.j,
            k: self.k + other.k,
        }
    }
}

// quaternion += scalar
impl<T: AddAssign<T>> AddAssign<T> for Quaternion<T> {
    fn add_assign(&mut self,other: T) {
        self.r += other;
    }
}

// quaternion += complex
impl<T: AddAssign<T>> AddAssign<Complex<T>> for Quaternion<T> {
    fn add_assign(&mut self,other: Complex<T>) {
        self.r += other.r;
        self.i += other.i;
    }
}

// quaternion += quaternion
impl<T: AddAssign<T>> AddAssign<Quaternion<T>> for Quaternion<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.i += other.i;
        self.j += other.j;
        self.k += other.k;
    }
}

// quaternion - scalar
impl<T: Sub<Output=T>> Sub<T> for Quaternion<T> {
    type Output = Self;
    fn sub(self,other: T) -> Self::Output {
        Quaternion {
            r: self.r - other,
            i: self.i,
            j: self.j,
            k: self.k,
        }
    }
}

// complex - quaternion
impl<T: Sub<Output=T> + Neg<Output=T>> Sub<Quaternion<T>> for Complex<T> {
    type Output = Quaternion<T>;
    fn sub(self,other: Quaternion<T>) -> Self::Output {
        Quaternion {
            r: self.r - other.r,
            i: self.i - other.i,
            j: -other.j,
            k: -other.k,
        }
    }
}

// quaternion - complex
impl<T: Sub<Output=T>> Sub<Complex<T>> for Quaternion<T> {
    type Output = Self;
    fn sub(self,other: Complex<T>) -> Self::Output {
        Quaternion {
            r: self.r - other.r,
            i: self.i - other.i,
            j: self.j,
            k: self.k,
        }
    }
}

// quaternion - quaternion
impl<T: Sub<Output=T>> Sub<Quaternion<T>> for Quaternion<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self::Output {
        Quaternion {
            r: self.r - other.r,
            i: self.i - other.i,
            j: self.j - other.j,
            k: self.k - other.k,
        }
    }
}

// quaternion -= scalar
impl<T: SubAssign> SubAssign<T> for Quaternion<T> {
    fn sub_assign(&mut self,other: T) {
        self.r -= other;
    }
}

// quaternion -= complex
impl<T: SubAssign> SubAssign<Complex<T>> for Quaternion<T> {
    fn sub_assign(&mut self,other: Complex<T>) {
        self.r -= other.r;
        self.i -= other.i;
    }
}

// quaternion -= quaternion
impl<T: SubAssign> SubAssign<Quaternion<T>> for Quaternion<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.i -= other.i;
        self.j -= other.j;
        self.k -= other.k;
    }
}

// quaternion * scalar
impl<T: Copy + Mul<Output=T>> Mul<T> for Quaternion<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self::Output {
        Quaternion {
            r: self.r * other,
            i: self.i * other,
            j: self.j * other,
            k: self.k * other,
        }
    }
}

// complex * quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Quaternion<T>> for Complex<T> {
    type Output = Quaternion<T>;
    fn mul(self,other: Quaternion<T>) -> Self::Output {
        Quaternion {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
            j: self.r * other.j - self.i * other.k,
            k: self.r * other.k + self.i * other.j,
        }
    }
}

// quaternion * complex
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Complex<T>> for Quaternion<T> {
    type Output = Self;
    fn mul(self,other: Complex<T>) -> Self::Output {
        Quaternion {
            r: self.r * other.r - self.i * other.i,
            i: self.r * other.i + self.i * other.r,
            j: self.j * other.r + self.k * other.i,
            k: self.j * other.i + self.k * other.r,
        }
    }
}

// quaternion * quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Quaternion<T>> for Quaternion<T> {
    type Output = Self;
    fn mul(self,other: Self) -> Self::Output {
        Quaternion {
            r: self.r * other.r - self.i * other.i - self.j * other.j - self.k * other.k,
            i: self.r * other.i + self.i * other.r + self.j * other.k - self.k * other.j,
            j: self.r * other.j - self.i * other.k + self.j * other.r + self.k * other.i,
            k: self.r * other.k + self.i * other.j - self.j * other.i + self.k * other.r,
        }
    }
}

// quaternion * vector
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> Mul<Vec3<T>> for Quaternion<T> {
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

// quaternion *= scalar
impl<T: Copy + MulAssign> MulAssign<T> for Quaternion<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.i *= other;
        self.j *= other;
        self.k *= other;
    }
}

// quaternion *= complex
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> MulAssign<Complex<T>> for Quaternion<T> {
    fn mul_assign(&mut self,other: Complex<T>) {
        let r = self.r * other.r - self.i * other.i;
        let i = self.i * other.r + self.r * other.i;
        let j = self.j * other.r + self.k * other.i;
        let k = self.k * other.r - self.j * other.i;
        self.r = r;
        self.i = i;
        self.j = j;
        self.k = k;
    }
}

// quaternion *= quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T>> MulAssign<Quaternion<T>> for Quaternion<T> {
    fn mul_assign(&mut self,other: Quaternion<T>) {
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

// quaternion / scalar
impl<T: Copy + Div<Output=T>> Div<T> for Quaternion<T> {
    type Output = Self;
    fn div(self,other: T) -> Self::Output {
        Quaternion {
            r: self.r / other,
            i: self.i / other,
            j: self.j / other,
            k: self.k / other,
        }
    }
}

// complex / quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T> + Neg<Output=T>> Div<Quaternion<T>> for Complex<T> {
    type Output = Quaternion<T>;
    fn div(self,other: Quaternion<T>) -> Self::Output {
        let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
        Quaternion {
            r: (self.r * other.r + self.i * other.i) / f,
            i: (self.i * other.r - self.r * other.i) / f,
            j: (self.i * other.k - self.r * other.j) / f,
            k: (-self.r * other.k - self.i * other.j) / f,
        }
    }
}

// quaternion / complex
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> Div<Complex<T>> for Quaternion<T> {
    type Output = Self;
    fn div(self,other: Complex<T>) -> Self::Output {
        let f = other.r * other.r + other.i * other.i;
        Quaternion {
            r: (self.r * other.r + self.i * other.i) / f,
            i: (self.i * other.r - self.r * other.i) / f,
            j: (self.j * other.r - self.k * other.i) / f,
            k: (self.k * other.r + self.j * other.i) / f,
        }
    }
}

// quaternion / quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> Div<Quaternion<T>> for Quaternion<T> {
    type Output = Self;
    fn div(self,other: Self) -> Self::Output {
        let f = other.r * other.r + other.i * other.i + other.j * other.j + other.k * other.k;
        Quaternion {
            r: (self.r * other.r + self.i * other.i + self.j * other.j + self.k * other.k) / f,
            i: (self.i * other.r - self.j * other.k + self.k * other.j - self.r * other.i) / f,
            j: (self.j * other.r - self.k * other.i - self.r * other.j + self.i * other.k) / f,
            k: (self.k * other.r - self.r * other.k - self.i * other.j + self.j * other.i) / f,
        }
    }
}

// quaternion /= scalar
impl<T: Copy + DivAssign> DivAssign<T> for Quaternion<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.i /= other;
        self.j /= other;
        self.k /= other;
    }
}

// quaternion /= complex
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> DivAssign<Complex<T>> for Quaternion<T> {
    fn div_assign(&mut self,other: Complex<T>) {
        let f = other.r * other.r + other.i * other.i;
        let r = (self.r * other.r + self.i * other.i) / f;
        let i = (self.i * other.r - self.r * other.i) / f;
        let j = (self.j * other.r - self.k * other.i) / f;
        let k = (self.k * other.r + self.j * other.i) / f;
        self.r = r;
        self.i = i;
        self.j = j;
        self.k = k;
    }
}

// quaternion /= quaternion
impl<T: Copy + Add<Output=T> + Sub<Output=T> + Mul<Output=T> + Div<Output=T>> DivAssign<Quaternion<T>> for Quaternion<T> {
    fn div_assign(&mut self,other: Self) {
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

// -quaternion
impl<T: Neg<Output=T>> Neg for Quaternion<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Quaternion {
            r: -self.r,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        }
    }
}

#[allow(non_camel_case_types)]
pub type f32q = Quaternion<f32>;
#[allow(non_camel_case_types)]
pub type f64q = Quaternion<f64>;
