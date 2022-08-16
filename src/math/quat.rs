use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Debug,
            Formatter,
            Result
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
            Neg
        },
    },
};

/// Quaternion.
#[derive(Copy,Clone,Debug)]
pub struct Quat<T> {
    pub r: T,
    pub i: T,
    pub j: T,
    pub k: T,
}

impl<T> Quat<T> {
    pub fn new(r: T,i: T,j: T,k: T) -> Self {
        Quat {
            r: r,
            i: i,
            j: j,
            k: k,
        }
    }
}

impl<T: PartialEq> PartialEq for Quat<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) &&
        (self.i == other.i) &&
        (self.j == other.j) &&
        (self.k == other.k)
    }
}

impl<T: Display + Zero + PartialOrd> Display for Quat<T> {
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

impl<T: Add<T,Output=T>> Add<Quat<T>> for Quat<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Quat {
            r: self.r + other.r,
            i: self.i + other.i,
            j: self.j + other.j,
            k: self.k + other.k,
        }
    }
}

impl<T: AddAssign<T>> AddAssign<Quat<T>> for Quat<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.i += other.i;
        self.j += other.j;
        self.k += other.k;
    }
}

impl<T: Sub<T,Output=T>> Sub<Quat<T>> for Quat<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Quat {
            r: self.r - other.r,
            i: self.i - other.i,
            j: self.j - other.j,
            k: self.k - other.k,
        }
    }
}

impl<T: SubAssign<T>> SubAssign<Quat<T>> for Quat<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.i -= other.i;
        self.j -= other.j;
        self.k -= other.k;
    }
}

macro_rules! scalar_quat_mul {
    ($t:ty) => {
        impl Mul<Quat<$t>> for $t {
            type Output = Quat<$t>;
            fn mul(self,other: Quat<$t>) -> Quat<$t> {
                Quat {
                    r: self * other.r,
                    i: self * other.i,
                    j: self * other.j,
                    k: self * other.k,
                }
            }
        }        
    }
}

scalar_quat_mul!(f32);
scalar_quat_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Quat<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Quat {
            r: self.r * other,
            i: self.i * other,
            j: self.j * other,
            k: self.k * other,
        }
    }
}

// TODO: Quat * Quat

impl<T: Copy + MulAssign<T>> MulAssign<T> for Quat<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.i *= other;
        self.j *= other;
        self.k *= other;
    }
} 

// TODO: Quat *= Quat

impl<T: Copy + Div<T,Output=T>> Div<T> for Quat<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Quat {
            r: self.r / other,
            i: self.i / other,
            j: self.j / other,
            k: self.k / other,
        }
    }
}

// TODO: Quat / Quat

impl<T: Copy + DivAssign<T>> DivAssign<T> for Quat<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.i /= other;
        self.j /= other;
        self.k /= other;
    }
}

// TODO: Quat /= Quat

impl<T: Neg<Output=T>> Neg for Quat<T> {
    type Output = Quat<T>;
    fn neg(self) -> Quat<T> {
        Quat {
            r: -self.r,
            i: -self.i,
            j: -self.j,
            k: -self.k,
        }
    }
}

#[allow(non_camel_case_types)]
pub type f32q = Quat<f32>;
#[allow(non_camel_case_types)]
pub type f64q = Quat<f64>;