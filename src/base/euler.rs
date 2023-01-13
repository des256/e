use {
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

#[derive(Copy,Clone,Debug)]
pub struct Euler<T> {
    pub y: T,
    pub p: T,
    pub r: T,
}

macro_rules! euler_impl {
    ($($t:ty)*) => ($(

        impl Euler<$t> {
        }

        impl Display for Euler<$t> {
            fn fmt(&self,f: &mut Formatter) -> Result {
                write!(f,"(y:{},p:{},r:{})",self.y,self.p,self.r)
            }
        }

        // euler == euler
        impl PartialEq<Euler<$t>> for Euler<$t> {
            fn eq(&self,other: &Euler<$t>) -> bool {
                (self.y == other.y) &&
                (self.p == other.p) &&
                (self.r == other.r)
            }
        }

        // euler + euler
        impl Add<Euler<$t>> for Euler<$t> {
            type Output = Euler<$t>;
            fn add(self,other: Euler<$t>) -> Euler<$t> {
                Euler {
                    y: self.y + other.y,
                    p: self.p + other.p,
                    r: self.r + other.r,
                }
            }
        }

        // euler += euler
        impl AddAssign<Euler<$t>> for Euler<$t> {
            fn add_assign(&mut self,other: Euler<$t>) {
                self.y += other.y;
                self.p += other.p;
                self.r += other.r;
            }
        }

        // euler - euler
        impl Sub<Euler<$t>> for Euler<$t> {
            type Output = Euler<$t>;
            fn sub(self,other: Euler<$t>) -> Euler<$t> {
                Euler {
                    y: self.y - other.y,
                    p: self.p - other.p,
                    r: self.r - other.r,
                }
            }
        }

        // euler -= euler
        impl SubAssign<Euler<$t>> for Euler<$t> {
            fn sub_assign(&mut self,other: Euler<$t>) {
                self.y -= other.y;
                self.p -= other.p;
                self.r -= other.r;
            }
        }

        // scalar * euler
        impl Mul<Euler<$t>> for $t {
            type Output = Euler<$t>;
            fn mul(self,other: Euler<$t>) -> Euler<$t> {
                Euler {
                    y: self * other.y,
                    p: self * other.p,
                    r: self * other.r,
                }
            }
        }

        // euler * scalar
        impl Mul<$t> for Euler<$t> {
            type Output = Euler<$t>;
            fn mul(self,other: $t) -> Euler<$t> {
                Euler {
                    y: self.y * other,
                    p: self.p * other,
                    r: self.r * other,
                }
            }
        }

        // euler *= scalar
        impl MulAssign<$t> for Euler<$t> {
            fn mul_assign(&mut self,other: $t) {
                self.y *= other;
                self.p *= other;
                self.r *= other;
            }
        }

        // euler *= euler
        impl MulAssign<Euler<$t>> for Euler<$t> {
            fn mul_assign(&mut self,other: Euler<$t>) {
                self.y *= other.y;
                self.p *= other.p;
                self.r *= other.r;
            }
        }

        // euler / scalar
        impl Div<$t> for Euler<$t> {
            type Output = Euler<$t>;
            fn div(self,other: $t) -> Euler<$t> {
                Euler {
                    y: self.y / other,
                    p: self.p / other,
                    r: self.r / other,
                }
            }
        }

        // euler /= scalar
        impl DivAssign<$t> for Euler<$t> {
            fn div_assign(&mut self,other: $t) {
                self.y /= other;
                self.p /= other;
                self.r /= other;
            }
        }

        // -euler
        impl Neg for Euler<$t> {
            type Output = Self;
            fn neg(self) -> Euler<$t> {
                Euler {
                    y: -self.y,
                    p: -self.p,
                    r: -self.r,
                }
            }
        }
    )*)
}

euler_impl! { f32 f64 }

#[allow(non_camel_case_types)]
pub type f32e = Euler<f32>;
#[allow(non_camel_case_types)]
pub type f64e = Euler<f64>;
