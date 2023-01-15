use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
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

/// 2D vector.
#[derive(Copy,Clone,Debug)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

/// Create vector from splatting one value.
impl<T: Copy> Vec2<T> {
    pub fn splat(value: T) -> Self {
        Vec2 { x: value,y: value, }
    }
}

/// The dot-product.
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Vec2<T> {
    pub fn dot(self,other: &Self) -> T {
        self.x * other.x + self.y * other.y
    }
}

/// The norm/length of the vector.
impl<T: Copy + Real + Mul<T,Output=T> + Add<T,Output=T>> Vec2<T> {
    pub fn norm(&self) -> T {
        self.dot(&self).sqrt()
    }
}

/// Normalize the vector (scale so norm becomes 1).
impl<T: Copy + Real + Mul<T,Output=T> + Add<T,Output=T> + Zero + PartialEq<T> + DivAssign<T>> Vec2<T> {
    pub fn normalize(&mut self) {
        let d = self.norm();
        if d != T::ZERO {
            self.x /= d;
            self.y /= d;
        }
    }
}

/// Component-wise scaling of the vector.
impl<T: Copy + Mul<T,Output=T>> Vec2<T> {
    pub fn scale(&self,other: &Self) -> Self {
        Vec2 {
            x: self.x * other.x,
            y: self.y * other.y,
        }
    }
}

/// Partial equality operators `==` and `!=`.
impl<T: PartialEq> PartialEq for Vec2<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y)
    }
}

/// Prettyprint the vector.
impl<T: Display> Display for Vec2<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{})",self.x,self.y)
    }
}

/// The Zero trait, adds ZERO constant to the vector.
impl<T: Zero> Zero for Vec2<T> { const ZERO: Self = Vec2 { x: T::ZERO,y: T::ZERO, }; }

/// The addition operator `+`.
impl<T: Add<T,Output=T>> Add<Vec2<T>> for Vec2<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Vec2 { x: self.x + other.x,y: self.y + other.y, }
    }
}

/// The addition-assignment operator `+=`.
impl<T: AddAssign<T>> AddAssign<Vec2<T>> for Vec2<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// The subtraction operator `-`.
impl<T: Sub<T,Output=T>> Sub<Vec2<T>> for Vec2<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Vec2 { x: self.x - other.x,y: self.y - other.y, }
    }
}

/// The subtraction-assignment operator `-=`.
impl<T: SubAssign<T>> SubAssign<Vec2<T>> for Vec2<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

macro_rules! scalar_vec2_mul {
    ($($t:ty)+) => {
        $(
            /// Scalar-vector multiplication operator `*`.
            impl Mul<Vec2<$t>> for $t {
                type Output = Vec2<$t>;
                fn mul(self,other: Vec2<$t>) -> Vec2<$t> {
                    Vec2 { x: self * other.x,y: self * other.y, }
                }
            }
        )+
    }
}

scalar_vec2_mul!(usize u8 u16 u32 u64 u128 isize i8 i16 i32 i64 i128 f32 f64);

/// Vector-scalar multiplication operator `*`.
impl<T: Copy + Mul<T,Output=T>> Mul<T> for Vec2<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Vec2 { x: self.x * other,y: self.y * other, }
    }
}

/// Vector-scalar multiply-assignment operator `*=`.
impl<T: Copy + MulAssign<T>> MulAssign<T> for Vec2<T> {
    fn mul_assign(&mut self,other: T) {
        self.x *= other;
        self.y *= other;
    }
}

/// Vector-scalar division operator `/`.
impl<T: Copy + Div<T,Output=T>> Div<T> for Vec2<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Vec2 { x: self.x / other,y: self.y / other, }
    }
}

/// Vector-scalar divide-assignment operator `/=`.
impl<T: Copy + DivAssign<T>> DivAssign<T> for Vec2<T> {
    fn div_assign(&mut self,other: T) {
        self.x /= other;
        self.y /= other;
    }
}

/// Negation operator `-`.
impl<T: Neg<Output=T>> Neg for Vec2<T> {
    type Output = Vec2<T>;
    fn neg(self) -> Self::Output {
        Vec2 { x: -self.x,y: -self.y, }
    }
}

impl Vec2<bool> {
    /// select vector components from boolean choice.
    pub fn select<T>(&self,a: Vec2<T>,b: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: if self.x { b.x } else { a.x },
            y: if self.y { b.y } else { a.y },
        }
    }

    /// True if all components are true.
    pub fn all(&self) -> bool {
        self.x && self.y
    }

    /// True if any component is true.
    pub fn any(&self) -> bool {
        self.x || self.y
    }

    /// Boolean inverse.
    pub fn not(&self) -> Self {
        Vec2 {
            x: !self.x,
            y: !self.y,
        }
    }
}

impl<T: Signed> Vec2<T> {
    pub fn abs(self) -> Self {
        Vec2 {
            x: self.x.abs(),
            y: self.y.abs(),
        }
    }
}

impl<T: Signed> Vec2<T> {
    pub fn signum(self) -> Self {
        Vec2 {
            x: self.x.signum(),
            y: self.y.signum(),
        }
    }
}

impl<T> Vec2<T> {
    pub fn min(self,other: Self) -> Self {
        Vec2 {
            x: self.x.min(other.x),
            y: self.y.min(other.y),
        }
    }
}

impl<T> Vec2<T> {
    pub fn max(self,other: Self) -> Self {
        Vec2 {
            x: self.x.max(other.x),
            y: self.y.max(other.y),
        }
    }
}

impl<T> Vec2<T> {
    pub fn clamp(self,min: Self,max: Self) -> Self {
        Vec2 {
            x: self.x.clamp(min.x,max.x),
            y: self.y.clamp(min.y,max.y),
        }
    }
}

impl<T> Vec2<T> {
    pub fn sclamp(self,min: T,max: T) -> Self {
        Vec2 {
            x: self.x.clamp(min,max),
            y: self.y.clamp(min,max),
        }
    }
}

macro_rules! vec2_float {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {
                pub fn to_radians(self) -> Self {
                    Vec2 {
                        x: self.x.to_radians(),
                        y: self.y.to_radians(),
                    }
                }

                pub fn to_degrees(self) -> Self {
                    Vec2 {
                        x: self.x.to_degrees(),
                        y: self.y.to_degrees(),
                    }
                }

                pub fn sin(self) -> Self {
                    Vec2 {
                        x: self.x.sin(),
                        y: self.y.sin(),
                    }
                }

                pub fn cos(self) -> Self {
                    Vec2 {
                        x: self.x.cos(),
                        y: self.y.cos(),
                    }
                }

                pub fn tan(self) -> Self {
                    Vec2 {
                        x: self.x.tan(),
                        y: self.y.tan(),
                    }
                }

                pub fn sinh(self) -> Self {
                    Vec2 {
                        x: self.x.sinh(),
                        y: self.y.sinh(),
                    }
                }

                pub fn cosh(self) -> Self {
                    Vec2 {
                        x: self.x.cosh(),
                        y: self.y.cosh(),
                    }
                }

                pub fn tanh(self) -> Self {
                    Vec2 {
                        x: self.x.tanh(),
                        y: self.y.tanh(),
                    }
                }

                pub fn asin(self) -> Self {
                    Vec2 {
                        x: self.x.asin(),
                        y: self.y.asin(),
                    }
                }

                pub fn acos(self) -> Self {
                    Vec2 {
                        x: self.x.acos(),
                        y: self.y.acos(),
                    }
                }

                pub fn atan(self) -> Self {
                    Vec2 {
                        x: self.x.atan(),
                        y: self.y.atan(),
                    }
                }
                
                pub fn atan2(self,other: Self) -> Self {
                    Vec2 {
                        x: self.x.atan2(other.x),
                        y: self.y.atan2(other.y),
                    }
                }

                pub fn asinh(self) -> Self {
                    Vec2 {
                        x: self.x.asinh(),
                        y: self.y.asinh(),
                    }
                }

                pub fn acosh(self) -> Self {
                    Vec2 {
                        x: self.x.acosh(),
                        y: self.y.acosh(),
                    }
                }

                pub fn atanh(self) -> Self {
                    Vec2 {
                        x: self.x.atanh(),
                        y: self.y.atanh(),
                    }
                }

                pub fn powf(self,other: Self) -> Self {
                    Vec2 {
                        x: self.x.powf(other.x),
                        y: self.y.powf(other.y),
                    }
                }

                pub fn spowf(self,other: $t) -> Self {
                    Vec2 {
                        x: self.x.powf(other),
                        y: self.y.powf(other),
                    }
                }

                pub fn exp(self) -> Self {
                    Vec2 {
                        x: self.x.exp(),
                        y: self.y.exp(),
                    }
                }

                pub fn ln(self) -> Self {
                    Vec2 {
                        x: self.x.ln(),
                        y: self.y.ln(),
                    }
                }

                pub fn exp2(self) -> Self {
                    Vec2 {
                        x: self.x.exp2(),
                        y: self.y.exp2(),
                    }
                }

                pub fn log2(self) -> Self {
                    Vec2 {
                        x: self.x.log2(),
                        y: self.y.log2(),
                    }
                }

                pub fn sqrt(self) -> Self {
                    Vec2 {
                        x: self.x.sqrt(),
                        y: self.y.sqrt(),
                    }
                }

                pub fn rem_euclid(self,other: Self) -> Self {
                    Vec2 {
                        x: self.x.rem_euclid(other.x),
                        y: self.y.rem_euclid(other.y),
                    }
                }

                /*
                pub fn invsqrt(self) -> Self {
                    Vec2 {
                        x: self.x.invsqrt(),
                        y: self.y.invsqrt(),
                    }
                }

                pub fn modf(self) -> (Self,Self) {
                    let x = self.x.modf();
                    let y = self.y.modf();
                    (
                        Vec2 { x: x.0,y: y.0, },
                        Vec2 { x: x.1,y: y.1, },
                    )
                }

                pub fn mix(self,other: Self,a: Self) -> Self {
                    Vec2 {
                        x: self.x.mix(other.x,a.x),
                        y: self.y.mix(other.y,a.y),
                    }
                }

                pub fn smix(self,other: Self,a: $t) -> Self {
                    Vec2 {
                        x: self.x.mix(other.x,a),
                        y: self.y.mix(other.y,a),
                    }
                }

                pub fn step(self,edge: Self) -> Self {
                    Vec2 {
                        x: self.x.step(edge.x),
                        y: self.y.step(edge.y),
                    }
                }

                pub fn sstep(self,edge: $t) -> Self {
                    Vec2 {
                        x: self.x.step(edge),
                        y: self.y.step(edge),
                    }
                }

                pub fn smoothstep(self,edge0: Self,edge1: Self) -> Self {
                    Vec2 {
                        x: self.x.smoothstep(edge0.x,edge1.x),
                        y: self.y.smoothstep(edge0.y,edge1.y),
                    }
                }

                pub fn ssmoothstep(self,edge0: $t,edge1: $t) -> Self {
                    Vec2 {
                        x: self.x.smoothstep(edge0,edge1),
                        y: self.y.smoothstep(edge0,edge1),
                    }
                }
                */

                pub fn is_nan(self) -> Vec2<bool> {
                    Vec2 { 
                        x: self.x.is_nan(),
                        y: self.y.is_nan(),
                    }
                }

                pub fn is_infinite(self) -> Vec2<bool> {
                    Vec2 { 
                        x: self.x.is_infinite(),
                        y: self.y.is_infinite(),
                    }
                }

                pub fn fma(self,b: Self,c: Self) -> Self {
                    Vec2 {
                        x: self.x * b.x + c.x,
                        y: self.y * b.y + c.y,
                    }
                }

                //pub fn dot(self,other: Self) -> $t {
                //    self.x * other.x + self.y * other.y
                //}

                pub fn length(&self) -> $t {
                    (self.x * self.x + self.y * self.y).sqrt()
                }

                pub fn distance(self,other: Self) -> $t {
                    (other - self).length()
                }

                //pub fn normalize(self) -> Self {
                //    let d = self.length();
                //    if d != 0 as $t {
                //        self / d
                //    }
                //    else {
                //        self
                //    }
                //}

                pub fn faceforward(self,n: Self,i: Self) -> Self {
                    if n.dot(&i) < 0.0 { self } else { -self }
                }

                pub fn reflect(self,n: Self) -> Self {
                    self - 2.0 * n.dot(&self) * n
                }

                pub fn refract(self,n: Self,eta: $t) -> Self {
                    let nds = n.dot(&self);
                    let k = 1.0 - eta * eta * (1.0 - nds * nds);
                    if k >= 0.0 { eta * self - (eta * nds + k.sqrt()) * n } else { Self::ZERO }
                }

                pub fn outer2(self,other: Vec2<$t>) -> Mat2x2<$t> {
                    Mat2x2 {
                        x: Vec2 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                        },
                        y: Vec2 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                        },
                    }
                }

                pub fn outer3(self,other: Vec3<$t>) -> Mat3x2<$t> {
                    Mat3x2 {
                        x: Vec2 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                        },
                        y: Vec2 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                        },
                        z: Vec2 {
                            x: self.x * other.z,
                            y: self.y * other.z,
                        },
                    }                    
                }

                pub fn outer4(self,other: Vec4<$t>) -> Mat4x2<$t> {
                    Mat4x2 {
                        x: Vec2 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                        },
                        y: Vec2 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                        },
                        z: Vec2 {
                            x: self.x * other.z,
                            y: self.y * other.z,
                        },
                        w: Vec2 {
                            x: self.x * other.w,
                            y: self.y * other.w,
                        },
                    }
                }

                pub fn less_than(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x < other.x,
                        y: self.y < other.y,
                    }
                }

                pub fn less_than_equal(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x <= other.x,
                        y: self.y <= other.y,
                    }
                }

                pub fn greater_than(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x > other.x,
                        y: self.y > other.y,
                    }
                }

                pub fn greater_than_equal(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x >= other.x,
                        y: self.y >= other.y,
                    }
                }

                pub fn equal(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x == other.x,
                        y: self.y == other.y,
                    }
                }

                pub fn not_equal(self,other: Self) -> Vec2<bool> {
                    Vec2 {
                        x: self.x != other.x,
                        y: self.y != other.y,
                    }
                }
            }
        )+
    }
}

vec2_float!(f32 f64);
