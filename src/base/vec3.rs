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

#[derive(Copy,Clone,Debug)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T: Copy> Vec3<T> {
    pub fn new(x: T,y: T,z: T) -> Self {
        Vec3 { x: x,y: y,z: z, }
    }
}

impl<T: Copy> From<[T; 3]> for Vec3<T> {
    fn from(array: [T; 3]) -> Self {
        Vec3 { x: array[0],y: array[1],z: array[2], }
    }
}

impl<T: Copy> From<&[T; 3]> for Vec3<T> {
    fn from(slice: &[T; 3]) -> Self {
        Vec3 { x: slice[0],y: slice[1],z: slice[2], }
    }
}

impl<T: PartialEq> PartialEq for Vec3<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z)
    }
}

impl<T: Display> Display for Vec3<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{})",self.x,self.y,self.z)
    }
}

impl<T: Zero> Zero for Vec3<T> { const ZERO: Self = Vec3 { x: T::ZERO,y: T::ZERO,z: T::ZERO, }; }

impl<T: Add<T,Output=T>> Add<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Vec3 { x: self.x + other.x,y: self.y + other.y,z: self.z + other.z, }
    }
}

impl<T: AddAssign<T>> AddAssign<Vec3<T>> for Vec3<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

impl<T: Sub<T,Output=T>> Sub<Vec3<T>> for Vec3<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Vec3 { x: self.x - other.x,y: self.y - other.y,z: self.z - other.z, }
    }
}

impl<T: SubAssign<T>> SubAssign<Vec3<T>> for Vec3<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

macro_rules! scalar_vec3_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Vec3<$t>> for $t {
                type Output = Vec3<$t>;
                fn mul(self,other: Vec3<$t>) -> Vec3<$t> {
                    Vec3 { x: self * other.x,y: self * other.y,z: self * other.z, }
                }
            }
        )+
    }
}

scalar_vec3_mul!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Vec3<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Vec3 { x: self.x * other,y: self.y * other,z: self.z * other, }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Vec3<T> {
    fn mul_assign(&mut self,other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Vec3<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Vec3 { x: self.x / other,y: self.y / other,z: self.z / other, }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for Vec3<T> {
    fn div_assign(&mut self,other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

macro_rules! vec3_signed {
    ($($t:ty)+) => {
        $(
            impl Neg for Vec3<$t> {
                type Output = Vec3<$t>;
                fn neg(self) -> Self::Output {
                    Vec3 { x: -self.x,y: -self.y,z: -self.z, }
                }
            }
        )+
    }
}

vec3_signed!(i8 i16 i32 i64 i128 isize f32 f64);

impl Vec3<bool> {
    pub fn select<T>(&self,a: Vec3<T>,b: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: if self.x { b.x } else { a.x },
            y: if self.y { b.y } else { a.y },
            z: if self.z { b.z } else { a.z },
        }
    }

    pub fn all(&self) -> bool {
        self.x && self.y && self.z
    }

    pub fn any(&self) -> bool {
        self.x || self.y || self.z
    }

    pub fn not(&self) -> Self {
        Vec3 {
            x: !self.x,
            y: !self.y,
            z: !self.z,
        }
    }
}

macro_rules! vec3_int_float {
    ($($t:ty)+) => {
        $(
            impl Vec3<$t> {
                pub fn abs(self) -> Self {
                    Vec3 {
                        x: self.x.abs(),
                        y: self.y.abs(),
                        z: self.z.abs(),
                    }
                }

                pub fn signum(self) -> Self {
                    Vec3 {
                        x: self.x.signum(),
                        y: self.y.signum(),
                        z: self.z.signum(),
                    }
                }

                pub fn min(self,other: Self) -> Self {
                    Vec3 {
                        x: self.x.min(other.x),
                        y: self.y.min(other.y),
                        z: self.z.min(other.z),
                    }
                }

                pub fn max(self,other: Self) -> Self {
                    Vec3 {
                        x: self.x.max(other.x),
                        y: self.y.max(other.y),
                        z: self.z.max(other.z),
                    }
                }

                pub fn clamp(self,min: Self,max: Self) -> Self {
                    Vec3 {
                        x: self.x.clamp(min.x,max.x),
                        y: self.y.clamp(min.y,max.y),
                        z: self.z.clamp(min.z,max.z),
                    }
                }

                pub fn sclamp(self,min: $t,max: $t) -> Self {
                    Vec3 {
                        x: self.x.clamp(min,max),
                        y: self.y.clamp(min,max),
                        z: self.z.clamp(min,max),
                    }
                }
            }
        )+
    }
}

vec3_int_float!(i8 i16 i32 i64 i128 isize f32 f64);

macro_rules! vec3_float {
    ($($t:ty)+) => {
        $(
            impl Vec3<$t> {
                pub fn to_radians(self) -> Self {
                    Vec3 {
                        x: self.x.to_radians(),
                        y: self.y.to_radians(),
                        z: self.z.to_radians(),
                    }
                }

                pub fn to_degrees(self) -> Self {
                    Vec3 {
                        x: self.x.to_degrees(),
                        y: self.y.to_degrees(),
                        z: self.z.to_degrees(),
                    }
                }

                pub fn sin(self) -> Self {
                    Vec3 {
                        x: self.x.sin(),
                        y: self.y.sin(),
                        z: self.z.sin(),
                    }
                }

                pub fn cos(self) -> Self {
                    Vec3 {
                        x: self.x.cos(),
                        y: self.y.cos(),
                        z: self.z.cos(),
                    }
                }

                pub fn tan(self) -> Self {
                    Vec3 {
                        x: self.x.tan(),
                        y: self.y.tan(),
                        z: self.z.tan(),
                    }
                }

                pub fn sinh(self) -> Self {
                    Vec3 {
                        x: self.x.sinh(),
                        y: self.y.sinh(),
                        z: self.z.sinh(),
                    }
                }

                pub fn cosh(self) -> Self {
                    Vec3 {
                        x: self.x.cosh(),
                        y: self.y.cosh(),
                        z: self.z.cosh(),
                    }
                }

                pub fn tanh(self) -> Self {
                    Vec3 {
                        x: self.x.tanh(),
                        y: self.y.tanh(),
                        z: self.z.tanh(),
                    }
                }
                
                pub fn asin(self) -> Self {
                    Vec3 {
                        x: self.x.asin(),
                        y: self.y.asin(),
                        z: self.z.asin(),
                    }
                }

                pub fn acos(self) -> Self {
                    Vec3 {
                        x: self.x.acos(),
                        y: self.y.acos(),
                        z: self.z.acos(),
                    }
                }

                pub fn atan(self) -> Self {
                    Vec3 {
                        x: self.x.atan(),
                        y: self.y.atan(),
                        z: self.z.atan(),
                    }
                }
                
                pub fn atan2(self,other: Self) -> Self {
                    Vec3 {
                        x: self.x.atan2(other.x),
                        y: self.y.atan2(other.y),
                        z: self.z.atan2(other.z),
                    }
                }

                pub fn asinh(self) -> Self {
                    Vec3 {
                        x: self.x.asinh(),
                        y: self.y.asinh(),
                        z: self.z.asinh(),
                    }
                }

                pub fn acosh(self) -> Self {
                    Vec3 {
                        x: self.x.acosh(),
                        y: self.y.acosh(),
                        z: self.z.acosh(),
                    }
                }

                pub fn atanh(self) -> Self {
                    Vec3 {
                        x: self.x.atanh(),
                        y: self.y.atanh(),
                        z: self.z.atanh(),
                    }
                }

                pub fn powf(self,other: Self) -> Self {
                    Vec3 {
                        x: self.x.powf(other.x),
                        y: self.y.powf(other.y),
                        z: self.z.powf(other.z),
                    }
                }

                pub fn spowf(self,other: $t) -> Self {
                    Vec3 {
                        x: self.x.powf(other),
                        y: self.y.powf(other),
                        z: self.z.powf(other),
                    }
                }

                pub fn exp(self) -> Self {
                    Vec3 {
                        x: self.x.exp(),
                        y: self.y.exp(),
                        z: self.z.exp(),
                    }
                }

                pub fn ln(self) -> Self {
                    Vec3 {
                        x: self.x.ln(),
                        y: self.y.ln(),
                        z: self.z.ln(),
                    }
                }

                pub fn exp2(self) -> Self {
                    Vec3 {
                        x: self.x.exp2(),
                        y: self.y.exp2(),
                        z: self.z.exp2(),
                    }
                }

                pub fn log2(self) -> Self {
                    Vec3 {
                        x: self.x.log2(),
                        y: self.y.log2(),
                        z: self.z.log2(),
                    }
                }

                pub fn sqrt(self) -> Self {
                    Vec3 {
                        x: self.x.sqrt(),
                        y: self.y.sqrt(),
                        z: self.z.sqrt(),
                    }
                }

                pub fn rem_euclid(self,other: Self) -> Self {
                    Vec3 {
                        x: self.x.rem_euclid(other.x),
                        y: self.y.rem_euclid(other.y),
                        z: self.z.rem_euclid(other.z),
                    }
                }

                /*
                pub fn invsqrt(self) -> Self {
                    Vec3 {
                        x: self.x.invsqrt(),
                        y: self.y.invsqrt(),
                        z: self.z.invsqrt(),
                    }
                }

                pub fn modf(self) -> (Self,Self) {
                    let x = self.x.modf();
                    let y = self.y.modf();
                    let z = self.z.modf();
                    (
                        Vec3 { x: x.0,y: y.0,z: z.0, },
                        Vec3 { x: x.1,y: y.1,z: z.1, },
                    )
                }

                pub fn mix(self,other: Self,a: Self) -> Self {
                    Vec3 {
                        x: self.x.mix(other.x,a.x),
                        y: self.y.mix(other.y,a.y),
                        z: self.z.mix(other.z,a.z),
                    }
                }

                pub fn smix(self,other: Self,a: $t) -> Self {
                    Vec3 {
                        x: self.x.mix(other.x,a),
                        y: self.y.mix(other.y,a),
                        z: self.z.mix(other.z,a),
                    }
                }

                pub fn step(self,edge: Self) -> Self {
                    Vec3 {
                        x: self.x.step(edge.x),
                        y: self.y.step(edge.y),
                        z: self.z.step(edge.z),
                    }
                }

                pub fn sstep(self,edge: $t) -> Self {
                    Vec3 {
                        x: self.x.step(edge),
                        y: self.y.step(edge),
                        z: self.z.step(edge),
                    }
                }

                pub fn smoothstep(self,edge0: Self,edge1: Self) -> Self {
                    Vec3 {
                        x: self.x.smoothstep(edge0.x,edge1.x),
                        y: self.y.smoothstep(edge0.y,edge1.y),
                        z: self.z.smoothstep(edge0.z,edge1.z),
                    }
                }

                pub fn ssmoothstep(self,edge0: $t,edge1: $t) -> Self {
                    Vec3 {
                        x: self.x.smoothstep(edge0,edge1),
                        y: self.y.smoothstep(edge0,edge1),
                        z: self.z.smoothstep(edge0,edge1),
                    }
                }
                */
                
                pub fn is_nan(self) -> Vec3<bool> {
                    Vec3 { 
                        x: self.x.is_nan(),
                        y: self.y.is_nan(),
                        z: self.z.is_nan(),
                    }
                }

                pub fn is_infinite(self) -> Vec3<bool> {
                    Vec3 { 
                        x: self.x.is_infinite(),
                        y: self.y.is_infinite(),
                        z: self.z.is_infinite(),
                    }
                }

                pub fn fma(self,b: Self,c: Self) -> Self {
                    Vec3 {
                        x: self.x * b.x + c.x,
                        y: self.y * b.y + c.y,
                        z: self.z * b.z + c.z,
                    }
                }

                pub fn dot(self,other: Self) -> $t {
                    self.x * other.x + self.y * other.y + self.z * other.z
                }

                pub fn cross(self,other: Self) -> Self {
                    Vec3 {
                        x: self.y * other.z - self.z * other.y,
                        y: self.z * other.x - self.x * other.z,
                        z: self.x * other.y - self.y * other.x,
                    }
                }

                pub fn length(&self) -> $t {
                    (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
                }

                pub fn distance(self,other: Self) -> $t {
                    (other - self).length()
                }

                pub fn normalize(self) -> Self {
                    let d = self.length();
                    if d != 0 as $t {
                        self / d
                    }
                    else {
                        self
                    }
                }

                pub fn faceforward(self,n: Self,i: Self) -> Self {
                    if n.dot(i) < 0.0 { self } else { -self }
                }

                pub fn reflect(self,n: Self) -> Self {
                    self - 2.0 * n.dot(self) * n
                }

                pub fn refract(self,n: Self,eta: $t) -> Self {
                    let nds = n.dot(self);
                    let k = 1.0 - eta * eta * (1.0 - nds * nds);
                    if k >= 0.0 { eta * self - (eta * nds + k.sqrt()) * n } else { Self::ZERO }
                }

                pub fn outer2(self,other: Vec2<$t>) -> Mat2x3<$t> {
                    Mat2x3 {
                        x: Vec3 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                            z: self.z * other.x,
                        },
                        y: Vec3 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                            z: self.z * other.y,
                        },
                    }
                }

                pub fn outer3(self,other: Vec3<$t>) -> Mat3x3<$t> {
                    Mat3x3 {
                        x: Vec3 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                            z: self.z * other.x,
                        },
                        y: Vec3 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                            z: self.z * other.y,
                        },
                        z: Vec3 {
                            x: self.x * other.z,
                            y: self.y * other.z,
                            z: self.z * other.z,
                        },
                    }                    
                }

                pub fn outer4(self,other: Vec4<$t>) -> Mat4x3<$t> {
                    Mat4x3 {
                        x: Vec3 {
                            x: self.x * other.x,
                            y: self.y * other.x,
                            z: self.z * other.x,
                        },
                        y: Vec3 {
                            x: self.x * other.y,
                            y: self.y * other.y,
                            z: self.z * other.y,
                        },
                        z: Vec3 {
                            x: self.x * other.z,
                            y: self.y * other.z,
                            z: self.z * other.z,
                        },
                        w: Vec3 {
                            x: self.x * other.w,
                            y: self.y * other.w,
                            z: self.z * other.w,
                        },
                    }
                }

                pub fn less_than(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x < other.x,
                        y: self.y < other.y,
                        z: self.z < other.z,
                    }
                }

                pub fn less_than_equal(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x <= other.x,
                        y: self.y <= other.y,
                        z: self.z <= other.z,
                    }
                }

                pub fn greater_than(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x > other.x,
                        y: self.y > other.y,
                        z: self.z > other.z,
                    }
                }

                pub fn greater_than_equal(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x >= other.x,
                        y: self.y >= other.y,
                        z: self.z >= other.z,
                    }
                }

                pub fn equal(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x == other.x,
                        y: self.y == other.y,
                        z: self.z == other.z,
                    }
                }

                pub fn not_equal(self,other: Self) -> Vec3<bool> {
                    Vec3 {
                        x: self.x != other.x,
                        y: self.y != other.y,
                        z: self.z != other.z,
                    }
                }
            }
        )+
    }
}

vec3_float!(f32 f64);
