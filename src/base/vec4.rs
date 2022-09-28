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
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T: Copy> Vec4<T> {
    pub fn new(x: T,y: T,z: T,w: T) -> Self {
        Vec4 { x: x,y: y,z: z,w: w, }
    }
}

impl<T: Copy> From<[T; 4]> for Vec4<T> {
    fn from(array: [T; 4]) -> Self {
        Vec4 { x: array[0],y: array[1],z: array[2],w: array[3], }
    }
}

impl<T: Copy> From<&[T; 4]> for Vec4<T> {
    fn from(slice: &[T; 4]) -> Self {
        Vec4 { x: slice[0],y: slice[1],z: slice[2],w: slice[3], }
    }
}

impl<T: PartialEq> PartialEq for Vec4<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w)
    }
}

impl<T: Display> Display for Vec4<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{},{})",self.x,self.y,self.z,self.w)
    }
}

impl<T: Zero> Zero for Vec4<T> { const ZERO: Self = Vec4 { x: T::ZERO,y: T::ZERO,z: T::ZERO,w: T::ZERO, }; }

impl<T: Add<T,Output=T>> Add<Vec4<T>> for Vec4<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Vec4 { x: self.x + other.x,y: self.y + other.y,z: self.z + other.z,w: self.w + other.w, }
    }
}

impl<T: AddAssign<T>> AddAssign<Vec4<T>> for Vec4<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<T: Sub<T,Output=T>> Sub<Vec4<T>> for Vec4<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Vec4 { x: self.x - other.x,y: self.y - other.y,z: self.z - other.z,w: self.w - other.w, }
    }
}

impl<T: SubAssign<T>> SubAssign<Vec4<T>> for Vec4<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

macro_rules! scalar_vec4_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Vec4<$t>> for $t {
                type Output = Vec4<$t>;
                fn mul(self,other: Vec4<$t>) -> Vec4<$t> {
                    Vec4 { x: self * other.x,y: self * other.y,z: self * other.z,w: self * other.w, }
                }
            }
        )+
    }
}

scalar_vec4_mul!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Vec4<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Vec4 { x: self.x * other,y: self.y * other,z: self.z * other,w: self.w * other, }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Vec4<T> {
    fn mul_assign(&mut self,other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Vec4<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Vec4 { x: self.x / other,y: self.y / other,z: self.z / other,w: self.w / other, }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for Vec4<T> {
    fn div_assign(&mut self,other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

macro_rules! vec4_signed {
    ($($t:ty)+) => {
        $(
            impl Neg for Vec4<$t> {
                type Output = Vec4<$t>;
                fn neg(self) -> Self::Output {
                    Vec4 { x: -self.x,y: -self.y,z: -self.z,w: -self.w, }
                }
            }
        )+
    }
}

vec4_signed!(i8 i16 i32 i64 i128 isize f32 f64);

macro_rules! vec4_float {
    ($($t:ty)+) => {
        $(
            impl Vec4<$t> {
                pub fn dot(a: Self,b: Self) -> $t {
                    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
                }

                pub fn abs(&self) -> $t {
                    (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
                }

                pub fn normalize(&self) -> Self {
                    let d = self.abs();
                    if d != 0 as $t {
                        *self / d
                    }
                    else {
                        *self
                    }
                }
            }
        )+
    }
}

vec4_float!(f32 f64);
