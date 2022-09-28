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

macro_rules! vec3_float {
    ($($t:ty)+) => {
        $(
            impl Vec3<$t> {
                pub fn dot(a: Self,b: Self) -> $t {
                    a.x * b.x + a.y * b.y + a.z * b.z
                }

                pub fn cross(a: Self,b: Self) -> Self {
                    Vec3 {
                        x: a.y * b.z - a.z * b.y,
                        y: a.z * b.x - a.x * b.z,
                        z: a.x * b.y - a.y * b.x,
                    }
                }

                pub fn abs(&self) -> $t {
                    (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
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

vec3_float!(f32 f64);
