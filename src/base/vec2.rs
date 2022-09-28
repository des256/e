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

#[derive(Copy,Clone,Debug)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T: Copy> Vec2<T> {
    pub fn new(x: T,y: T) -> Self {
        Vec2 { x: x,y: y, }
    }
}

impl<T: Copy> From<[T; 2]> for Vec2<T> {
    fn from(array: [T; 2]) -> Self {
        Vec2 { x: array[0],y: array[1], }
    }
}

impl<T: Copy> From<&[T; 2]> for Vec2<T> {
    fn from(slice: &[T; 2]) -> Self {
        Vec2 { x: slice[0],y: slice[1], }
    }
}

impl<T: PartialEq> PartialEq for Vec2<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y)
    }
}

impl<T: Display> Display for Vec2<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{})",self.x,self.y)
    }
}

impl<T: Zero> Zero for Vec2<T> { const ZERO: Self = Vec2 { x: T::ZERO,y: T::ZERO, }; }

impl<T: Add<T,Output=T>> Add<Vec2<T>> for Vec2<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Vec2 { x: self.x + other.x,y: self.y + other.y, }
    }
}

impl<T: AddAssign<T>> AddAssign<Vec2<T>> for Vec2<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

impl<T: Sub<T,Output=T>> Sub<Vec2<T>> for Vec2<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Vec2 { x: self.x - other.x,y: self.y - other.y, }
    }
}

impl<T: SubAssign<T>> SubAssign<Vec2<T>> for Vec2<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

macro_rules! scalar_vec2_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Vec2<$t>> for $t {
                type Output = Vec2<$t>;
                fn mul(self,other: Vec2<$t>) -> Vec2<$t> {
                    Vec2 { x: self * other.x,y: self * other.y, }
                }
            }
        )+
    }
}

scalar_vec2_mul!(u8 u16 u32 u64 u128 usize i8 i16 i32 i64 i128 isize f32 f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Vec2<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Vec2 { x: self.x * other,y: self.y * other, }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Vec2<T> {
    fn mul_assign(&mut self,other: T) {
        self.x *= other;
        self.y *= other;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Vec2<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Vec2 { x: self.x / other,y: self.y / other, }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for Vec2<T> {
    fn div_assign(&mut self,other: T) {
        self.x /= other;
        self.y /= other;
    }
}

macro_rules! vec2_signed {
    ($($t:ty)+) => {
        $(
            impl Neg for Vec2<$t> {
                type Output = Vec2<$t>;
                fn neg(self) -> Self::Output {
                    Vec2 { x: -self.x,y: -self.y, }
                }
            }
        )+
    }
}

vec2_signed!(i8 i16 i32 i64 i128 isize f32 f64);

macro_rules! vec2_float {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {
                pub fn dot(a: Self,b: Self) -> $t {
                    a.x * b.x + a.y * b.y
                }

                pub fn abs(&self) -> $t {
                    (self.x * self.x + self.y * self.y).sqrt()
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

vec2_float!(f32 f64);
