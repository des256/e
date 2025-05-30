use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 2D vector of numbers.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Add<Output = T>,
{
    /// Unit vector in positive X-direction.
    pub const UNIT_X: Self = Vec2 {
        x: T::ONE,
        y: T::ZERO,
    };

    /// Unit vector in positive Y-direction.
    pub const UNIT_Y: Self = Vec2 {
        x: T::ZERO,
        y: T::ONE,
    };

    /// Calculate dot product.
    pub fn dot(self, other: &Vec2<T>) -> T {
        self.x * other.x + self.y * other.y
    }
}

impl<T> Display for Vec2<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({},{})", self.x, self.y)
    }
}

impl<T> Zero for Vec2<T>
where
    T: Zero,
{
    const ZERO: Vec2<T> = Vec2 {
        x: T::ZERO,
        y: T::ZERO,
    };
}

/// Vector + vector.
impl<T> Add<Vec2<T>> for Vec2<T>
where
    T: Add<Output = T>,
{
    type Output = Vec2<T>;
    fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

/// Vector += vector.
impl<T> AddAssign<Vec2<T>> for Vec2<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

/// Vector - vector.
impl<T> Sub<Vec2<T>> for Vec2<T>
where
    T: Sub<Output = T>,
{
    type Output = Vec2<T>;
    fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

/// Vector -= vector.
impl<T> SubAssign<Vec2<T>> for Vec2<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
    }
}

/// Vector * scalar.
impl<T> Mul<T> for Vec2<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Vec2<T>;
    fn mul(self, other: T) -> Self::Output {
        Vec2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

/// Vector *= scalar.
impl<T> MulAssign<T> for Vec2<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
    }
}

/// Vector / scalar.
impl<T> Div<T> for Vec2<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Vec2<T>;
    fn div(self, other: T) -> Self::Output {
        Vec2 {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

/// Vector /= scalar.
impl<T> DivAssign<T> for Vec2<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
    }
}

/// -Vector.
impl<T> Neg for Vec2<T>
where
    T: Neg<Output = T>,
{
    type Output = Vec2<T>;
    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

macro_rules! vec2_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * vector.
            impl Mul<Vec2<$t>> for $t {
                type Output = Vec2<$t>;
                fn mul(self,other: Vec2<$t>) -> Self::Output {
                    Vec2 {
                        x: self * other.x,
                        y: self * other.y,
                    }
                }
            }
        )+
    }
}

vec2_impl! { usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 f32 f64 }

// implementations where $t: Real
macro_rules! vec2_real_impl {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {

                /// Calculate vector length.
                pub fn length(&self) -> $t {
                    self.dot(&self).sqrt()
                }

                /// Normalize vector.
                pub fn normalize(&mut self) {
                    let d = self.length();
                    if d != <$t>::ZERO {
                        *self /= d;
                    }
                }
            }

            impl BitOr<Vec2<$t>> for Vec2<$t> {
                type Output = $t;
                fn bitor(self,other: Vec2<$t>) -> $t {
                    self.x * other.x + self.y * other.y
                }
            }
        )+
    }
}

vec2_real_impl! { f32 f64 }

// if `T as U` exists, `Vec2<U>::from(Vec2<T>)` should also exist
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! vec2_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Vec2<$t>> for Vec2<$u> {
                fn from(value: Vec2<$t>) -> Self { Vec2 { x: value.x as $u,y: value.y as $u, } }
            }
        )+
    }
}

vec2_from_impl! { (usize,isize) (usize,u8) (usize,i8) (usize,u16) (usize,i16) (usize,u32) (usize,i32) (usize,u64) (usize,i64) (usize,u128) (usize,i128) (usize,f32) (usize,f64) }
vec2_from_impl! { (isize,usize) (isize,u8) (isize,i8) (isize,u16) (isize,i16) (isize,u32) (isize,i32) (isize,u64) (isize,i64) (isize,u128) (isize,i128) (isize,f32) (isize,f64) }
vec2_from_impl! { (u8,usize) (u8,isize) (u8,i8) (u8,u16) (u8,i16) (u8,u32) (u8,i32) (u8,u64) (u8,i64) (u8,u128) (u8,i128) (u8,f32) (u8,f64) }
vec2_from_impl! { (i8,usize) (i8,isize) (i8,u8) (i8,u16) (i8,i16) (i8,u32) (i8,i32) (i8,u64) (i8,i64) (i8,u128) (i8,i128) (i8,f32) (i8,f64) }
vec2_from_impl! { (u16,usize) (u16,isize) (u16,u8) (u16,i8) (u16,i16) (u16,u32) (u16,i32) (u16,u64) (u16,i64) (u16,u128) (u16,i128) (u16,f32) (u16,f64) }
vec2_from_impl! { (i16,usize) (i16,isize) (i16,u8) (i16,i8) (i16,u16) (i16,u32) (i16,i32) (i16,u64) (i16,i64) (i16,u128) (i16,i128) (i16,f32) (i16,f64) }
vec2_from_impl! { (u32,usize) (u32,isize) (u32,u8) (u32,i8) (u32,u16) (u32,i16) (u32,i32) (u32,u64) (u32,i64) (u32,u128) (u32,i128) (u32,f32) (u32,f64) }
vec2_from_impl! { (i32,usize) (i32,isize) (i32,u8) (i32,i8) (i32,u16) (i32,i16) (i32,u32) (i32,u64) (i32,i64) (i32,u128) (i32,i128) (i32,f32) (i32,f64) }
vec2_from_impl! { (u64,usize) (u64,isize) (u64,u8) (u64,i8) (u64,u16) (u64,i16) (u64,u32) (u64,i32) (u64,i64) (u64,u128) (u64,i128) (u64,f32) (u64,f64) }
vec2_from_impl! { (i64,usize) (i64,isize) (i64,u8) (i64,i8) (i64,u16) (i64,i16) (i64,u32) (i64,i32) (i64,u64) (i64,u128) (i64,i128) (i64,f32) (i64,f64) }
vec2_from_impl! { (u128,usize) (u128,isize) (u128,u8) (u128,i8) (u128,u16) (u128,i16) (u128,u32) (u128,i32) (u128,u64) (u128,i64) (u128,i128) (u128,f32) (u128,f64) }
vec2_from_impl! { (i128,usize) (i128,isize) (i128,u8) (i128,i8) (i128,u16) (i128,i16) (i128,u32) (i128,i32) (i128,u64) (i128,i64) (i128,u128) (i128,f32) (i128,f64) }
vec2_from_impl! { (f32,usize) (f32,isize) (f32,u8) (f32,i8) (f32,u16) (f32,i16) (f32,u32) (f32,i32) (f32,u64) (f32,i64) (f32,u128) (f32,i128) (f32,f64) }
vec2_from_impl! { (f64,usize) (f64,isize) (f64,u8) (f64,i8) (f64,u16) (f64,i16) (f64,u32) (f64,i32) (f64,u64) (f64,i64) (f64,u128) (f64,i128) (f64,f32) }
