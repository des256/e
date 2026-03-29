use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{Add, AddAssign, BitOr, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    },
};

/// 4D vector, generic over the component type.
///
/// Supports arithmetic operators for vector addition, subtraction,
/// scalar multiplication/division, negation, and indexing by component.
///
/// For `f32`/`f64` vectors, additional methods are available: [`length`](Vec4::length),
/// [`normalized`](Vec4::normalized),
/// [`length_squared`](Vec4::length_squared), and the `|` operator as a dot product alias.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let v = vec4(1.0f32, 2.0, 3.0, 4.0);
/// assert_eq!(v.xyz(), vec3(1.0, 2.0, 3.0));
/// assert_eq!(v.xy(), vec2(1.0, 2.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Vec4<T> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
    /// Z component.
    pub z: T,
    /// W component.
    pub w: T,
}

/// Create a new 4D vector.
pub const fn vec4<T>(x: T, y: T, z: T, w: T) -> Vec4<T> {
    Vec4 { x, y, z, w }
}

impl<T: Copy> Vec4<T> {
    /// Project to [`Vec3`] by dropping the W component.
    pub fn xyz(self) -> Vec3<T> {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }

    /// Project to [`Vec2`] by dropping the Z and W components.
    pub fn xy(self) -> Vec2<T> {
        Vec2 { x: self.x, y: self.y }
    }
}

impl<T> Vec4<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Add<Output = T>,
{
    /// Unit vector in positive X-direction.
    pub const UNIT_X: Self = Vec4 {
        x: T::ONE,
        y: T::ZERO,
        z: T::ZERO,
        w: T::ZERO,
    };

    /// Unit vector in positive Y-direction.
    pub const UNIT_Y: Self = Vec4 {
        x: T::ZERO,
        y: T::ONE,
        z: T::ZERO,
        w: T::ZERO,
    };

    /// Unit vector in positive Z-direction.
    pub const UNIT_Z: Self = Vec4 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ONE,
        w: T::ZERO,
    };

    /// Unit vector in positive W-direction.
    pub const UNIT_W: Self = Vec4 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
        w: T::ONE,
    };

    /// Calculate dot product.
    pub fn dot(self, other: Vec4<T>) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
    }
}

impl<T> Index<usize> for Vec4<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            3 => &self.w,
            _ => panic!("Vec4 index out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Vec4<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            3 => &mut self.w,
            _ => panic!("Vec4 index out of range"),
        }
    }
}

impl<T> Display for Vec4<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({},{},{},{})", self.x, self.y, self.z, self.w)
    }
}

impl<T> Zero for Vec4<T>
where
    T: Zero,
{
    const ZERO: Vec4<T> = Vec4 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
        w: T::ZERO,
    };
}

/// Vector + vector.
impl<T> Add<Vec4<T>> for Vec4<T>
where
    T: Add<Output = T>,
{
    type Output = Vec4<T>;
    fn add(self, other: Self) -> Self {
        Vec4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

/// Vector += vector.
impl<T> AddAssign<Vec4<T>> for Vec4<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

/// Vector * scalar.
impl<T> Mul<T> for Vec4<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Vec4<T>;
    fn mul(self, other: T) -> Self::Output {
        Vec4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

/// Vector *= scalar.
impl<T> MulAssign<T> for Vec4<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

/// Vector / scalar.
impl<T> Div<T> for Vec4<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Vec4<T>;
    fn div(self, other: T) -> Self::Output {
        Vec4 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
            w: self.w / other,
        }
    }
}

/// Vector /= scalar.
impl<T> DivAssign<T> for Vec4<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

/// Vector - vector.
impl<T> Sub<Vec4<T>> for Vec4<T>
where
    T: Sub<Output = T>,
{
    type Output = Vec4<T>;
    fn sub(self, other: Self) -> Self {
        Vec4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

/// Vector -= vector.
impl<T> SubAssign<Vec4<T>> for Vec4<T>
where
    T: SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

/// -Vector.
impl<T> Neg for Vec4<T>
where
    T: Neg<Output = T>,
{
    type Output = Vec4<T>;
    fn neg(self) -> Self::Output {
        Vec4 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
            w: -self.w,
        }
    }
}

macro_rules! vec4_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * vector.
            impl Mul<Vec4<$t>> for $t {
                type Output = Vec4<$t>;
                fn mul(self,other: Vec4<$t>) -> Self::Output {
                    Vec4 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                        w: self * other.w,
                    }
                }
            }
        )+
    }
}

vec4_impl! { usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 F16 f32 f64 }

// implementations where $t: Real
macro_rules! vec4_real_impl {
    ($($t:ty)+) => {
        $(
            impl Vec4<$t> {

                /// Calculate squared vector length.
                pub fn length_squared(&self) -> $t {
                    self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w
                }

                /// Calculate vector length.
                pub fn length(&self) -> $t {
                    self.length_squared().sqrt()
                }

                /// Return a normalized copy of the vector.
                ///
                /// Returns the original vector unchanged if its length is zero.
                pub fn normalized(self) -> Self {
                    let d = self.length();
                    if d != <$t>::ZERO { self / d } else { self }
                }
            }

            /// Dot product via the `|` operator.
            ///
            /// Alias for [`dot`](Vec4::dot); matches the geometric algebra
            /// inner product notation used by [`MultiVec201`] and [`MultiVec301`].
            impl BitOr<Vec4<$t>> for Vec4<$t> {
                type Output = $t;
                fn bitor(self,other: Vec4<$t>) -> $t {
                    self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w
                }
            }
        )+
    }
}

vec4_real_impl! { F16 f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! vec4_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Vec4<$t>> for Vec4<$u> {
                fn from(value: Vec4<$t>) -> Self { Vec4 { x: value.x as $u,y: value.y as $u,z: value.z as $u,w: value.w as $u, } }
            }
        )+
    }
}

vec4_from_impl! { (u8,u16) (u8,u32) (u8,u64) (u8,u128) (u8,i16) (u8,i32) (u8,i64) (u8,i128) (u8,usize) }
vec4_from_impl! { (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,isize) }
vec4_from_impl! { (u16,u32) (u16,u64) (u16,u128) (u16,i32) (u16,i64) (u16,i128) (u16,usize) }
vec4_from_impl! { (i16,i32) (i16,i64) (i16,i128) (i16,isize) }
vec4_from_impl! { (u32,u64) (u32,u128) (u32,i64) (u32,i128) }
vec4_from_impl! { (i32,i64) (i32,i128) }
vec4_from_impl! { (u64,u128) (u64,i128) }
vec4_from_impl! { (i64,i128) }
vec4_from_impl! { (f32,f64) (f64,f32) }
