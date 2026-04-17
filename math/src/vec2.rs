use {
    crate::*,
    codec::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{
            Add, AddAssign, BitOr, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub,
            SubAssign,
        },
    },
};

/// 2D vector, generic over the component type.
///
/// Supports arithmetic operators for vector addition, subtraction,
/// scalar multiplication/division, negation, and indexing by component.
///
/// For `f32`/`f64` vectors, additional methods are available: [`length`](Vec2::length),
/// [`normalized`](Vec2::normalized),
/// [`length_squared`](Vec2::length_squared), and the `|` operator as a dot product alias.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let a = vec2(1.0f32, 2.0);
/// let b = vec2(3.0, 4.0);
/// let sum = a + b;
/// assert_eq!(sum, vec2(4.0, 6.0));
/// assert_eq!(a.dot(b), 11.0);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Vec2<T> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
}

/// Create a new 2D vector.
pub const fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vec2 { x, y }
}

impl<T: Copy + Zero> Vec2<T> {
    /// Extend to [`Vec3`] by appending a Z component.
    pub fn extend(self, z: T) -> Vec3<T> {
        Vec3 {
            x: self.x,
            y: self.y,
            z,
        }
    }
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
    pub fn dot(self, other: Vec2<T>) -> T {
        self.x * other.x + self.y * other.y
    }
}

impl<T> Index<usize> for Vec2<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("Vec2 index out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Vec2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("Vec2 index out of range"),
        }
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

vec2_impl! { usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 F16 f32 f64 }

// implementations where $t: Real
macro_rules! vec2_real_impl {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {

                /// Calculate squared vector length.
                pub fn length_squared(&self) -> $t {
                    self.x * self.x + self.y * self.y
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
            /// This is an alias for [`dot`](Vec2::dot), provided for
            /// consistency with the geometric algebra inner product notation
            /// used by [`MultiVec201`] and [`MultiVec301`].
            impl BitOr<Vec2<$t>> for Vec2<$t> {
                type Output = $t;
                fn bitor(self,other: Vec2<$t>) -> $t {
                    self.x * other.x + self.y * other.y
                }
            }
        )+
    }
}

vec2_real_impl! { F16 f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
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

vec2_from_impl! { (u8,u16) (u8,u32) (u8,u64) (u8,u128) (u8,i16) (u8,i32) (u8,i64) (u8,i128) (u8,usize) }
vec2_from_impl! { (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,isize) }
vec2_from_impl! { (u16,u32) (u16,u64) (u16,u128) (u16,i32) (u16,i64) (u16,i128) (u16,usize) }
vec2_from_impl! { (i16,i32) (i16,i64) (i16,i128) (i16,isize) }
vec2_from_impl! { (u32,u64) (u32,u128) (u32,i64) (u32,i128) }
vec2_from_impl! { (i32,i64) (i32,i128) }
vec2_from_impl! { (u64,u128) (u64,i128) }
vec2_from_impl! { (i64,i128) }
vec2_from_impl! { (f32,f64) (f64,f32) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_vec2_roundtrip() {
        let val = vec2(1.0f32, 2.0);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Vec2::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }

    #[test]
    fn test_codec_vec2_wire_format() {
        let mut buf = Vec::new();
        vec2(1.0f32, 2.0).encode(&mut buf);
        assert_eq!(buf, [0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40]);
    }

    #[test]
    fn test_codec_vec2_truncated() {
        assert!(Vec2::<f32>::decode(&[0x01, 0x02]).is_err());
    }
}
