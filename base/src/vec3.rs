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

/// 3D vector, generic over the component type.
///
/// Supports arithmetic operators for vector addition, subtraction,
/// scalar multiplication/division, negation, and indexing by component.
///
/// For `f32`/`f64` vectors, additional methods are available: [`length`](Vec3::length),
/// [`normalized`](Vec3::normalized),
/// [`length_squared`](Vec3::length_squared), [`angle_between`](Vec3::angle_between),
/// and the `|` operator as a dot product alias.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let a = vec3(1.0f32, 0.0, 0.0);
/// let b = vec3(0.0, 1.0, 0.0);
/// let c = a.cross(b);
/// assert_eq!(c, vec3(0.0, 0.0, 1.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Vec3<T> {
    /// X component.
    pub x: T,
    /// Y component.
    pub y: T,
    /// Z component.
    pub z: T,
}

/// Create a new 3D vector.
pub const fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
}

impl<T: Copy> Vec3<T> {
    /// Project to Vec2 (drop z).
    pub fn xy(self) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.y,
        }
    }

    /// Swizzle xz.
    pub fn xz(self) -> Vec2<T> {
        Vec2 {
            x: self.x,
            y: self.z,
        }
    }

    /// Swizzle yz.
    pub fn yz(self) -> Vec2<T> {
        Vec2 {
            x: self.y,
            y: self.z,
        }
    }

    /// Extend to [`Vec4`] by appending a W component.
    pub fn extend(self, w: T) -> Vec4<T> {
        Vec4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }
}

impl<T> Vec3<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    /// Unit vector in positive X-direction.
    pub const UNIT_X: Self = Vec3 {
        x: T::ONE,
        y: T::ZERO,
        z: T::ZERO,
    };

    /// Unit vector in positive Y-direction.
    pub const UNIT_Y: Self = Vec3 {
        x: T::ZERO,
        y: T::ONE,
        z: T::ZERO,
    };

    /// Unit vector in positive Z-direction.
    pub const UNIT_Z: Self = Vec3 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ONE,
    };

    /// Calculate dot product.
    pub fn dot(self, other: Vec3<T>) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Calculate cross product.
    pub fn cross(self, other: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
}

impl<T> Index<usize> for Vec3<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 index out of range"),
        }
    }
}

impl<T> IndexMut<usize> for Vec3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("Vec3 index out of range"),
        }
    }
}

impl<T> Display for Vec3<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({},{},{})", self.x, self.y, self.z)
    }
}

impl<T> Zero for Vec3<T>
where
    T: Zero,
{
    const ZERO: Vec3<T> = Vec3 {
        x: T::ZERO,
        y: T::ZERO,
        z: T::ZERO,
    };
}

/// Vector + vector.
impl<T> Add<Vec3<T>> for Vec3<T>
where
    T: Add<Output = T>,
{
    type Output = Vec3<T>;
    fn add(self, other: Self) -> Self {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

/// Vector += vector.
impl<T> AddAssign<Vec3<T>> for Vec3<T>
where
    T: AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

/// Vector - vector.
impl<T> Sub<Vec3<T>> for Vec3<T>
where
    T: Copy + Sub<Output = T>,
{
    type Output = Vec3<T>;
    fn sub(self, other: Self) -> Self {
        Vec3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

/// Vector -= vector.
impl<T> SubAssign<Vec3<T>> for Vec3<T>
where
    T: Copy + SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
    }
}

/// -Vector.
impl<T> Neg for Vec3<T>
where
    T: Neg<Output = T>,
{
    type Output = Vec3<T>;
    fn neg(self) -> Self::Output {
        Vec3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

/// Vector * scalar.
impl<T> Mul<T> for Vec3<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = Vec3<T>;
    fn mul(self, other: T) -> Self::Output {
        Vec3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
        }
    }
}

/// Vector *= scalar.
impl<T> MulAssign<T> for Vec3<T>
where
    T: Copy + MulAssign,
{
    fn mul_assign(&mut self, other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
    }
}

/// Vector / scalar.
impl<T> Div<T> for Vec3<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = Vec3<T>;
    fn div(self, other: T) -> Self::Output {
        Vec3 {
            x: self.x / other,
            y: self.y / other,
            z: self.z / other,
        }
    }
}

/// Vector /= scalar.
impl<T> DivAssign<T> for Vec3<T>
where
    T: Copy + DivAssign,
{
    fn div_assign(&mut self, other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
    }
}

macro_rules! vec3_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * vector.
            impl Mul<Vec3<$t>> for $t {
                type Output = Vec3<$t>;
                fn mul(self,other: Vec3<$t>) -> Self::Output {
                    Vec3 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                    }
                }
            }
        )+
    }
}

vec3_impl! { usize isize u8 i8 u16 i16 u32 i32 u64 i64 u128 i128 F16 f32 f64 }

macro_rules! vec3_real_impl {
    ($($t:ty)+) => {
        $(
            impl Vec3<$t> {

                /// Calculate squared vector length.
                pub fn length_squared(&self) -> $t {
                    self.x * self.x + self.y * self.y + self.z * self.z
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
            /// Alias for [`dot`](Vec3::dot); matches the geometric algebra
            /// inner product notation used by [`MultiVec201`] and [`MultiVec301`].
            impl BitOr<Vec3<$t>> for Vec3<$t> {
                type Output = $t;
                fn bitor(self,other: Vec3<$t>) -> $t {
                    self.x * other.x + self.y * other.y + self.z * other.z
                }
            }
        )+
    }
}

vec3_real_impl! { F16 f32 f64 }

// lossless conversions matching std::convert::From for the corresponding primitive types
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! vec3_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Vec3<$t>> for Vec3<$u> {
                fn from(value: Vec3<$t>) -> Self { Vec3 { x: value.x as $u,y: value.y as $u,z: value.z as $u, } }
            }
        )+
    }
}

vec3_from_impl! { (u8,u16) (u8,u32) (u8,u64) (u8,u128) (u8,i16) (u8,i32) (u8,i64) (u8,i128) (u8,usize) }
vec3_from_impl! { (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,isize) }
vec3_from_impl! { (u16,u32) (u16,u64) (u16,u128) (u16,i32) (u16,i64) (u16,i128) (u16,usize) }
vec3_from_impl! { (i16,i32) (i16,i64) (i16,i128) (i16,isize) }
vec3_from_impl! { (u32,u64) (u32,u128) (u32,i64) (u32,i128) }
vec3_from_impl! { (i32,i64) (i32,i128) }
vec3_from_impl! { (u64,u128) (u64,i128) }
vec3_from_impl! { (i64,i128) }
vec3_from_impl! { (f32,f64) (f64,f32) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_vec3_roundtrip() {
        let val = vec3(1.0f64, 2.0, 3.0);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Vec3::<f64>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
