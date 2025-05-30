use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{Display, Formatter, Result},
        ops::{
            Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
            DivAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
        },
    },
};

/// 2D projective multivector.
///
/// This is an implementation of 2D projective geometric algebra. The multivector contains the coefficients and operators are overloaded for geometric product (`*`), inner product (`|`), outer product (`^`) and regressive product (`&`).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MultiVec201<T> {
    pub s: T,    // scalar
    pub e0: T,   // ^2 = 0, line at infinity, line equation offset
    pub e1: T,   // ^2 = 1, line perpendicular to X-axis, line X-factor
    pub e2: T,   // ^2 = 1, line perpendicular to Y-axis, line Y-factor
    pub e01: T,  // ^2 = 0, point Y-coordinate
    pub e20: T,  // ^2 = 0, point X-coordinate
    pub e12: T,  // ^2 = -1, point at origin
    pub e012: T, // all surface pseudoscalar
}

impl<T> MultiVec201<T>
where
    T: Copy + Zero + One + Neg<Output = T>,
{
    pub fn from_point(p: &Vec2<T>) -> Self {
        MultiVec201 {
            s: T::ZERO,
            e0: T::ZERO,
            e1: T::ZERO,
            e2: T::ZERO,
            e01: p.y,
            e20: p.x,
            e12: T::ONE,
            e012: T::ZERO,
        }
    }

    pub fn from_direction(d: &Vec2<T>) -> Self {
        MultiVec201 {
            s: T::ZERO,
            e0: T::ZERO,
            e1: T::ZERO,
            e2: T::ZERO,
            e01: d.y,
            e20: d.x,
            e12: T::ZERO,
            e012: T::ZERO,
        }
    }

    pub fn from_line_equation(a: T, b: T, c: T) -> Self {
        MultiVec201 {
            s: T::ZERO,
            e0: c,
            e1: a,
            e2: b,
            e01: T::ZERO,
            e20: T::ZERO,
            e12: T::ZERO,
            e012: T::ZERO,
        }
    }

    pub fn reverse(self) -> Self {
        MultiVec201 {
            s: self.s,
            e0: self.e0,
            e1: self.e1,
            e2: self.e2,
            e01: -self.e01,
            e20: -self.e20,
            e12: -self.e12,
            e012: -self.e012,
        }
    }

    pub fn conj(self) -> Self {
        MultiVec201 {
            s: self.s,
            e0: -self.e0,
            e1: -self.e1,
            e2: -self.e2,
            e01: -self.e01,
            e20: -self.e20,
            e12: -self.e12,
            e012: self.e012,
        }
    }

    pub fn involute(self) -> Self {
        MultiVec201 {
            s: self.s,
            e0: -self.e0,
            e1: -self.e1,
            e2: -self.e2,
            e01: self.e01,
            e20: self.e20,
            e12: self.e12,
            e012: -self.e012,
        }
    }
}

impl<T> Display for MultiVec201<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "[{}, ({}, {},{}),  ({},{}, {}), {}]",
            self.s, self.e0, self.e1, self.e2, self.e01, self.e20, self.e12, self.e012
        )
    }
}

impl<T> Zero for MultiVec201<T>
where
    T: Zero,
{
    const ZERO: Self = MultiVec201 {
        s: T::ZERO,
        e0: T::ZERO,
        e1: T::ZERO,
        e2: T::ZERO,
        e01: T::ZERO,
        e20: T::ZERO,
        e12: T::ZERO,
        e012: T::ZERO,
    };
}

/// Multivector + multivector.
impl<T> Add<MultiVec201<T>> for MultiVec201<T>
where
    T: Add<Output = T>,
{
    type Output = MultiVec201<T>;
    fn add(self, other: Self) -> Self {
        MultiVec201 {
            s: self.s + other.s,
            e0: self.e0 + other.e0,
            e1: self.e1 + other.e1,
            e2: self.e2 + other.e2,
            e01: self.e20 + other.e20,
            e20: self.e01 + other.e01,
            e12: self.e12 + other.e12,
            e012: self.e012 + other.e012,
        }
    }
}

/// Multivector += multivector.
impl<T> AddAssign<MultiVec201<T>> for MultiVec201<T>
where
    MultiVec201<T>: Copy + Add<Output = MultiVec201<T>>,
{
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

/// Multivector - multivector.
impl<T> Sub<MultiVec201<T>> for MultiVec201<T>
where
    T: Sub<Output = T>,
{
    type Output = MultiVec201<T>;
    fn sub(self, other: Self) -> Self {
        MultiVec201 {
            s: self.s - other.s,
            e0: self.e0 - other.e0,
            e1: self.e1 - other.e1,
            e2: self.e2 - other.e2,
            e01: self.e01 - other.e01,
            e20: self.e20 - other.e20,
            e12: self.e12 - other.e12,
            e012: self.e012 - other.e012,
        }
    }
}

/// Multivector -= multivector.
impl<T> SubAssign<MultiVec201<T>> for MultiVec201<T>
where
    MultiVec201<T>: Copy + Sub<Output = MultiVec201<T>>,
{
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

/// Multivector * scalar.
impl<T> Mul<T> for MultiVec201<T>
where
    T: Copy + Mul<Output = T>,
{
    type Output = MultiVec201<T>;
    fn mul(self, other: T) -> Self {
        MultiVec201 {
            s: self.s * other,
            e0: self.e0 * other,
            e1: self.e1 * other,
            e2: self.e2 * other,
            e01: self.e01 * other,
            e20: self.e20 * other,
            e12: self.e12 * other,
            e012: self.e012 * other,
        }
    }
}

/// Multivector *= scalar.
impl<T> MulAssign<T> for MultiVec201<T>
where
    MultiVec201<T>: Copy + Mul<T, Output = MultiVec201<T>>,
{
    fn mul_assign(&mut self, other: T) {
        *self = *self * other;
    }
}

/// geometric product: Multivector * multivector.
impl<T> Mul<MultiVec201<T>> for MultiVec201<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    type Output = MultiVec201<T>;
    fn mul(self, other: MultiVec201<T>) -> Self {
        MultiVec201 {
            s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 - other.e12 * self.e12,
            e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1
                + other.e20 * self.e2
                + other.e1 * self.e01
                - other.e2 * self.e20
                - other.e012 * self.e12
                - other.e12 * self.e012,
            e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e2 * self.e12,
            e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e1 * self.e12,
            e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1
                + other.e012 * self.e2
                + other.s * self.e01
                + other.e12 * self.e20
                - other.e20 * self.e12
                + other.e2 * self.e012,
            e20: other.e20 * self.s - other.e2 * self.e0
                + other.e012 * self.e1
                + other.e0 * self.e2
                - other.e12 * self.e01
                + other.s * self.e20
                + other.e01 * self.e12
                + other.e1 * self.e012,
            e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.s * self.e12,
            e012: other.e012 * self.s
                + other.e12 * self.e0
                + other.e20 * self.e1
                + other.e01 * self.e2
                + other.e2 * self.e01
                + other.e1 * self.e20
                + other.e0 * self.e12
                + other.s * self.e012,
        }
    }
}

/// geometric product: Multivector *= multivector.
impl<T> MulAssign<MultiVec201<T>> for MultiVec201<T>
where
    MultiVec201<T>: Copy + Mul<MultiVec201<T>, Output = MultiVec201<T>>,
{
    fn mul_assign(&mut self, other: MultiVec201<T>) {
        *self = *self * other;
    }
}

/// inner product (dot product): Multivector | multivector.
impl<T> BitOr<MultiVec201<T>> for MultiVec201<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    type Output = MultiVec201<T>;
    fn bitor(self, other: MultiVec201<T>) -> MultiVec201<T> {
        MultiVec201 {
            s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 - other.e12 * self.e12,
            e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1
                + other.e20 * self.e2
                + other.e1 * self.e01
                - other.e2 * self.e20
                - other.e012 * self.e12
                - other.e12 * self.e012,
            e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e2 * self.e12,
            e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e1 * self.e12,
            e01: other.e01 * self.s
                + other.e012 * self.e2
                + other.s * self.e01
                + other.e2 * self.e012,
            e20: other.e20 * self.s
                + other.e012 * self.e1
                + other.s * self.e20
                + other.e1 * self.e012,
            e12: other.e12 * self.s + other.s * self.e12,
            e012: other.e012 * self.s + other.s * self.e012,
        }
    }
}

/// inner product (dot product): Multivector |= multivector.
impl<T> BitOrAssign for MultiVec201<T>
where
    MultiVec201<T>: Copy + BitOr<MultiVec201<T>, Output = MultiVec201<T>>,
{
    fn bitor_assign(&mut self, other: MultiVec201<T>) {
        *self = *self | other;
    }
}

/// outer product (wedge product, meet): Multivector ^ multivector.
impl<T> BitXor for MultiVec201<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    type Output = MultiVec201<T>;
    fn bitxor(self, other: MultiVec201<T>) -> MultiVec201<T> {
        MultiVec201 {
            s: other.s * self.s,
            e0: other.e0 * self.s + other.s * self.e0,
            e1: other.e1 * self.s + other.s * self.e1,
            e2: other.e2 * self.s + other.s * self.e2,
            e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1 + other.s * self.e01,
            e20: other.e20 * self.s - other.e2 * self.e0 + other.e0 * self.e2 + other.s * self.e20,
            e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.s * self.e12,
            e012: other.e012 * self.s
                + other.e12 * self.e0
                + other.e20 * self.e1
                + other.e01 * self.e2
                + other.e2 * self.e01
                + other.e1 * self.e20
                + other.e0 * self.e12
                + other.s * self.e012,
        }
    }
}

/// outer product (wedge product, meet): Multivector ^= multivector.
impl<T> BitXorAssign for MultiVec201<T>
where
    MultiVec201<T>: Copy + BitXor<MultiVec201<T>, Output = MultiVec201<T>>,
{
    fn bitxor_assign(&mut self, other: MultiVec201<T>) {
        *self = *self ^ other;
    }
}

/// regressive product (join): Multivector & multivector.
impl<T> BitAnd for MultiVec201<T>
where
    T: Copy + Mul<Output = T> + Sub<Output = T> + Add<Output = T>,
{
    type Output = MultiVec201<T>;
    fn bitand(self, other: MultiVec201<T>) -> MultiVec201<T> {
        MultiVec201 {
            e012: self.e012 * other.e012,
            e12: self.e12 * other.e012 + self.e012 * other.e12,
            e20: self.e20 * other.e012 + self.e012 * other.e20,
            e01: self.e01 * other.e012 + self.e012 * other.e01,
            e2: self.e2 * other.e012 + self.e20 * other.e12 - self.e12 * other.e20
                + self.e012 * other.e2,
            e1: self.e1 * other.e012 - self.e01 * other.e12
                + self.e12 * other.e01
                + self.e012 * other.e1,
            e0: self.e0 * other.e012 + self.e01 * other.e20 - self.e20 * other.e01
                + self.e012 * other.e0,
            s: self.s * other.e012
                + self.e0 * other.e12
                + self.e1 * other.e20
                + self.e2 * other.e01
                + self.e01 * other.e2
                + self.e20 * other.e1
                + self.e12 * other.e0
                + self.e012 * other.s,
        }
    }
}

/// regressive product (join): Multivector &= multivector.
impl<T> BitAndAssign for MultiVec201<T>
where
    MultiVec201<T>: Copy + BitAnd<MultiVec201<T>, Output = MultiVec201<T>>,
{
    fn bitand_assign(&mut self, other: MultiVec201<T>) {
        *self = *self & other;
    }
}

/// Multivector / scalar.
impl<T> Div<T> for MultiVec201<T>
where
    T: Copy + Div<Output = T>,
{
    type Output = MultiVec201<T>;
    fn div(self, other: T) -> Self {
        MultiVec201 {
            s: self.s / other,
            e0: self.e0 / other,
            e1: self.e1 / other,
            e2: self.e2 / other,
            e01: self.e01 / other,
            e20: self.e20 / other,
            e12: self.e12 / other,
            e012: self.e012 / other,
        }
    }
}

/// Multivector /= scalar.
impl<T> DivAssign<T> for MultiVec201<T>
where
    MultiVec201<T>: Copy + Div<T, Output = MultiVec201<T>>,
{
    fn div_assign(&mut self, other: T) {
        *self = *self / other;
    }
}

/// dual: !multivector.
impl<T> Not for MultiVec201<T> {
    type Output = MultiVec201<T>;
    fn not(self) -> Self {
        MultiVec201 {
            s: self.e012,
            e0: self.e12,
            e1: self.e20,
            e2: self.e01,
            e20: self.e1,
            e01: self.e2,
            e12: self.e0,
            e012: self.s,
        }
    }
}

macro_rules! multivec201_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * multivector.
            impl Mul<MultiVec201<$t>> for $t {
                type Output = MultiVec201<$t>;
                fn mul(self,other: MultiVec201<$t>) -> Self::Output {
                    MultiVec201 {
                        s: self * other.s,
                        e0: self * other.e0,
                        e1: self * other.e1,
                        e2: self * other.e2,
                        e01: self * other.e01,
                        e20: self * other.e20,
                        e12: self * other.e12,
                        e012: self * other.e012,
                    }
                }
            }
        )+
    }
}

multivec201_impl! { f32 f64 }
