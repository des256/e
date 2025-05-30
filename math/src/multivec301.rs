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
            BitOr,
            BitAnd,
            BitXor,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
            BitOrAssign,
            BitAndAssign,
            BitXorAssign,
            Neg,
            Not,
        },
    },
};

/// 3D projective multivector.
/// 
/// This is an implementation of 3D projective geometric algebra. The multivector contains the coefficients and operators are overloaded for geometric product (`*`), inner product (`|`), outer product (`^`) and regressive product (`&`).
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct MultiVec301<T> {
    pub s: T,  // scalar

    // distance of plane to origin
    pub e0: T,  // ^2 = 0, inf. plane, plane distance to origin

    // normal of the plane
    pub e1: T,  // ^2 = 1, X-perp. plane
    pub e2: T,  // ^2 = 1, Y-perp. plane
    pub e3: T,  // ^2 = 1, Z-perp. plane

    // line at infinity, distance of line to origin
    pub e01: T,  // ^2 = 0, inf. X-perp. horizon
    pub e02: T,  // ^2 = 0, inf. Y-perp. horizon
    pub e03: T,  // ^2 = 0, inf. Z-perp. horizon

    // (normal to?) line through origin
    pub e12: T,  // ^2 = -1, Z-axis
    pub e31: T,  // ^2 = -1, Y-axis
    pub e23: T,  // ^2 = -1, X-axis

    // point at infinity, distance of point to origin
    pub e021: T,  // ^2 = 0, inf. Z-point, Z distance to origin
    pub e013: T,  // ^2 = 0, inf. Y-point, Y distance to origin
    pub e032: T,  // ^2 = 0, inf. X-point, X distance to origin

    // origin
    pub e123: T,  // ^2 = -1, origin

    pub e0123: T,  // all volume pseudoscalar
}

impl<T> MultiVec301<T> where T: Copy + Zero + One + Neg<Output=T> {
    pub fn from_point(p: Vec3<T>) -> Self {
        MultiVec301 {
            s: T::ZERO,
            e0: T::ZERO,
            e1: T::ZERO,
            e2: T::ZERO,
            e3: T::ZERO,
            e01: T::ZERO,
            e02: T::ZERO,
            e03: T::ZERO,
            e12: T::ZERO,
            e31: T::ZERO,
            e23: T::ZERO,
            e021: p.z,
            e013: p.y,
            e032: p.x,
            e123: T::ONE,
            e0123: T::ZERO,
        }
    }

    pub fn from_direction(d: Vec3<T>) -> Self {
        MultiVec301 {
            s: T::ZERO,
            e0: T::ZERO,
            e1: T::ZERO,
            e2: T::ZERO,
            e3: T::ZERO,
            e01: T::ZERO,
            e02: T::ZERO,
            e03: T::ZERO,
            e12: T::ZERO,
            e31: T::ZERO,
            e23: T::ZERO,
            e021: d.z,
            e013: d.y,
            e032: d.x,
            e123: T::ZERO,
            e0123: T::ZERO,
        }
    }

    pub fn from_plane_equation(a: T,b: T,c: T,d: T) -> Self {
        MultiVec301 {
            s: T::ZERO,
            e0: d,
            e1: a,e2: b,e3: c,
            e01: T::ZERO,
            e02: T::ZERO,
            e03: T::ZERO,
            e12: T::ZERO,
            e31: T::ZERO,
            e23: T::ZERO,
            e021: T::ZERO,
            e013: T::ZERO,
            e032: T::ZERO,
            e123: T::ZERO,
            e0123: T::ZERO,
        }
    }

    pub fn reverse(self) -> Self {
        MultiVec301 {
            s: T::ZERO,
            e0: self.e0,
            e1: self.e1,
            e2: self.e2,
            e3: self.e3,
            e01: -self.e01,
            e02: -self.e02,
            e03: -self.e03,
            e12: -self.e12,
            e31: -self.e31,
            e23: -self.e23,
            e021: -self.e021,
            e013: -self.e013,
            e032: -self.e032,
            e123: -self.e123,
            e0123: self.e0123,
        }
    }

    pub fn conj(self) -> Self {
        MultiVec301 {
            s: self.s,
            e0: self.e0,
            e1: -self.e1,
            e2: -self.e2,
            e3: -self.e3,
            e01: -self.e01,
            e02: -self.e02,
            e03: -self.e03,
            e12: -self.e12,
            e31: -self.e31,
            e23: -self.e23,
            e021: self.e021,
            e013: self.e013,
            e032: self.e032,
            e123: self.e123,
            e0123: self.e0123,
        }
    }

    pub fn involute(self) -> Self {
        MultiVec301 {
            s: self.s,
            e0: -self.e0,
            e1: -self.e1,
            e2: -self.e2,
            e3: -self.e3,
            e01: self.e01,
            e02: self.e02,
            e03: self.e03,
            e12: self.e12,
            e31: self.e31,
            e23: self.e23,
            e021: -self.e021,
            e013: -self.e013,
            e032: -self.e032,
            e123: -self.e123,
            e0123: self.e0123,
        }
    }
}

impl<T> Display for MultiVec301<T> where T: Display {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"[{}, ({}, {},{},{}),  ({},{},{}, {},{},{}), ({},{},{}, {}),  {}]",
            self.s,
            self.e0,self.e1,self.e2,self.e3,
            self.e01,self.e02,self.e03,self.e12,self.e31,self.e23,
            self.e021,self.e013,self.e032,self.e123,
            self.e0123,
        )
    }
}

impl<T> Zero for MultiVec301<T> where T: Zero {
    const ZERO: Self = MultiVec301 {
        s: T::ZERO,
        e0: T::ZERO,
        e1: T::ZERO,
        e2: T::ZERO,
        e3: T::ZERO,
        e01: T::ZERO,
        e02: T::ZERO,
        e03: T::ZERO,
        e12: T::ZERO,
        e31: T::ZERO,
        e23: T::ZERO,
        e021: T::ZERO,
        e013: T::ZERO,
        e032: T::ZERO,
        e123: T::ZERO,
        e0123: T::ZERO,
    };
}

/// Multivector + multivector.
impl<T> Add<MultiVec301<T>> for MultiVec301<T> where T: Add<Output=T> {
    type Output = MultiVec301<T>;
    fn add(self,other: Self) -> Self {
        MultiVec301 {
            s: self.s + other.s,
            e0: self.e0 + other.e0,
            e1: self.e1 + other.e1,
            e2: self.e2 + other.e2,
            e3: self.e3 + other.e3,
            e01: self.e01 + other.e01,
            e02: self.e02 + other.e02,
            e03: self.e03 + other.e03,
            e12: self.e12 + other.e12,
            e31: self.e31 + other.e31,
            e23: self.e23 + other.e23,
            e021: self.e021 + other.e021,
            e013: self.e013 + other.e013,
            e032: self.e032 + other.e032,
            e123: self.e123 + other.e123,
            e0123: self.e0123 + other.e0123,
        }
    }
}

/// Multivector += multivector.
impl<T> AddAssign<MultiVec301<T>> for MultiVec301<T> where MultiVec301<T>: Copy + Add<Output=MultiVec301<T>> {
    fn add_assign(&mut self,other: MultiVec301<T>) {
        *self = *self + other;
    }
}

/// Multivector - multivector.
impl<T> Sub<MultiVec301<T>> for MultiVec301<T> where T: Sub<Output=T> {
    type Output = MultiVec301<T>;
    fn sub(self,other: Self) -> Self {
        MultiVec301 {
            s: self.s - other.s,
            e0: self.e0 - other.e0,
            e1: self.e1 - other.e1,
            e2: self.e2 - other.e2,
            e3: self.e3 - other.e3,
            e01: self.e01 - other.e01,
            e02: self.e02 - other.e02,
            e03: self.e03 - other.e03,
            e12: self.e12 - other.e12,
            e31: self.e31 - other.e31,
            e23: self.e23 - other.e23,
            e021: self.e021 - other.e021,
            e013: self.e013 - other.e013,
            e032: self.e032 - other.e032,
            e123: self.e123 - other.e123,
            e0123: self.e0123 - other.e0123,
        }
    }
}

/// Multivector -= multivector.
impl<T> SubAssign<MultiVec301<T>> for MultiVec301<T> where MultiVec301<T>: Copy + Sub<Output=MultiVec301<T>> {
    fn sub_assign(&mut self,other: MultiVec301<T>) {
        *self = *self - other;
    }
}

/// Multivector * scalar.
impl<T> Mul<T> for MultiVec301<T> where T: Copy + Mul<Output=T> {
    type Output = MultiVec301<T>;
    fn mul(self,other: T) -> Self {
        MultiVec301 {
            s: self.s * other,
            e0: self.e0 * other,
            e1: self.e1 * other,
            e2: self.e2 * other,
            e3: self.e3 * other,
            e01: self.e01 * other,
            e02: self.e02 * other,
            e03: self.e03 * other,
            e12: self.e12 * other,
            e31: self.e31 * other,
            e23: self.e23 * other,
            e021: self.e021 * other,
            e013: self.e013 * other,
            e032: self.e032 * other,
            e123: self.e123 * other,
            e0123: self.e0123 * other,
        }
    }
}

/// Multivector *= scalar.
impl<T> MulAssign<T> for MultiVec301<T> where MultiVec301<T>: Copy + Mul<T,Output=MultiVec301<T>> {
    fn mul_assign(&mut self,other: T) {
        *self = *self * other;
    }
}

/// geometric product: Multivector * multivector.
impl<T> Mul<MultiVec301<T>> for MultiVec301<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = Self;
    fn mul(self,other: MultiVec301<T>) -> Self {
        MultiVec301 {
            s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 + other.e3 * self.e3 - other.e12 * self.e12 - other.e31 * self.e31 - other.e23 * self.e23 - other.e123 * self.e123,
            e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1 - other.e02 * self.e2 - other.e03 * self.e3 + other.e1 * self.e01 + other.e2 * self.e02 + other.e3 * self.e03 + other.e021 * self.e12 + other.e013 * self.e31 + other.e032 * self.e23 + other.e12 * self.e021 + other.e31 * self.e013 + other.e23 * self.e032 + other.e0123 * self.e123 - other.e123 * self.e0123,
            e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e31 * self.e3 + other.e2 * self.e12 - other.e3 * self.e31 - other.e123 * self.e23 - other.e23 * self.e123,
            e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e23 * self.e3 - other.e1 * self.e12 - other.e123 * self.e31 + other.e3 * self.e23 - other.e31 * self.e123,
            e3: other.e3 * self.s - other.e31 * self.e1 + other.e23 * self.e2 + other.s * self.e3 - other.e123 * self.e12 + other.e1 * self.e31 - other.e2 * self.e23 - other.e12 * self.e123,
            e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1 - other.e021 * self.e2 + other.e013 * self.e3 + other.s * self.e01 - other.e12 * self.e02 + other.e31 * self.e03 + other.e02 * self.e12 - other.e03 * self.e31 - other.e0123 * self.e23 - other.e2 * self.e021 + other.e3 * self.e013 + other.e123 * self.e032 - other.e032 * self.e123 - other.e23 * self.e0123,
            e02: other.e02 * self.s + other.e2 * self.e0 + other.e021 * self.e1 - other.e0 * self.e2 - other.e032 * self.e3 + other.e12 * self.e01 + other.s * self.e02 - other.e23 * self.e03 - other.e01 * self.e12 - other.e0123 * self.e31 + other.e03 * self.e23 + other.e1 * self.e021 + other.e123 * self.e013 - other.e3 * self.e032 - other.e013 * self.e123 - other.e31 * self.e0123,
            e03: other.e03 * self.s + other.e3 * self.e0 - other.e013 * self.e1 + other.e032 * self.e2 - other.e0 * self.e3 - other.e31 * self.e01 + other.e23 * self.e02 + other.s * self.e03 - other.e0123 * self.e12 + other.e01 * self.e31 - other.e02 * self.e23 + other.e123 * self.e021 - other.e1 * self.e013 + other.e2 * self.e032 - other.e021 * self.e123 - other.e12 * self.e0123,
            e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.e123 * self.e3 + other.s * self.e12 + other.e23 * self.e31 - other.e31 * self.e23 + other.e3 * self.e123,
            e31: other.e31 * self.s - other.e3 * self.e1 + other.e123 * self.e2 + other.e1 * self.e3 - other.e23 * self.e12 + other.s * self.e31 + other.e12 * self.e23 + other.e2 * self.e123,
            e23: other.e23 * self.s + other.e123 * self.e1 + other.e3 * self.e2 - other.e2 * self.e3 + other.e31 * self.e12 - other.e12 * self.e31 + other.s * self.e23 + other.e1 * self.e123,
            e021: other.e021 * self.s - other.e12 * self.e0 + other.e02 * self.e1 - other.e01 * self.e2 + other.e0123 * self.e3 - other.e2 * self.e01 + other.e1 * self.e02 - other.e123 * self.e03 - other.e0 * self.e12 + other.e032 * self.e31 - other.e013 * self.e23 + other.s * self.e021 + other.e23 * self.e013 - other.e31 * self.e032 + other.e03 * self.e123 - other.e3 * self.e0123,
            e013: other.e013 * self.s - other.e31 * self.e0 - other.e03 * self.e1 + other.e0123 * self.e2 + other.e01 * self.e3 + other.e3 * self.e01 - other.e123 * self.e02 - other.e1 * self.e03 - other.e032 * self.e12 - other.e0 * self.e31 + other.e021 * self.e23 - other.e23 * self.e021 + other.s * self.e013 + other.e12 * self.e032 + other.e02 * self.e123 - other.e2 * self.e0123,
            e032: other.e032 * self.s - other.e23 * self.e0 + other.e0123 * self.e1 + other.e03 * self.e2 - other.e02 * self.e3 - other.e123 * self.e01 - other.e3 * self.e02 + other.e2 * self.e03 + other.e013 * self.e12 - other.e021 * self.e31 - other.e0 * self.e23 + other.e31 * self.e021 - other.e12 * self.e013 + other.s * self.e032 + other.e01 * self.e123 - other.e1 * self.e0123,
            e123: other.e123 * self.s + other.e23 * self.e1 + other.e31 * self.e2 + other.e12 * self.e3 + other.e3 * self.e12 + other.e2 * self.e31 + other.e1 * self.e23 + other.s * self.e123,
            e0123: other.e0123 * self.s + other.e123 * self.e0 + other.e032 * self.e1 + other.e013 * self.e2 + other.e021 * self.e3 + other.e23 * self.e01 + other.e31 * self.e02 + other.e12 * self.e03 + other.e03 * self.e12 + other.e02 * self.e31 + other.e01 * self.e23 - other.e3 * self.e021 - other.e2 * self.e013 - other.e1 * self.e032 - other.e0 * self.e123 + other.s * self.e0123,
        }
    }
}

/// geometric product: Multivector *= multivector.
impl<T> MulAssign<MultiVec301<T>> for MultiVec301<T> where MultiVec301<T>: Copy + Mul<MultiVec301<T>,Output=MultiVec301<T>> {
    fn mul_assign(&mut self,other: MultiVec301<T>) {
        *self = *self * other;
    }
}

/// inner product (dot product): Multivector | multivector.
impl<T> BitOr<MultiVec301<T>> for MultiVec301<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = MultiVec301<T>;
    fn bitor(self,other: MultiVec301<T>) -> MultiVec301<T> {
        MultiVec301 {
            s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 + other.e3 * self.e3 - other.e12 * self.e12 - other.e31 * self.e31 - other.e23 * self.e23 - other.e123 * self.e123,
            e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1 - other.e02 * self.e2 - other.e03 * self.e3 + other.e1 * self.e01 + other.e2 * self.e02 + other.e3 * self.e03 + other.e021 * self.e12 + other.e013 * self.e31 + other.e032 * self.e23 + other.e12 * self.e021 + other.e31 * self.e013 + other.e23 * self.e032 + other.e0123 * self.e123 - other.e123 * self.e0123,
            e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e31 * self.e3 + other.e2 * self.e12 - other.e3 * self.e31 - other.e123 * self.e23 - other.e23 * self.e123,
            e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e23 * self.e3 - other.e1 * self.e12 - other.e123 * self.e31 + other.e3 * self.e23 - other.e31 * self.e123,
            e3: other.e3 * self.s - other.e31 * self.e1 + other.e23 * self.e2 + other.s * self.e3 - other.e123 * self.e12 + other.e1 * self.e31 - other.e2 * self.e23 - other.e12 * self.e123,
            e01: other.e01 * self.s - other.e021 * self.e2 + other.e013 * self.e3 + other.s * self.e01 - other.e0123 * self.e23 - other.e2 * self.e021 + other.e3 * self.e013 - other.e23 * self.e0123,
            e02: other.e02 * self.s + other.e021 * self.e1 - other.e032 * self.e3 + other.s * self.e02 - other.e0123 * self.e31 + other.e1 * self.e021 - other.e3 * self.e032 - other.e31 * self.e0123,
            e03: other.e03 * self.s - other.e013 * self.e1 + other.e032 * self.e2 + other.s * self.e03 - other.e0123 * self.e12 - other.e1 * self.e013 + other.e2 * self.e032 - other.e12 * self.e0123,
            e12: other.e12 * self.s + other.e123 * self.e3 + other.s * self.e12 + other.e3 * self.e123,
            e31: other.e31 * self.s + other.e123 * self.e2 + other.s * self.e31 + other.e2 * self.e123,
            e23: other.e23 * self.s + other.e123 * self.e1 + other.s * self.e23 + other.e1 * self.e123,
            e021: other.e021 * self.s + other.e0123 * self.e3 + other.s * self.e021 - other.e3 * self.e0123,
            e013: other.e013 * self.s + other.e0123 * self.e2 + other.s * self.e013 - other.e2 * self.e0123,
            e032: other.e032 * self.s + other.e0123 * self.e1 + other.s * self.e032 - other.e1 * self.e0123,
            e123: other.e123 * self.s + other.s * self.e123,
            e0123: other.e0123 * self.s + other.s * self.e0123,
        }
    }
}

/// inner product (dot product): Multivector |= multivector.
impl<T> BitOrAssign for MultiVec301<T> where MultiVec301<T>: Copy + BitOr<MultiVec301<T>,Output=MultiVec301<T>> {
    fn bitor_assign(&mut self,other: MultiVec301<T>) {
        *self = *self | other;
    }
}

/// outer product (wedge product, meet): Multivector ^ multivector.
impl<T> BitXor for MultiVec301<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> {
    type Output = MultiVec301<T>;
    fn bitxor(self,other: MultiVec301<T>) -> MultiVec301<T> {
        MultiVec301 {
            s: other.s * self.s,
            e0: other.e0 * self.s + other.s * self.e0,
            e1: other.e1 * self.s + other.s * self.e1,
            e2: other.e2 * self.s + other.s * self.e2,
            e3: other.e3 * self.s + other.s * self.e3,
            e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1 + other.s * self.e01,
            e02: other.e02 * self.s + other.e2 * self.e0 - other.e0 * self.e2 + other.s * self.e02,
            e03: other.e03 * self.s + other.e3 * self.e0 - other.e0 * self.e3 + other.s * self.e03,
            e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.s * self.e12,
            e31: other.e31 * self.s - other.e3 * self.e1 + other.e1 * self.e3 + other.s * self.e31,
            e23: other.e23 * self.s + other.e3 * self.e2 - other.e2 * self.e3 + other.s * self.e23,
            e021: other.e021 * self.s - other.e12 * self.e0 + other.e02 * self.e1 - other.e01 * self.e2 - other.e2 * self.e01 + other.e1 * self.e02 - other.e0 * self.e12 + other.s * self.e021,
            e013: other.e013 * self.s - other.e31 * self.e0 - other.e03 * self.e1 + other.e01 * self.e3 + other.e3 * self.e01 - other.e1 * self.e03 - other.e0 * self.e31 + other.s * self.e013,
            e032: other.e032 * self.s - other.e23 * self.e0 + other.e03 * self.e2 - other.e02 * self.e3 - other.e3 * self.e02 + other.e2 * self.e03 - other.e0 * self.e23 + other.s * self.e032,
            e123: other.e123 * self.s + other.e23 * self.e1 + other.e31 * self.e2 + other.e12 * self.e3 + other.e3 * self.e12 + other.e2 * self.e31 + other.e1 * self.e23 + other.s * self.e123,
            e0123: other.e0123 * self.s + other.e123 * self.e0 + other.e032 * self.e1 + other.e013 * self.e2 + other.e021 * self.e3 + other.e23 * self.e01 + other.e31 * self.e02 + other.e12 * self.e03 + other.e03 * self.e12 + other.e02 * self.e31 + other.e01 * self.e23 - other.e3 * self.e021 - other.e2 * self.e013 - other.e1 * self.e032 - other.e0 * self.e123 + other.s * self.e0123,
        }
    }
}

/// outer product (wedge product, meet): Multivector ^= multivector.
impl<T> BitXorAssign for MultiVec301<T> where MultiVec301<T>: Copy + BitXor<MultiVec301<T>,Output=MultiVec301<T>> {
    fn bitxor_assign(&mut self,other: MultiVec301<T>) {
        *self = *self ^ other;
    }
}

/// regressive product (join): Multivector & multivector.
impl<T> BitAnd for MultiVec301<T> where T: Copy + Mul<Output=T> + Sub<Output=T> + Add<Output=T> + Neg<Output=T> {
    type Output = MultiVec301<T>;
    fn bitand(self,other: MultiVec301<T>) -> MultiVec301<T> {
        MultiVec301 {
            e0123: self.e0123 * other.e0123,
            e123: -self.e123 * other.e0123 + self.e0123 * other.e123,
            e032: self.e032 * other.e0123 + self.e0123 * other.e032,
            e013: self.e013 * other.e0123 + self.e0123 * other.e013,
            e021: self.e021 * other.e0123 + self.e0123 * other.e021,
            e23: self.e23 * other.e0123 + self.e032 * other.e123 - self.e123 * other.e032 + self.e0123 * other.e23,
            e31: self.e31 * other.e0123 + self.e013 * other.e123 - self.e123 * other.e013 + self.e0123 * other.e31,
            e12: self.e12 * other.e0123 + self.e021 * other.e123 - self.e123 * other.e021 + self.e0123 * other.e12,
            e03: self.e03 * other.e0123 + self.e013 * other.e032 - self.e032 * other.e013 + self.e0123 * other.e03,
            e02: self.e02 * other.e0123 - self.e021 * other.e032 + self.e032 * other.e021 + self.e0123 * other.e02,
            e01: self.e01 * other.e0123 + self.e021 * other.e013 - self.e013 * other.e021 + self.e0123 * other.e01,
            e3: self.e3 * other.e0123 + self.e03 * other.e123 - self.e31 * other.e032 + self.e23 * other.e013 + self.e013 * other.e23 - self.e032 * other.e31 + self.e123 * other.e03 + self.e0123 * other.e3,
            e2: self.e2 * other.e0123 + self.e02 * other.e123 + self.e12 * other.e032 - self.e23 * other.e021 - self.e021 * other.e23 + self.e032 * other.e12 + self.e123 * other.e02 + self.e0123 * other.e2,
            e1: self.e1 * other.e0123 + self.e01 * other.e123 - self.e12 * other.e013 + self.e31 * other.e021 + self.e021 * other.e31 - self.e013 * other.e12 + self.e123 * other.e01 + self.e0123 * other.e1,
            e0: self.e0 * other.e0123 - self.e01 * other.e032 - self.e02 * other.e013 - self.e03 * other.e021 - self.e021 * other.e03 - self.e013 * other.e02 - self.e032 * other.e01 + self.e0123 * other.e0,
            s: self.s * other.e0123 - self.e0 * other.e123 - self.e1 * other.e032 - self.e2 * other.e013 - self.e3 * other.e021 + self.e01 * other.e23 + self.e02 * other.e31 + self.e03 * other.e12 + self.e12 * other.e03 + self.e31 * other.e02 + self.e23 * other.e01 + self.e021 * other.e3 + self.e013 * other.e2 + self.e032 * other.e1 + self.e123 * other.e0 + self.e0123 * other.s,
        }
    }
}

/// regressive product (join): Multivector &= multivector.
impl<T> BitAndAssign for MultiVec301<T> where MultiVec301<T>: Copy + BitAnd<MultiVec301<T>,Output=MultiVec301<T>> {
    fn bitand_assign(&mut self,other: MultiVec301<T>) {
        *self = *self & other;
    }
}

/// Multivector / scalar.
impl<T> Div<T> for MultiVec301<T> where T: Copy + Div<Output=T> {
    type Output = MultiVec301<T>;
    fn div(self,other: T) -> Self {
        MultiVec301 {
            s: self.s / other,
            e0: self.e0 / other,
            e1: self.e1 / other,
            e2: self.e2 / other,
            e3: self.e3 / other,
            e01: self.e01 / other,
            e02: self.e02 / other,
            e03: self.e03 / other,
            e12: self.e12 / other,
            e31: self.e31 / other,
            e23: self.e23 / other,
            e021: self.e021 / other,
            e013: self.e013 / other,
            e032: self.e032 / other,
            e123: self.e123 / other,
            e0123: self.e0123 / other,
        }
    }
}

/// Multivector /= scalar.
impl<T> DivAssign<T> for MultiVec301<T> where MultiVec301<T>: Copy + Div<T,Output=MultiVec301<T>> {
    fn div_assign(&mut self,other: T) {
        *self = *self / other;
    }
}

/// dual: !multivector.
impl<T> Not for MultiVec301<T> {
    type Output = MultiVec301<T>;
    fn not(self) -> Self {
        MultiVec301 {
            s: self.e0123,
            e0: self.e123,
            e1: self.e032,
            e2: self.e013,
            e3: self.e021,
            e01: self.e23,
            e02: self.e31,
            e03: self.e12,
            e12: self.e03,
            e31: self.e02,
            e23: self.e01,
            e021: self.e3,
            e013: self.e2,
            e032: self.e1,
            e123: self.e0,
            e0123: self.s,
        }
    }
}

macro_rules! multivec301_impl {
    ($($t:ty)+) => {
        $(
            /// Scalar * multivector.
            impl Mul<MultiVec301<$t>> for $t {
                type Output = MultiVec301<$t>;
                fn mul(self,other: MultiVec301<$t>) -> Self::Output {
                    MultiVec301 {
                        s: self * other.s,
                        e0: self * other.e0,
                        e1: self * other.e1,
                        e2: self * other.e2,
                        e3: self * other.e3,
                        e01: self * other.e01,
                        e02: self * other.e02,
                        e03: self * other.e03,
                        e12: self * other.e12,
                        e31: self * other.e31,
                        e23: self * other.e23,
                        e021: self * other.e021,
                        e013: self * other.e013,
                        e032: self * other.e032,
                        e123: self * other.e123,
                        e0123: self * other.e0123,
                    }
                }
            }
        )+
    }
}

multivec301_impl! { f32 f64 }