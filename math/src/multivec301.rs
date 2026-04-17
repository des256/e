use {
    crate::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{
            Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div,
            DivAssign, Mul, MulAssign, Neg, Not, Sub, SubAssign,
        },
    },
};

/// 3D projective geometric algebra (PGA) multivector.
///
/// A full multivector in the `P(R*₃,₀,₁)` algebra, containing one
/// coefficient per basis blade (16 total: scalar through pseudoscalar).
///
/// Operator overloads follow PGA conventions:
/// - `*` — geometric product
/// - `|` — inner product (dot / contraction)
/// - `^` — outer product (wedge / meet)
/// - `&` — regressive product (join)
/// - `!` — Hodge dual
///
/// Construct geometric primitives with [`from_point`](MultiVec301::from_point),
/// [`from_direction`](MultiVec301::from_direction), and
/// [`from_plane_equation`](MultiVec301::from_plane_equation).
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct MultiVec301<T> {
    /// Scalar (grade 0).
    pub s: T,
    /// `e0` basis vector (`e0² = 0`): ideal plane / plane distance to origin.
    pub e0: T,
    /// `e1` basis vector (`e1² = 1`): X-perpendicular plane.
    pub e1: T,
    /// `e2` basis vector (`e2² = 1`): Y-perpendicular plane.
    pub e2: T,
    /// `e3` basis vector (`e3² = 1`): Z-perpendicular plane.
    pub e3: T,
    /// `e01` bivector (`e01² = 0`): ideal X-perpendicular horizon line.
    pub e01: T,
    /// `e02` bivector (`e02² = 0`): ideal Y-perpendicular horizon line.
    pub e02: T,
    /// `e03` bivector (`e03² = 0`): ideal Z-perpendicular horizon line.
    pub e03: T,
    /// `e12` bivector (`e12² = -1`): Z-axis line.
    pub e12: T,
    /// `e31` bivector (`e31² = -1`): Y-axis line.
    pub e31: T,
    /// `e23` bivector (`e23² = -1`): X-axis line.
    pub e23: T,
    /// `e021` trivector (`e021² = 0`): ideal Z-point / Z distance to origin.
    pub e021: T,
    /// `e013` trivector (`e013² = 0`): ideal Y-point / Y distance to origin.
    pub e013: T,
    /// `e032` trivector (`e032² = 0`): ideal X-point / X distance to origin.
    pub e032: T,
    /// `e123` trivector (`e123² = -1`): origin point.
    pub e123: T,
    /// `e0123` quadvector: pseudoscalar (full volume element).
    pub e0123: T,
}

macro_rules! multivec301_impl {
    ($($t:ty)+) => {
        $(
            impl MultiVec301<$t> {
                /// Embed a 3D point as a grade-3 element (normalized: `e123 = 1`).
                pub fn from_point(p: Vec3<$t>) -> Self {
                    MultiVec301 {
                        s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: p.z,e013: p.y,e032: p.x,e123: 1.0,
                        e0123: 0.0,
                    }
                }

                /// Embed a 3D direction as a grade-3 ideal element (`e123 = 0`).
                pub fn from_direction(d: Vec3<$t>) -> Self {
                    MultiVec301 {
                        s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: d.z,e013: d.y,e032: d.x,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Embed the plane `ax + by + cz + d = 0` as a grade-1 element.
                pub fn from_plane_equation(a: $t, b: $t, c: $t, d: $t) -> Self {
                    MultiVec301 {
                        s: 0.0,e0: d,e1: a,e2: b,e3: c,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Plane from normal vector and signed distance from origin.
                pub fn from_plane_normal(normal: Vec3<$t>, distance: $t) -> Self {
                    MultiVec301 {
                        s: 0.0,e0: distance,e1: normal.x,e2: normal.y,e3: normal.z,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Line through two points (grade-2 bivector via wedge product).
                ///
                /// The resulting line is not normalized.
                pub fn from_line_through(a: Vec3<$t>, b: Vec3<$t>) -> Self {
                    MultiVec301 {
                        s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: b.y - a.y,
                        e02: b.z - a.z,
                        e03: b.x - a.x,
                        e12: a.x * b.y - a.y * b.x,
                        e31: a.z * b.x - a.x * b.z,
                        e23: a.y * b.z - a.z * b.y,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Plane through three points (grade-1 element via triple wedge product).
                ///
                /// The resulting plane is not normalized.
                pub fn from_plane_through(a: Vec3<$t>, b: Vec3<$t>, c: Vec3<$t>) -> Self {
                    let bx = b.x - a.x; let by = b.y - a.y; let bz = b.z - a.z;
                    let cx = c.x - a.x; let cy = c.y - a.y; let cz = c.z - a.z;
                    let nx = by * cz - bz * cy;
                    let ny = bz * cx - bx * cz;
                    let nz = bx * cy - by * cx;
                    let d = -(nx * a.x + ny * a.y + nz * a.z);
                    MultiVec301 {
                        s: 0.0,e0: d,e1: nx,e2: ny,e3: nz,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Translation motor: translates by the given displacement.
                ///
                /// Returns the motor `1 + (d/2)·(dx·e01 + dy·e02 + dz·e03)`,
                /// which applies translation via the sandwich product `T * x * ~T`.
                pub fn from_translator(displacement: Vec3<$t>) -> Self {
                    MultiVec301 {
                        s: 1.0,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: 0.5 * displacement.x,
                        e02: 0.5 * displacement.y,
                        e03: 0.5 * displacement.z,
                        e12: 0.0,e31: 0.0,e23: 0.0,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Rotation motor: rotates by `angle` radians around a line
                /// through the origin along `axis`.
                ///
                /// `axis` should be normalized. Returns the motor
                /// `cos(a/2) + sin(a/2) * L` where `L` is the Euclidean
                /// line. Apply via the sandwich product `R * x * ~R`.
                pub fn from_rotor(axis: Vec3<$t>, angle: $t) -> Self {
                    let half_angle = 0.5 * angle;
                    let c = half_angle.cos();
                    let s = half_angle.sin();
                    MultiVec301 {
                        s: c,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: 0.0,e02: 0.0,e03: 0.0,
                        e12: s * axis.z,e31: s * axis.y,e23: s * axis.x,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// General rigid-motion motor: rotation around `axis` through
                /// `point` by `angle` radians.
                ///
                /// `axis` should be normalized. Combines rotation and the
                /// translation needed to rotate around an off-origin line.
                /// Apply via the sandwich product `M * x * ~M`.
                pub fn from_motor(point: Vec3<$t>, axis: Vec3<$t>, angle: $t) -> Self {
                    let half_angle = 0.5 * angle;
                    let c = half_angle.cos();
                    let s = half_angle.sin();
                    let bx = s * axis.x;
                    let by = s * axis.y;
                    let bz = s * axis.z;
                    let mx = s * (point.y * axis.z - point.z * axis.y);
                    let my = s * (point.z * axis.x - point.x * axis.z);
                    let mz = s * (point.x * axis.y - point.y * axis.x);
                    MultiVec301 {
                        s: c,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                        e01: mx,e02: my,e03: mz,
                        e12: bz,e31: by,e23: bx,
                        e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                        e0123: 0.0,
                    }
                }

                /// Reverse: negate grades 2 and 3 (flip bivector and trivector signs).
                pub fn reverse(self) -> Self {
                    MultiVec301 {
                        s: self.s,e0: self.e0,e1: self.e1,e2: self.e2,e3: self.e3,
                        e01: -self.e01,e02: -self.e02,e03: -self.e03,
                        e12: -self.e12,e31: -self.e31,e23: -self.e23,
                        e021: -self.e021,e013: -self.e013,e032: -self.e032,e123: -self.e123,
                        e0123: self.e0123,
                    }
                }

                /// Clifford conjugate: negate grades 1 and 2.
                pub fn conj(self) -> Self {
                    MultiVec301 {
                        s: self.s,
                        e0: -self.e0,e1: -self.e1,e2: -self.e2,e3: -self.e3,
                        e01: -self.e01,e02: -self.e02,e03: -self.e03,
                        e12: -self.e12,e31: -self.e31,e23: -self.e23,
                        e021: self.e021,e013: self.e013,e032: self.e032,e123: self.e123,
                        e0123: self.e0123,
                    }
                }

                /// Grade involution: negate odd grades (1 and 3).
                pub fn involute(self) -> Self {
                    MultiVec301 {
                        s: self.s,
                        e0: -self.e0,e1: -self.e1,e2: -self.e2,e3: -self.e3,
                        e01: self.e01,e02: self.e02,e03: self.e03,
                        e12: self.e12,e31: self.e31,e23: self.e23,
                        e021: -self.e021,e013: -self.e013,e032: -self.e032,e123: -self.e123,
                        e0123: self.e0123,
                    }
                }
            }

            impl Display for MultiVec301<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "[{}, ({}, {},{},{}),  ({},{},{}, {},{},{}), ({},{},{}, {}),  {}]",
                        self.s,
                        self.e0,self.e1,self.e2,self.e3,
                        self.e01,self.e02,self.e03,self.e12,self.e31,self.e23,
                        self.e021,self.e013,self.e032,self.e123,
                        self.e0123,
                    )
                }
            }

            impl Zero for MultiVec301<$t> {
                const ZERO: Self = MultiVec301 {
                    s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,e3: 0.0,
                    e01: 0.0,e02: 0.0,e03: 0.0,
                    e12: 0.0,e31: 0.0,e23: 0.0,
                    e021: 0.0,e013: 0.0,e032: 0.0,e123: 0.0,
                    e0123: 0.0,
                };
            }

            /// Multivector + multivector.
            impl Add for MultiVec301<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self {
                    MultiVec301 {
                        s: self.s + other.s,e0: self.e0 + other.e0,
                        e1: self.e1 + other.e1,e2: self.e2 + other.e2,e3: self.e3 + other.e3,
                        e01: self.e01 + other.e01,e02: self.e02 + other.e02,e03: self.e03 + other.e03,
                        e12: self.e12 + other.e12,e31: self.e31 + other.e31,e23: self.e23 + other.e23,
                        e021: self.e021 + other.e021,e013: self.e013 + other.e013,e032: self.e032 + other.e032,e123: self.e123 + other.e123,
                        e0123: self.e0123 + other.e0123,
                    }
                }
            }

            /// Multivector += multivector.
            impl AddAssign for MultiVec301<$t> {
                fn add_assign(&mut self, other: Self) { *self = *self + other; }
            }

            /// Multivector - multivector.
            impl Sub for MultiVec301<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self {
                    MultiVec301 {
                        s: self.s - other.s,e0: self.e0 - other.e0,
                        e1: self.e1 - other.e1,e2: self.e2 - other.e2,e3: self.e3 - other.e3,
                        e01: self.e01 - other.e01,e02: self.e02 - other.e02,e03: self.e03 - other.e03,
                        e12: self.e12 - other.e12,e31: self.e31 - other.e31,e23: self.e23 - other.e23,
                        e021: self.e021 - other.e021,e013: self.e013 - other.e013,e032: self.e032 - other.e032,e123: self.e123 - other.e123,
                        e0123: self.e0123 - other.e0123,
                    }
                }
            }

            /// Multivector -= multivector.
            impl SubAssign for MultiVec301<$t> {
                fn sub_assign(&mut self, other: Self) { *self = *self - other; }
            }

            /// -Multivector.
            impl Neg for MultiVec301<$t> {
                type Output = Self;
                fn neg(self) -> Self {
                    MultiVec301 {
                        s: -self.s,e0: -self.e0,e1: -self.e1,e2: -self.e2,e3: -self.e3,
                        e01: -self.e01,e02: -self.e02,e03: -self.e03,
                        e12: -self.e12,e31: -self.e31,e23: -self.e23,
                        e021: -self.e021,e013: -self.e013,e032: -self.e032,e123: -self.e123,
                        e0123: -self.e0123,
                    }
                }
            }

            /// Multivector * scalar.
            impl Mul<$t> for MultiVec301<$t> {
                type Output = Self;
                fn mul(self, other: $t) -> Self {
                    MultiVec301 {
                        s: self.s * other,e0: self.e0 * other,
                        e1: self.e1 * other,e2: self.e2 * other,e3: self.e3 * other,
                        e01: self.e01 * other,e02: self.e02 * other,e03: self.e03 * other,
                        e12: self.e12 * other,e31: self.e31 * other,e23: self.e23 * other,
                        e021: self.e021 * other,e013: self.e013 * other,e032: self.e032 * other,e123: self.e123 * other,
                        e0123: self.e0123 * other,
                    }
                }
            }

            /// Scalar * multivector.
            impl Mul<MultiVec301<$t>> for $t {
                type Output = MultiVec301<$t>;
                fn mul(self, other: MultiVec301<$t>) -> MultiVec301<$t> {
                    other * self
                }
            }

            /// Multivector *= scalar.
            impl MulAssign<$t> for MultiVec301<$t> {
                fn mul_assign(&mut self, other: $t) { *self = *self * other; }
            }

            /// Multivector / scalar.
            impl Div<$t> for MultiVec301<$t> {
                type Output = Self;
                fn div(self, other: $t) -> Self {
                    MultiVec301 {
                        s: self.s / other,e0: self.e0 / other,
                        e1: self.e1 / other,e2: self.e2 / other,e3: self.e3 / other,
                        e01: self.e01 / other,e02: self.e02 / other,e03: self.e03 / other,
                        e12: self.e12 / other,e31: self.e31 / other,e23: self.e23 / other,
                        e021: self.e021 / other,e013: self.e013 / other,e032: self.e032 / other,e123: self.e123 / other,
                        e0123: self.e0123 / other,
                    }
                }
            }

            /// Multivector /= scalar.
            impl DivAssign<$t> for MultiVec301<$t> {
                fn div_assign(&mut self, other: $t) { *self = *self / other; }
            }

            /// Geometric product: Multivector * multivector.
            impl Mul<MultiVec301<$t>> for MultiVec301<$t> {
                type Output = Self;
                fn mul(self, other: Self) -> Self {
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

            /// Geometric product: Multivector *= multivector.
            impl MulAssign<MultiVec301<$t>> for MultiVec301<$t> {
                fn mul_assign(&mut self, other: Self) { *self = *self * other; }
            }

            /// Inner product (dot product): Multivector | multivector.
            impl BitOr for MultiVec301<$t> {
                type Output = Self;
                fn bitor(self, other: Self) -> Self {
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

            /// Inner product: Multivector |= multivector.
            impl BitOrAssign for MultiVec301<$t> {
                fn bitor_assign(&mut self, other: Self) { *self = *self | other; }
            }

            /// Outer product (wedge / meet): Multivector ^ multivector.
            impl BitXor for MultiVec301<$t> {
                type Output = Self;
                fn bitxor(self, other: Self) -> Self {
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

            /// Outer product: Multivector ^= multivector.
            impl BitXorAssign for MultiVec301<$t> {
                fn bitxor_assign(&mut self, other: Self) { *self = *self ^ other; }
            }

            /// Regressive product (join): Multivector & multivector.
            impl BitAnd for MultiVec301<$t> {
                type Output = Self;
                fn bitand(self, other: Self) -> Self {
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

            /// Regressive product: Multivector &= multivector.
            impl BitAndAssign for MultiVec301<$t> {
                fn bitand_assign(&mut self, other: Self) { *self = *self & other; }
            }

            /// Dual: !multivector.
            impl Not for MultiVec301<$t> {
                type Output = Self;
                fn not(self) -> Self {
                    MultiVec301 {
                        s: self.e0123,e0: self.e123,
                        e1: self.e032,e2: self.e013,e3: self.e021,
                        e01: self.e23,e02: self.e31,e03: self.e12,
                        e12: self.e03,e31: self.e02,e23: self.e01,
                        e021: self.e3,e013: self.e2,e032: self.e1,e123: self.e0,
                        e0123: self.s,
                    }
                }
            }
        )+
    }
}

multivec301_impl! { f32 f64 }
