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

/// 2D projective geometric algebra (PGA) multivector.
///
/// A full multivector in the `P(R*₂,₀,₁)` algebra, containing one
/// coefficient per basis blade (8 total: scalar through pseudoscalar).
///
/// Operator overloads follow PGA conventions:
/// - `*` — geometric product
/// - `|` — inner product (dot / contraction)
/// - `^` — outer product (wedge / meet)
/// - `&` — regressive product (join)
/// - `!` — Hodge dual
///
/// Construct geometric primitives with [`from_point`](MultiVec201::from_point),
/// [`from_direction`](MultiVec201::from_direction), and
/// [`from_line_equation`](MultiVec201::from_line_equation).
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

macro_rules! multivec201_impl {
    ($($t:ty)+) => {
        $(
            impl MultiVec201<$t> {
                /// Embed a 2D point as a grade-2 element (normalized: `e12 = 1`).
                pub fn from_point(p: &Vec2<$t>) -> Self {
                    MultiVec201 {
                        s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,
                        e01: p.y,e20: p.x,e12: 1.0,
                        e012: 0.0,
                    }
                }

                /// Embed a 2D direction as a grade-2 ideal element (`e12 = 0`).
                pub fn from_direction(d: &Vec2<$t>) -> Self {
                    MultiVec201 {
                        s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,
                        e01: d.y,e20: d.x,e12: 0.0,
                        e012: 0.0,
                    }
                }

                /// Embed the line `ax + by + c = 0` as a grade-1 element.
                pub fn from_line_equation(a: $t, b: $t, c: $t) -> Self {
                    MultiVec201 {
                        s: 0.0,e0: c,e1: a,e2: b,
                        e01: 0.0,e20: 0.0,e12: 0.0,
                        e012: 0.0,
                    }
                }

                /// Line through two points (grade-1 element via wedge product).
                ///
                /// The resulting line is not normalized. To get the signed distance
                /// from a point to this line via the inner product, normalize so
                /// that `e1² + e2² = 1`.
                pub fn from_line_through(a: &Vec2<$t>, b: &Vec2<$t>) -> Self {
                    MultiVec201 {
                        s: 0.0,
                        e0: a.x * b.y - a.y * b.x,
                        e1: a.y - b.y,
                        e2: b.x - a.x,
                        e01: 0.0,e20: 0.0,e12: 0.0,
                        e012: 0.0,
                    }
                }

                /// Line through a point along a direction (grade-1 element).
                pub fn from_point_and_direction(p: &Vec2<$t>, d: &Vec2<$t>) -> Self {
                    MultiVec201 {
                        s: 0.0,
                        e0: p.x * d.y - p.y * d.x,
                        e1: -d.y,
                        e2: d.x,
                        e01: 0.0,e20: 0.0,e12: 0.0,
                        e012: 0.0,
                    }
                }

                /// Translation motor: translates by the given displacement.
                ///
                /// Returns the motor `1 + (dx/2)*e20 + (dy/2)*e01`, which
                /// applies translation via the sandwich product `T * x * ~T`.
                pub fn from_translator(displacement: &Vec2<$t>) -> Self {
                    MultiVec201 {
                        s: 1.0,e0: 0.0,e1: 0.0,e2: 0.0,
                        e01: 0.5 * displacement.y,
                        e20: 0.5 * displacement.x,
                        e12: 0.0,
                        e012: 0.0,
                    }
                }

                /// Rotation motor: rotates by `angle` radians around `center`.
                ///
                /// Returns the motor `cos(a/2) + sin(a/2) * P` where `P` is
                /// the embedded point. Apply via the sandwich product `R * x * ~R`.
                pub fn from_rotor(center: &Vec2<$t>, angle: $t) -> Self {
                    let half_angle = 0.5 * angle;
                    let c = half_angle.cos();
                    let s = half_angle.sin();
                    MultiVec201 {
                        s: c,e0: 0.0,e1: 0.0,e2: 0.0,
                        e01: s * center.y,
                        e20: s * center.x,
                        e12: s,
                        e012: 0.0,
                    }
                }

                /// Reverse: negate grades 2 and 3 (flip bivector and trivector signs).
                pub fn reverse(self) -> Self {
                    MultiVec201 {
                        s: self.s,e0: self.e0,e1: self.e1,e2: self.e2,
                        e01: -self.e01,e20: -self.e20,e12: -self.e12,
                        e012: -self.e012,
                    }
                }

                /// Clifford conjugate: negate grades 1 and 2.
                pub fn conj(self) -> Self {
                    MultiVec201 {
                        s: self.s,
                        e0: -self.e0,e1: -self.e1,e2: -self.e2,
                        e01: -self.e01,e20: -self.e20,e12: -self.e12,
                        e012: self.e012,
                    }
                }

                /// Grade involution: negate odd grades (1 and 3).
                pub fn involute(self) -> Self {
                    MultiVec201 {
                        s: self.s,
                        e0: -self.e0,e1: -self.e1,e2: -self.e2,
                        e01: self.e01,e20: self.e20,e12: self.e12,
                        e012: -self.e012,
                    }
                }
            }

            impl Display for MultiVec201<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(
                        f,
                        "[{}, ({}, {},{}),  ({},{}, {}), {}]",
                        self.s, self.e0, self.e1, self.e2, self.e01, self.e20, self.e12, self.e012
                    )
                }
            }

            impl Zero for MultiVec201<$t> {
                const ZERO: Self = MultiVec201 {
                    s: 0.0,e0: 0.0,e1: 0.0,e2: 0.0,
                    e01: 0.0,e20: 0.0,e12: 0.0,e012: 0.0,
                };
            }

            /// Multivector + multivector.
            impl Add for MultiVec201<$t> {
                type Output = Self;
                fn add(self, other: Self) -> Self {
                    MultiVec201 {
                        s: self.s + other.s,e0: self.e0 + other.e0,
                        e1: self.e1 + other.e1,e2: self.e2 + other.e2,
                        e01: self.e01 + other.e01,e20: self.e20 + other.e20,
                        e12: self.e12 + other.e12,e012: self.e012 + other.e012,
                    }
                }
            }

            /// Multivector += multivector.
            impl AddAssign for MultiVec201<$t> {
                fn add_assign(&mut self, other: Self) { *self = *self + other; }
            }

            /// Multivector - multivector.
            impl Sub for MultiVec201<$t> {
                type Output = Self;
                fn sub(self, other: Self) -> Self {
                    MultiVec201 {
                        s: self.s - other.s,e0: self.e0 - other.e0,
                        e1: self.e1 - other.e1,e2: self.e2 - other.e2,
                        e01: self.e01 - other.e01,e20: self.e20 - other.e20,
                        e12: self.e12 - other.e12,e012: self.e012 - other.e012,
                    }
                }
            }

            /// Multivector -= multivector.
            impl SubAssign for MultiVec201<$t> {
                fn sub_assign(&mut self, other: Self) { *self = *self - other; }
            }

            /// -Multivector.
            impl Neg for MultiVec201<$t> {
                type Output = Self;
                fn neg(self) -> Self {
                    MultiVec201 {
                        s: -self.s,e0: -self.e0,e1: -self.e1,e2: -self.e2,
                        e01: -self.e01,e20: -self.e20,e12: -self.e12,e012: -self.e012,
                    }
                }
            }

            /// Multivector * scalar.
            impl Mul<$t> for MultiVec201<$t> {
                type Output = Self;
                fn mul(self, other: $t) -> Self {
                    MultiVec201 {
                        s: self.s * other,e0: self.e0 * other,
                        e1: self.e1 * other,e2: self.e2 * other,
                        e01: self.e01 * other,e20: self.e20 * other,
                        e12: self.e12 * other,e012: self.e012 * other,
                    }
                }
            }

            /// Scalar * multivector.
            impl Mul<MultiVec201<$t>> for $t {
                type Output = MultiVec201<$t>;
                fn mul(self, other: MultiVec201<$t>) -> MultiVec201<$t> {
                    other * self
                }
            }

            /// Multivector *= scalar.
            impl MulAssign<$t> for MultiVec201<$t> {
                fn mul_assign(&mut self, other: $t) { *self = *self * other; }
            }

            /// Multivector / scalar.
            impl Div<$t> for MultiVec201<$t> {
                type Output = Self;
                fn div(self, other: $t) -> Self {
                    MultiVec201 {
                        s: self.s / other,e0: self.e0 / other,
                        e1: self.e1 / other,e2: self.e2 / other,
                        e01: self.e01 / other,e20: self.e20 / other,
                        e12: self.e12 / other,e012: self.e012 / other,
                    }
                }
            }

            /// Multivector /= scalar.
            impl DivAssign<$t> for MultiVec201<$t> {
                fn div_assign(&mut self, other: $t) { *self = *self / other; }
            }

            /// Geometric product: Multivector * multivector.
            impl Mul<MultiVec201<$t>> for MultiVec201<$t> {
                type Output = Self;
                fn mul(self, other: Self) -> Self {
                    MultiVec201 {
                        s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 - other.e12 * self.e12,
                        e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1
                            + other.e20 * self.e2 + other.e1 * self.e01 - other.e2 * self.e20
                            - other.e012 * self.e12 - other.e12 * self.e012,
                        e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e2 * self.e12,
                        e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e1 * self.e12,
                        e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1
                            + other.e012 * self.e2 + other.s * self.e01 + other.e12 * self.e20
                            - other.e20 * self.e12 + other.e2 * self.e012,
                        e20: other.e20 * self.s - other.e2 * self.e0 + other.e012 * self.e1
                            + other.e0 * self.e2 - other.e12 * self.e01 + other.s * self.e20
                            + other.e01 * self.e12 + other.e1 * self.e012,
                        e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.s * self.e12,
                        e012: other.e012 * self.s + other.e12 * self.e0 + other.e20 * self.e1
                            + other.e01 * self.e2 + other.e2 * self.e01 + other.e1 * self.e20
                            + other.e0 * self.e12 + other.s * self.e012,
                    }
                }
            }

            /// Geometric product: Multivector *= multivector.
            impl MulAssign<MultiVec201<$t>> for MultiVec201<$t> {
                fn mul_assign(&mut self, other: Self) { *self = *self * other; }
            }

            /// Inner product (dot product): Multivector | multivector.
            impl BitOr for MultiVec201<$t> {
                type Output = Self;
                fn bitor(self, other: Self) -> Self {
                    MultiVec201 {
                        s: other.s * self.s + other.e1 * self.e1 + other.e2 * self.e2 - other.e12 * self.e12,
                        e0: other.e0 * self.s + other.s * self.e0 - other.e01 * self.e1
                            + other.e20 * self.e2 + other.e1 * self.e01 - other.e2 * self.e20
                            - other.e012 * self.e12 - other.e12 * self.e012,
                        e1: other.e1 * self.s + other.s * self.e1 - other.e12 * self.e2 + other.e2 * self.e12,
                        e2: other.e2 * self.s + other.e12 * self.e1 + other.s * self.e2 - other.e1 * self.e12,
                        e01: other.e01 * self.s + other.e012 * self.e2
                            + other.s * self.e01 + other.e2 * self.e012,
                        e20: other.e20 * self.s + other.e012 * self.e1
                            + other.s * self.e20 + other.e1 * self.e012,
                        e12: other.e12 * self.s + other.s * self.e12,
                        e012: other.e012 * self.s + other.s * self.e012,
                    }
                }
            }

            /// Inner product: Multivector |= multivector.
            impl BitOrAssign for MultiVec201<$t> {
                fn bitor_assign(&mut self, other: Self) { *self = *self | other; }
            }

            /// Outer product (wedge / meet): Multivector ^ multivector.
            impl BitXor for MultiVec201<$t> {
                type Output = Self;
                fn bitxor(self, other: Self) -> Self {
                    MultiVec201 {
                        s: other.s * self.s,
                        e0: other.e0 * self.s + other.s * self.e0,
                        e1: other.e1 * self.s + other.s * self.e1,
                        e2: other.e2 * self.s + other.s * self.e2,
                        e01: other.e01 * self.s + other.e1 * self.e0 - other.e0 * self.e1 + other.s * self.e01,
                        e20: other.e20 * self.s - other.e2 * self.e0 + other.e0 * self.e2 + other.s * self.e20,
                        e12: other.e12 * self.s + other.e2 * self.e1 - other.e1 * self.e2 + other.s * self.e12,
                        e012: other.e012 * self.s + other.e12 * self.e0 + other.e20 * self.e1
                            + other.e01 * self.e2 + other.e2 * self.e01 + other.e1 * self.e20
                            + other.e0 * self.e12 + other.s * self.e012,
                    }
                }
            }

            /// Outer product: Multivector ^= multivector.
            impl BitXorAssign for MultiVec201<$t> {
                fn bitxor_assign(&mut self, other: Self) { *self = *self ^ other; }
            }

            /// Regressive product (join): Multivector & multivector.
            impl BitAnd for MultiVec201<$t> {
                type Output = Self;
                fn bitand(self, other: Self) -> Self {
                    MultiVec201 {
                        e012: self.e012 * other.e012,
                        e12: self.e12 * other.e012 + self.e012 * other.e12,
                        e20: self.e20 * other.e012 + self.e012 * other.e20,
                        e01: self.e01 * other.e012 + self.e012 * other.e01,
                        e2: self.e2 * other.e012 + self.e20 * other.e12 - self.e12 * other.e20
                            + self.e012 * other.e2,
                        e1: self.e1 * other.e012 - self.e01 * other.e12
                            + self.e12 * other.e01 + self.e012 * other.e1,
                        e0: self.e0 * other.e012 + self.e01 * other.e20 - self.e20 * other.e01
                            + self.e012 * other.e0,
                        s: self.s * other.e012 + self.e0 * other.e12 + self.e1 * other.e20
                            + self.e2 * other.e01 + self.e01 * other.e2 + self.e20 * other.e1
                            + self.e12 * other.e0 + self.e012 * other.s,
                    }
                }
            }

            /// Regressive product: Multivector &= multivector.
            impl BitAndAssign for MultiVec201<$t> {
                fn bitand_assign(&mut self, other: Self) { *self = *self & other; }
            }

            /// Dual: !multivector.
            impl Not for MultiVec201<$t> {
                type Output = Self;
                fn not(self) -> Self {
                    MultiVec201 {
                        s: self.e012,e0: self.e12,e1: self.e20,e2: self.e01,
                        e20: self.e1,e01: self.e2,e12: self.e0,
                        e012: self.s,
                    }
                }
            }
        )+
    }
}

multivec201_impl! { f32 f64 }
