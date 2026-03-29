use {
    crate::*,
    std::ops::{Add, Mul, MulAssign},
};

/// 3D rigid-body pose (position and orientation).
///
/// Combines a translation ([`Vec3`]) and a rotation ([`Quat`]) into a
/// single transform. Applying a pose to a vector (`Pose * Vec3`) first
/// rotates, then translates. Poses compose via `Pose * Pose`.
///
/// Convert to a 4x4 matrix with `Mat4x4::from(pose)`.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let p = Pose { p: vec3(1.0f32, 0.0, 0.0), o: Quat::ONE };
/// let v = p * vec3(0.0, 0.0, 0.0);
/// assert_eq!(v, vec3(1.0, 0.0, 0.0));
/// ```
#[derive(Copy,Clone,Debug,PartialEq)]
pub struct Pose<T> {
    /// Position (translation).
    pub p: Vec3<T>,
    /// Orientation (rotation).
    pub o: Quat<T>,
}

/// Create a new pose from position and orientation.
pub const fn pose<T>(p: Vec3<T>, o: Quat<T>) -> Pose<T> {
    Pose { p, o }
}

macro_rules! pose_impl {
    ($($t:ty)+) => {
        $(
            impl One for Pose<$t> {
                const ONE: Self = Pose { p: Vec3::ZERO, o: Quat::ONE };
            }

            impl Pose<$t> {
                /// Invert the pose.
                pub fn inv(self) -> Self {
                    let o = self.o.inv();
                    Pose { p: o * -self.p, o }
                }
            }

            /// Pose * vector.
            impl Mul<Vec3<$t>> for Pose<$t> {
                type Output = Vec3<$t>;
                fn mul(self, other: Vec3<$t>) -> Vec3<$t> {
                    self.o * other + self.p
                }
            }

            /// Pose * pose.
            impl Mul<Pose<$t>> for Pose<$t> {
                type Output = Pose<$t>;
                fn mul(self, other: Pose<$t>) -> Pose<$t> {
                    Pose {
                        p: self.p + self.o * other.p,
                        o: self.o * other.o,
                    }
                }
            }

            /// Pose *= pose.
            impl MulAssign<Pose<$t>> for Pose<$t> {
                fn mul_assign(&mut self, other: Pose<$t>) {
                    self.p = self.p + self.o * other.p;
                    self.o = self.o * other.o;
                }
            }
        )+
    }
}

pose_impl! { f32 f64 }

impl From<Pose<f32>> for Pose<f64> { fn from(value: Pose<f32>) -> Self { Pose { p: value.p.into(), o: value.o.into() } } }
impl From<Pose<f64>> for Pose<f32> { fn from(value: Pose<f64>) -> Self { Pose { p: value.p.into(), o: value.o.into() } } }
