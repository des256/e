use {
    crate::*,
    std::ops::{Add, Mul, Sub},
};

/// Linear interpolation between two values.
///
/// Returns `a + (b - a) * t`. When `t = 0` returns `a`, when `t = 1` returns `b`.
///
/// Also available as methods on [`Vec2`], [`Vec3`], and [`Vec4`] for
/// component-wise interpolation, and on [`Quat`] as
/// [`nlerp`](Quat::nlerp) / [`slerp`](Quat::slerp).
pub fn lerp<T>(a: T, b: T, t: T) -> T
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    a + (b - a) * t
}

macro_rules! interp_real_impl {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {
                /// Linear interpolation between two vectors.
                pub fn lerp(self, other: Vec2<$t>, t: $t) -> Vec2<$t> {
                    Vec2 {
                        x: lerp(self.x, other.x, t),
                        y: lerp(self.y, other.y, t),
                    }
                }
            }

            impl Vec3<$t> {
                /// Linear interpolation between two vectors.
                pub fn lerp(self, other: Vec3<$t>, t: $t) -> Vec3<$t> {
                    Vec3 {
                        x: lerp(self.x, other.x, t),
                        y: lerp(self.y, other.y, t),
                        z: lerp(self.z, other.z, t),
                    }
                }
            }

            impl Vec4<$t> {
                /// Linear interpolation between two vectors.
                pub fn lerp(self, other: Vec4<$t>, t: $t) -> Vec4<$t> {
                    Vec4 {
                        x: lerp(self.x, other.x, t),
                        y: lerp(self.y, other.y, t),
                        z: lerp(self.z, other.z, t),
                        w: lerp(self.w, other.w, t),
                    }
                }
            }

            impl Quat<$t> {
                /// Normalized linear interpolation (fast approximation of slerp).
                pub fn nlerp(self, other: Quat<$t>, t: $t) -> Quat<$t> {
                    // Ensure shortest path
                    let dot = self.r * other.r + self.i * other.i + self.j * other.j + self.k * other.k;
                    let other = if dot < 0.0 { -other } else { other };
                    let result = Quat {
                        r: lerp(self.r, other.r, t),
                        i: lerp(self.i, other.i, t),
                        j: lerp(self.j, other.j, t),
                        k: lerp(self.k, other.k, t),
                    };
                    // Normalize
                    let len = (result.r * result.r + result.i * result.i + result.j * result.j + result.k * result.k).sqrt();
                    result / len
                }

                /// Spherical linear interpolation.
                pub fn slerp(self, other: Quat<$t>, t: $t) -> Quat<$t> {
                    let mut dot = self.r * other.r + self.i * other.i + self.j * other.j + self.k * other.k;
                    let other = if dot < 0.0 { dot = -dot; -other } else { other };
                    // Fall back to nlerp for nearly parallel quaternions
                    if dot > 0.9995 {
                        return self.nlerp(other, t);
                    }
                    let theta = dot.acos();
                    let sin_theta = theta.sin();
                    let wa = ((1.0 - t) * theta).sin() / sin_theta;
                    let wb = (t * theta).sin() / sin_theta;
                    Quat {
                        r: wa * self.r + wb * other.r,
                        i: wa * self.i + wb * other.i,
                        j: wa * self.j + wb * other.j,
                        k: wa * self.k + wb * other.k,
                    }
                }
            }
        )+
    }
}

interp_real_impl! { f32 f64 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lerp_scalar() {
        assert_eq!(lerp(0.0f32, 10.0, 0.5), 5.0);
        assert_eq!(lerp(0.0f32, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0f32, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_lerp_vec3() {
        let a = vec3(0.0f32, 0.0, 0.0);
        let b = vec3(10.0, 20.0, 30.0);
        let result = a.lerp(b, 0.5);
        assert_eq!(result, vec3(5.0, 10.0, 15.0));
    }

    #[test]
    fn test_nlerp_quat() {
        let a = Quat::<f32>::ONE;
        let b = Quat::<f32>::from_euler(1.570796327, 0.0, 0.0);
        let mid = a.nlerp(b, 0.5);
        let len = (mid.r * mid.r + mid.i * mid.i + mid.j * mid.j + mid.k * mid.k).sqrt();
        assert!((len - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_slerp_quat_endpoints() {
        let a = Quat::<f32>::ONE;
        let b = Quat::<f32>::from_euler(1.570796327, 0.0, 0.0);
        let start = a.slerp(b, 0.0);
        let end = a.slerp(b, 1.0);
        assert_eq!(start, a);
        assert!((end.r - b.r).abs() < 1e-5);
        assert!((end.i - b.i).abs() < 1e-5);
    }
}
