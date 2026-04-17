use {
    crate::*,
    codec::*,
    std::fmt::{Display, Formatter, Result},
};

/// Euler angles (XYZ intrinsic rotation order).
///
/// Stores rotation angles around X (pitch), Y (yaw), and Z (roll) axes.
/// Primarily useful for human-readable orientation editing (e.g. editor UIs).
/// For computation, convert to [`Quat`] or [`Mat3x3`] to avoid gimbal lock.
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Euler<T> {
    /// Rotation around the X axis (pitch) in radians.
    pub x: T,
    /// Rotation around the Y axis (yaw) in radians.
    pub y: T,
    /// Rotation around the Z axis (roll) in radians.
    pub z: T,
}

/// Create new Euler angles.
pub const fn euler<T>(x: T, y: T, z: T) -> Euler<T> {
    Euler { x, y, z }
}

macro_rules! euler_impl {
    ($($t:ty)+) => {
        $(
            impl Display for Euler<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "(x={}, y={}, z={})", self.x, self.y, self.z)
                }
            }

            impl Zero for Euler<$t> {
                const ZERO: Self = Euler { x: 0.0, y: 0.0, z: 0.0 };
            }

            impl From<Euler<$t>> for Quat<$t> {
                fn from(e: Euler<$t>) -> Quat<$t> {
                    Quat::<$t>::from_euler(e.x, e.y, e.z)
                }
            }

            impl From<Quat<$t>> for Euler<$t> {
                fn from(q: Quat<$t>) -> Euler<$t> {
                    let x = (2.0 * (q.r * q.i - q.j * q.k))
                        .atan2(1.0 - 2.0 * (q.i * q.i + q.j * q.j));

                    let sinp = 2.0 * (q.i * q.k + q.r * q.j);
                    let y = if sinp.abs() >= 1.0 {
                        std::f64::consts::FRAC_PI_2.copysign(sinp as f64) as $t
                    } else {
                        sinp.asin()
                    };

                    let z = (2.0 * (q.r * q.k - q.i * q.j))
                        .atan2(1.0 - 2.0 * (q.j * q.j + q.k * q.k));

                    Euler { x, y, z }
                }
            }

            impl From<Euler<$t>> for Mat3x3<$t> {
                fn from(e: Euler<$t>) -> Mat3x3<$t> {
                    let q: Quat<$t> = e.into();
                    q.into()
                }
            }
        )+
    }
}

euler_impl! { f32 f64 }

// Precision conversions
impl From<Euler<f32>> for Euler<f64> {
    fn from(value: Euler<f32>) -> Self {
        Euler {
            x: value.x as f64,
            y: value.y as f64,
            z: value.z as f64,
        }
    }
}

impl From<Euler<f64>> for Euler<f32> {
    fn from(value: Euler<f64>) -> Self {
        Euler {
            x: value.x as f32,
            y: value.y as f32,
            z: value.z as f32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_to_quat_identity() {
        let e = euler(0.0f32, 0.0, 0.0);
        let q: Quat<f32> = e.into();
        assert_eq!(q, Quat::ONE);
    }

    #[test]
    fn test_euler_quat_roundtrip() {
        let original = euler(0.3f32, 0.5, 0.7);
        let q: Quat<f32> = original.into();
        let back: Euler<f32> = q.into();
        assert!((back.x - original.x).abs() < 1e-5);
        assert!((back.y - original.y).abs() < 1e-5);
        assert!((back.z - original.z).abs() < 1e-5);
    }

    #[test]
    fn test_euler_to_mat3x3() {
        let e = euler(0.523598776f32, 0.785398163, 1.047197551);
        let m1: Mat3x3<f32> = e.into();
        let q: Quat<f32> = e.into();
        let m2: Mat3x3<f32> = q.into();
        assert_eq!(m1, m2);
    }

    #[test]
    fn test_euler_90_degrees() {
        let e = euler(std::f32::consts::FRAC_PI_2, 0.0, 0.0);
        let q: Quat<f32> = e.into();
        let back: Euler<f32> = q.into();
        assert!((back.x - e.x).abs() < 1e-5);
        assert!(back.y.abs() < 1e-5);
        assert!(back.z.abs() < 1e-5);
    }

    #[test]
    fn test_codec_euler_roundtrip() {
        let val = Euler {
            x: 0.1f32,
            y: 0.2,
            z: 0.3,
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Euler::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
