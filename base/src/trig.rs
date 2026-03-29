use crate::*;

macro_rules! trig_impl {
    ($($t:ty)+) => {
        $(
            impl Vec2<$t> {
                /// Angle between two vectors in radians.
                pub fn angle_between(self, other: Vec2<$t>) -> $t {
                    let d = self.x * other.x + self.y * other.y;
                    let la = (self.x * self.x + self.y * self.y).sqrt();
                    let lb = (other.x * other.x + other.y * other.y).sqrt();
                    let denom = la * lb;
                    if denom == 0.0 { return 0.0; }
                    clamp(d / denom, -1.0, 1.0).acos()
                }
            }

            impl Vec3<$t> {
                /// Angle between two vectors in radians.
                pub fn angle_between(self, other: Vec3<$t>) -> $t {
                    let d = self.x * other.x + self.y * other.y + self.z * other.z;
                    let la = (self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
                    let lb = (other.x * other.x + other.y * other.y + other.z * other.z).sqrt();
                    let denom = la * lb;
                    if denom == 0.0 { return 0.0; }
                    clamp(d / denom, -1.0, 1.0).acos()
                }
            }
        )+
    }
}

trig_impl! { f32 f64 }

/// Convert `f32` degrees to radians.
pub fn radians_f32(degrees: f32) -> f32 {
    degrees * (std::f32::consts::PI / 180.0)
}

/// Convert `f64` degrees to radians.
pub fn radians_f64(degrees: f64) -> f64 {
    degrees * (std::f64::consts::PI / 180.0)
}

/// Convert `f32` radians to degrees.
pub fn degrees_f32(radians: f32) -> f32 {
    radians * (180.0 / std::f32::consts::PI)
}

/// Convert `f64` radians to degrees.
pub fn degrees_f64(radians: f64) -> f64 {
    radians * (180.0 / std::f64::consts::PI)
}

/// Simultaneous sine and cosine of an `f32` angle in radians.
pub fn sincos_f32(angle: f32) -> (f32, f32) {
    (angle.sin(), angle.cos())
}

/// Simultaneous sine and cosine of an `f64` angle in radians.
pub fn sincos_f64(angle: f64) -> (f64, f64) {
    (angle.sin(), angle.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degrees_radians_roundtrip() {
        let deg = 45.0f32;
        let rad = radians_f32(deg);
        let back = degrees_f32(rad);
        assert!((back - deg).abs() < 1e-6);
    }

    #[test]
    fn test_radians_known_values() {
        assert!((radians_f32(180.0) - std::f32::consts::PI).abs() < 1e-6);
        assert!((radians_f64(90.0) - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_sincos() {
        let (s, c) = sincos_f32(std::f32::consts::FRAC_PI_4);
        assert!((s - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((c - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
    }

    #[test]
    fn test_angle_between_vec3() {
        let a = vec3(1.0f32, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        let angle = a.angle_between(b);
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }

    #[test]
    fn test_angle_between_parallel() {
        let a = vec3(1.0f32, 0.0, 0.0);
        let b = vec3(2.0, 0.0, 0.0);
        let angle = a.angle_between(b);
        assert!(angle.abs() < 1e-6);
    }

    #[test]
    fn test_angle_between_vec2() {
        let a = vec2(1.0f32, 0.0);
        let b = vec2(0.0, 1.0);
        let angle = a.angle_between(b);
        assert!((angle - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
    }
}
