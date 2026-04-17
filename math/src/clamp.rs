/// Clamp a value to the range `[min, max]`.
///
/// Returns `min` if `value < min`, `max` if `value > max`, otherwise `value`.
/// Works with any type that implements `PartialOrd`.
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min { min }
    else if value > max { max }
    else { value }
}

/// Clamp f32 to [0, 1].
pub fn saturate_f32(value: f32) -> f32 {
    clamp(value, 0.0, 1.0)
}

/// Clamp f64 to [0, 1].
pub fn saturate_f64(value: f64) -> f64 {
    clamp(value, 0.0, 1.0)
}

/// Remap an `f32` from the range `[in_lo, in_hi]` to `[out_lo, out_hi]`.
pub fn remap_f32(value: f32, in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32) -> f32 {
    out_lo + (value - in_lo) * (out_hi - out_lo) / (in_hi - in_lo)
}

/// Remap an `f64` from the range `[in_lo, in_hi]` to `[out_lo, out_hi]`.
pub fn remap_f64(value: f64, in_lo: f64, in_hi: f64, out_lo: f64, out_hi: f64) -> f64 {
    out_lo + (value - in_lo) * (out_hi - out_lo) / (in_hi - in_lo)
}

/// Hermite smoothstep for `f32`: `0` below `edge0`, `1` above `edge1`, smooth in between.
pub fn smoothstep_f32(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Hermite smoothstep for `f64`: `0` below `edge0`, `1` above `edge1`, smooth in between.
pub fn smoothstep_f64(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5, 0, 10), 5);
        assert_eq!(clamp(-1, 0, 10), 0);
        assert_eq!(clamp(15, 0, 10), 10);
    }

    #[test]
    fn test_saturate() {
        assert_eq!(saturate_f32(0.5), 0.5);
        assert_eq!(saturate_f32(-1.0), 0.0);
        assert_eq!(saturate_f32(2.0), 1.0);
    }

    #[test]
    fn test_remap() {
        assert_eq!(remap_f32(5.0, 0.0, 10.0, 0.0, 100.0), 50.0);
        assert_eq!(remap_f32(0.0, 0.0, 10.0, 20.0, 40.0), 20.0);
        assert_eq!(remap_f32(10.0, 0.0, 10.0, 20.0, 40.0), 40.0);
    }

    #[test]
    fn test_smoothstep() {
        assert_eq!(smoothstep_f32(0.0, 1.0, -1.0), 0.0);
        assert_eq!(smoothstep_f32(0.0, 1.0, 2.0), 1.0);
        assert_eq!(smoothstep_f32(0.0, 1.0, 0.5), 0.5);
        assert_eq!(smoothstep_f32(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep_f32(0.0, 1.0, 1.0), 1.0);
    }
}
