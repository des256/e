use {
    codec::Codec,
    std::fmt::{Display, Formatter, Result},
};

/// Asymmetric field-of-view specification in radians.
///
/// Stores four half-angles (left, right, bottom, top) measured from the
/// view axis. Used by `Mat4x4::perspective_from_fov` to build asymmetric
/// projection matrices (e.g. for VR per-eye rendering).
///
/// For a standard symmetric FOV, use [`Fov::symmetric`].
#[derive(Copy, Clone, Debug, Codec)]
pub struct Fov<T> {
    /// Left half-angle (typically negative).
    pub l: T,
    /// Right half-angle (typically positive).
    pub r: T,
    /// Bottom half-angle (typically negative).
    pub b: T,
    /// Top half-angle (typically positive).
    pub t: T,
}

/// Create a new FOV from left, right, bottom, top angles.
pub const fn fov<T>(l: T, r: T, b: T, t: T) -> Fov<T> {
    Fov { l, r, b, t }
}

macro_rules! fov_impl {
    ($($t:ty)+) => {
        $(
            impl Fov<$t> {
                /// Create symmetric FOV from vertical FOV angle and aspect ratio.
                pub fn symmetric(fovy: $t, aspect: $t) -> Self {
                    let half_v = fovy * 0.5;
                    let half_h = (half_v.tan() * aspect).atan();
                    Fov { l: -half_h, r: half_h, b: -half_v, t: half_v }
                }

                /// Horizontal FOV angle.
                pub fn horizontal(&self) -> $t {
                    self.r - self.l
                }

                /// Vertical FOV angle.
                pub fn vertical(&self) -> $t {
                    self.t - self.b
                }

                /// Aspect ratio (horizontal / vertical).
                pub fn aspect(&self) -> $t {
                    let h = self.r.tan() - self.l.tan();
                    let v = self.t.tan() - self.b.tan();
                    h / v
                }
            }

            impl Display for Fov<$t> {
                fn fmt(&self, f: &mut Formatter) -> Result {
                    write!(f, "({}..{}, {}..{})", self.l, self.r, self.b, self.t)
                }
            }
        )+
    }
}

fov_impl! { f32 f64 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_fov_roundtrip() {
        let val = fov(-1.0f32, 1.0, -0.75, 0.75);
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Fov::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded.l, val.l);
        assert_eq!(decoded.r, val.r);
        assert_eq!(decoded.b, val.b);
        assert_eq!(decoded.t, val.t);
    }
}
