use {
    crate::*,
    codec::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{Add, Div, Sub},
    },
};

/// 2D axis-aligned bounding box (min/max corners).
///
/// Well suited for collision detection and spatial queries where min/max
/// comparisons dominate.
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let a = aabb2(vec2(0.0f32, 0.0), vec2(5.0, 5.0));
/// let b = aabb2(vec2(3.0, 3.0), vec2(8.0, 8.0));
/// assert!(a.intersects(&b));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Aabb2<T> {
    /// Minimum corner.
    pub min: Vec2<T>,
    /// Maximum corner.
    pub max: Vec2<T>,
}

/// Create a new 2D AABB from min and max corners.
pub const fn aabb2<T>(min: Vec2<T>, max: Vec2<T>) -> Aabb2<T> {
    Aabb2 { min, max }
}

impl<T: Display> Display for Aabb2<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "[({},{})..({},{})]",
            self.min.x, self.min.y, self.max.x, self.max.y
        )
    }
}

impl<T> Aabb2<T>
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T>,
{
    /// Test if point is inside the AABB.
    pub fn contains(&self, p: Vec2<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x && p.y >= self.min.y && p.y <= self.max.y
    }

    /// Test if two AABBs overlap.
    pub fn intersects(&self, other: &Aabb2<T>) -> bool {
        self.min.x <= other.max.x
            && self.max.x >= other.min.x
            && self.min.y <= other.max.y
            && self.max.y >= other.min.y
    }

    /// Intersection of two AABBs (returns None if disjoint).
    pub fn intersection(&self, other: &Aabb2<T>) -> Option<Aabb2<T>> {
        let min_x = if self.min.x > other.min.x {
            self.min.x
        } else {
            other.min.x
        };
        let min_y = if self.min.y > other.min.y {
            self.min.y
        } else {
            other.min.y
        };
        let max_x = if self.max.x < other.max.x {
            self.max.x
        } else {
            other.max.x
        };
        let max_y = if self.max.y < other.max.y {
            self.max.y
        } else {
            other.max.y
        };
        if max_x >= min_x && max_y >= min_y {
            Some(Aabb2 {
                min: Vec2 { x: min_x, y: min_y },
                max: Vec2 { x: max_x, y: max_y },
            })
        } else {
            None
        }
    }

    /// Smallest AABB containing both.
    pub fn union(&self, other: &Aabb2<T>) -> Aabb2<T> {
        Aabb2 {
            min: Vec2 {
                x: if self.min.x < other.min.x {
                    self.min.x
                } else {
                    other.min.x
                },
                y: if self.min.y < other.min.y {
                    self.min.y
                } else {
                    other.min.y
                },
            },
            max: Vec2 {
                x: if self.max.x > other.max.x {
                    self.max.x
                } else {
                    other.max.x
                },
                y: if self.max.y > other.max.y {
                    self.max.y
                } else {
                    other.max.y
                },
            },
        }
    }

    /// Expand AABB to include a point.
    pub fn expand(&self, p: Vec2<T>) -> Aabb2<T> {
        Aabb2 {
            min: Vec2 {
                x: if p.x < self.min.x { p.x } else { self.min.x },
                y: if p.y < self.min.y { p.y } else { self.min.y },
            },
            max: Vec2 {
                x: if p.x > self.max.x { p.x } else { self.max.x },
                y: if p.y > self.max.y { p.y } else { self.max.y },
            },
        }
    }

    /// Size of the AABB.
    pub fn size(&self) -> Vec2<T> {
        Vec2 {
            x: self.max.x - self.min.x,
            y: self.max.y - self.min.y,
        }
    }
}

impl<T> Aabb2<T>
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Div<Output = T> + One,
{
    /// Center point of the AABB.
    pub fn center(&self) -> Vec2<T> {
        let two = T::ONE + T::ONE;
        Vec2 {
            x: (self.min.x + self.max.x) / two,
            y: (self.min.y + self.max.y) / two,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb2_contains() {
        let bb = aabb2(vec2(0.0f32, 0.0), vec2(10.0, 10.0));
        assert!(bb.contains(vec2(5.0, 5.0)));
        assert!(!bb.contains(vec2(11.0, 5.0)));
        assert!(bb.contains(vec2(0.0, 0.0)));
        assert!(bb.contains(vec2(10.0, 10.0)));
    }

    #[test]
    fn test_aabb2_intersection() {
        let a = aabb2(vec2(0.0f32, 0.0), vec2(10.0, 10.0));
        let b = aabb2(vec2(5.0, 5.0), vec2(15.0, 15.0));
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.min, vec2(5.0, 5.0));
        assert_eq!(i.max, vec2(10.0, 10.0));
    }

    #[test]
    fn test_aabb2_no_intersection() {
        let a = aabb2(vec2(0.0f32, 0.0), vec2(1.0, 1.0));
        let b = aabb2(vec2(5.0, 5.0), vec2(6.0, 6.0));
        assert!(a.intersection(&b).is_none());
    }

    #[test]
    fn test_aabb2_union() {
        let a = aabb2(vec2(0.0f32, 0.0), vec2(5.0, 5.0));
        let b = aabb2(vec2(3.0, 3.0), vec2(10.0, 10.0));
        let u = a.union(&b);
        assert_eq!(u.min, vec2(0.0, 0.0));
        assert_eq!(u.max, vec2(10.0, 10.0));
    }

    #[test]
    fn test_aabb2_center_and_size() {
        let bb = aabb2(vec2(2.0f32, 4.0), vec2(8.0, 10.0));
        assert_eq!(bb.center(), vec2(5.0, 7.0));
        assert_eq!(bb.size(), vec2(6.0, 6.0));
    }

    #[test]
    fn test_aabb2_expand() {
        let bb = aabb2(vec2(0.0f32, 0.0), vec2(5.0, 5.0));
        let expanded = bb.expand(vec2(10.0, -3.0));
        assert_eq!(expanded.min, vec2(0.0, -3.0));
        assert_eq!(expanded.max, vec2(10.0, 5.0));
    }

    #[test]
    fn test_codec_aabb2_roundtrip() {
        let val = Aabb2 {
            min: vec2(0.0f32, 0.0),
            max: vec2(10.0, 10.0),
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Aabb2::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
