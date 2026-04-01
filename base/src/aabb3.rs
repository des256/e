use {
    crate::*,
    std::{
        fmt::{Display, Formatter, Result},
        ops::{Add, Div, Sub},
    },
};

/// 3D axis-aligned bounding box (min/max corners).
///
/// # Examples
///
/// ```
/// use base::*;
///
/// let bb = aabb3(vec3(0.0f32, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
/// assert!(bb.contains(vec3(0.5, 0.5, 0.5)));
/// assert!(!bb.contains(vec3(2.0, 0.5, 0.5)));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Codec)]
pub struct Aabb3<T> {
    /// Minimum corner.
    pub min: Vec3<T>,
    /// Maximum corner.
    pub max: Vec3<T>,
}

/// Create a new 3D AABB from min and max corners.
pub const fn aabb3<T>(min: Vec3<T>, max: Vec3<T>) -> Aabb3<T> {
    Aabb3 { min, max }
}

impl<T: Display> Display for Aabb3<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(
            f,
            "[({},{},{})..({},{},{})]",
            self.min.x, self.min.y, self.min.z, self.max.x, self.max.y, self.max.z
        )
    }
}

impl<T> Aabb3<T>
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T>,
{
    /// Test if point is inside the AABB.
    pub fn contains(&self, p: Vec3<T>) -> bool {
        p.x >= self.min.x && p.x <= self.max.x
            && p.y >= self.min.y && p.y <= self.max.y
            && p.z >= self.min.z && p.z <= self.max.z
    }

    /// Test if two AABBs overlap.
    pub fn intersects(&self, other: &Aabb3<T>) -> bool {
        self.min.x <= other.max.x && self.max.x >= other.min.x
            && self.min.y <= other.max.y && self.max.y >= other.min.y
            && self.min.z <= other.max.z && self.max.z >= other.min.z
    }

    /// Intersection of two AABBs (returns None if disjoint).
    pub fn intersection(&self, other: &Aabb3<T>) -> Option<Aabb3<T>> {
        let min_x = if self.min.x > other.min.x { self.min.x } else { other.min.x };
        let min_y = if self.min.y > other.min.y { self.min.y } else { other.min.y };
        let min_z = if self.min.z > other.min.z { self.min.z } else { other.min.z };
        let max_x = if self.max.x < other.max.x { self.max.x } else { other.max.x };
        let max_y = if self.max.y < other.max.y { self.max.y } else { other.max.y };
        let max_z = if self.max.z < other.max.z { self.max.z } else { other.max.z };
        if max_x >= min_x && max_y >= min_y && max_z >= min_z {
            Some(Aabb3 {
                min: Vec3 { x: min_x, y: min_y, z: min_z },
                max: Vec3 { x: max_x, y: max_y, z: max_z },
            })
        } else {
            None
        }
    }

    /// Smallest AABB containing both.
    pub fn union(&self, other: &Aabb3<T>) -> Aabb3<T> {
        Aabb3 {
            min: Vec3 {
                x: if self.min.x < other.min.x { self.min.x } else { other.min.x },
                y: if self.min.y < other.min.y { self.min.y } else { other.min.y },
                z: if self.min.z < other.min.z { self.min.z } else { other.min.z },
            },
            max: Vec3 {
                x: if self.max.x > other.max.x { self.max.x } else { other.max.x },
                y: if self.max.y > other.max.y { self.max.y } else { other.max.y },
                z: if self.max.z > other.max.z { self.max.z } else { other.max.z },
            },
        }
    }

    /// Expand AABB to include a point.
    pub fn expand(&self, p: Vec3<T>) -> Aabb3<T> {
        Aabb3 {
            min: Vec3 {
                x: if p.x < self.min.x { p.x } else { self.min.x },
                y: if p.y < self.min.y { p.y } else { self.min.y },
                z: if p.z < self.min.z { p.z } else { self.min.z },
            },
            max: Vec3 {
                x: if p.x > self.max.x { p.x } else { self.max.x },
                y: if p.y > self.max.y { p.y } else { self.max.y },
                z: if p.z > self.max.z { p.z } else { self.max.z },
            },
        }
    }

    /// Size of the AABB.
    pub fn size(&self) -> Vec3<T> {
        Vec3 {
            x: self.max.x - self.min.x,
            y: self.max.y - self.min.y,
            z: self.max.z - self.min.z,
        }
    }
}

impl<T> Aabb3<T>
where
    T: Copy + PartialOrd + Add<Output = T> + Sub<Output = T> + Div<Output = T> + One,
{
    /// Center point of the AABB.
    pub fn center(&self) -> Vec3<T> {
        let two = T::ONE + T::ONE;
        Vec3 {
            x: (self.min.x + self.max.x) / two,
            y: (self.min.y + self.max.y) / two,
            z: (self.min.z + self.max.z) / two,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aabb3_contains() {
        let bb = aabb3(vec3(0.0f32, 0.0, 0.0), vec3(10.0, 10.0, 10.0));
        assert!(bb.contains(vec3(5.0, 5.0, 5.0)));
        assert!(!bb.contains(vec3(11.0, 5.0, 5.0)));
    }

    #[test]
    fn test_aabb3_intersection() {
        let a = aabb3(vec3(0.0f32, 0.0, 0.0), vec3(10.0, 10.0, 10.0));
        let b = aabb3(vec3(5.0, 5.0, 5.0), vec3(15.0, 15.0, 15.0));
        let i = a.intersection(&b).unwrap();
        assert_eq!(i.min, vec3(5.0, 5.0, 5.0));
        assert_eq!(i.max, vec3(10.0, 10.0, 10.0));
    }

    #[test]
    fn test_aabb3_center_and_size() {
        let bb = aabb3(vec3(0.0f32, 0.0, 0.0), vec3(10.0, 10.0, 10.0));
        assert_eq!(bb.center(), vec3(5.0, 5.0, 5.0));
        assert_eq!(bb.size(), vec3(10.0, 10.0, 10.0));
    }

    #[test]
    fn test_aabb3_intersects() {
        let a = aabb3(vec3(0.0f32, 0.0, 0.0), vec3(5.0, 5.0, 5.0));
        let b = aabb3(vec3(4.0, 4.0, 4.0), vec3(10.0, 10.0, 10.0));
        let c = aabb3(vec3(6.0, 6.0, 6.0), vec3(10.0, 10.0, 10.0));
        assert!(a.intersects(&b));
        assert!(!a.intersects(&c));
    }

    #[test]
    fn test_codec_aabb3_roundtrip() {
        let val = Aabb3 {
            min: vec3(0.0f32, 0.0, 0.0),
            max: vec3(10.0, 10.0, 10.0),
        };
        let mut buf = Vec::new();
        val.encode(&mut buf);
        let (decoded, len) = Aabb3::<f32>::decode(&buf).unwrap();
        assert_eq!(buf.len(), len);
        assert_eq!(decoded, val);
    }
}
