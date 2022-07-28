use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Formatter,
            Debug,
            Result,
        },
        ops::{
            Add,
            Sub,
            Mul,
            Div,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
            Neg,
        },
    },
};

#[derive(Copy,Clone,Debug)]
pub struct MultiVec3<T> {
    pub r: T,
    pub x: T,pub y: T,pub z: T,
    pub xy: T,pub xz: T,pub yz: T,
    pub xyz: T,
}

impl<T> MultiVec3<T> {
    pub fn new(
        r: T,
        x: T,y: T,z: T,
        xy: T,xz: T,yz: T,
        xyz: T,
    ) -> Self {
        MultiVec3 {
            r: r,
            x: x,y: y,z: z,
            xy: xy,xz: xz,yz: yz,
            xyz: xyz,
        }
    }
}

impl<T: PartialEq> PartialEq for MultiVec3<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) &&
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) &&
        (self.xy == other.xy) && (self.xz == other.xz) && (self.yz == other.yz) &&
        (self.xyz == other.xyz)
    }
}

impl<T: Zero> Zero for MultiVec3<T> {
    const ZERO: Self = MultiVec3 { r: T::ZERO,x: T::ZERO,y: T::ZERO, z: T::ZERO, xy: T::ZERO, xz: T::ZERO, yz: T::ZERO, xyz: T::ZERO, };
}

impl<T: Zero + PartialOrd + Display> Display for MultiVec3<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        let sx = if self.x < T::ZERO {
            format!("{}x",self.x)
        } else {
            format!("+{}x",self.x)
        };
        let sy = if self.y < T::ZERO {
            format!("{}x",self.y)
        } else {
            format!("+{}x",self.y)
        };
        let sz = if self.z < T::ZERO {
            format!("{}z",self.z)
        } else {
            format!("+{}z",self.z)
        };
        let sxy = if self.xy < T::ZERO {
            format!("{}xy",self.xy)
        } else {
            format!("+{}xy",self.xy)
        };
        let sxz = if self.xz < T::ZERO {
            format!("{}xz",self.xz)
        } else {
            format!("+{}xz",self.xz)
        };
        let syz = if self.yz < T::ZERO {
            format!("{}yz",self.yz)
        } else {
            format!("+{}yz",self.yz)
        };
        let sxyz = if self.xyz < T::ZERO {
            format!("{}xyz",self.xyz)
        } else {
            format!("+{}xyz",self.xyz)
        };
        write!(f,"{}{}{}{}{}{}{}{}",self.r,sx,sy,sz,sxy,sxz,syz,sxyz)
    }
}

impl<T: Add<T,Output=T>> Add<MultiVec3<T>> for MultiVec3<T> {
    type Output = Self;
    fn add(self,other: MultiVec3<T>) -> Self {
        MultiVec3 {
            r: self.r + other.r,
            x: self.x + other.x,y: self.y + other.y,z: self.z + other.z,
            xy: self.xy + other.xy,xz: self.xz + other.xz,yz: self.yz + other.yz,
            xyz: self.xyz + other.xyz,
        }
    }
}

impl<T: AddAssign<T>> AddAssign<MultiVec3<T>> for MultiVec3<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.x += other.x; self.y += other.y; self.z += other.z;
        self.xy += other.xy; self.xz += other.xz; self.yz += other.yz;
        self.xyz += other.xyz;
    }
}

impl<T: Sub<T,Output=T>> Sub<MultiVec3<T>> for MultiVec3<T> {
    type Output = Self;
    fn sub(self,other: MultiVec3<T>) -> Self {
        MultiVec3 {
            r: self.r - other.r,
            x: self.x - other.x,y: self.y - other.y,z: self.z - other.z,
            xy: self.xy - other.xy,xz: self.xz - other.xz,yz: self.yz - other.yz,
            xyz: self.xyz - other.xyz,
        }
    }
}

impl<T: SubAssign<T>> SubAssign<MultiVec3<T>> for MultiVec3<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.x -= other.x; self.y -= other.y; self.z -= other.z;
        self.xy -= other.xy; self.xz -= other.xz; self.yz -= other.yz;
        self.xyz -= other.xyz;
    }
}

macro_rules! scalar_multivec3_mul {
    ($t:ty) => {
        impl Mul<MultiVec3<$t>> for $t {
            type Output = MultiVec3<$t>;
            fn mul(self,other: MultiVec3<$t>) -> MultiVec3<$t> {
                MultiVec3 {
                    r: self * other.r,
                    x: self * other.x,y: self * other.y,z: self * other.z,
                    xy: self * other.xy,xz: self * other.xz,yz: self * other.yz,
                    xyz: self * other.xyz,
                }
            }
        }        
    }
}

scalar_multivec3_mul!(f32);
scalar_multivec3_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for MultiVec3<T> {
    type Output = MultiVec3<T>;
    fn mul(self,other: T) -> Self {
        MultiVec3 {
            r: self.r * other,
            x: self.x * other,y: self.y * other,z: self.z * other,
            xy: self.xy * other,xz: self.xz * other,yz: self.yz * other,
            xyz: self.xyz * other,
        }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for MultiVec3<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.x *= other; self.y *= other; self.z *= other;
        self.xy *= other; self.xz *= other; self.yz *= other;
        self.xyz *= other;
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> Mul<MultiVec3<T>> for MultiVec3<T> {
    type Output = MultiVec3<T>;
    fn mul(self,other: MultiVec3<T>) -> Self {
        MultiVec3 {
            r: self.r * other.r + self.x * other.x + self.y * other.y + self.z * other.z - self.xy * other.xy - self.xz * other.xz - self.yz * other.yz - self.xyz * other.xyz,
            x: self.r * other.x + self.x * other.r - self.y * other.xy - self.z * other.xz + self.xy * other.y + self.xz * other.z - self.yz * other.xyz - self.xyz * other.yz,
            y: self.r * other.y + self.x * other.xy + self.y * other.r - self.z * other.yz - self.xy * other.x + self.xz * other.xyz + self.yz * other.z + self.xyz * other.xz,
            z: self.r * other.z + self.x * other.xz + self.y * other.yz + self.z * other.r - self.xy * other.xyz - self.xz * other.x - self.yz * other.y - self.xyz * other.xy,
            xy: self.r * other.xy + self.x * other.y - self.y * other.x + self.z * other.xyz + self.xy * other.r - self.xz * other.yz + self.yz * other.xz + self.xyz * other.z,
            xz: self.r * other.xz + self.x * other.z - self.y * other.xyz - self.z * other.x + self.xy * other.yz + self.xz * other.r - self.yz * other.xy - self.xyz * other.y,
            yz: self.r * other.yz + self.x * other.xyz + self.y * other.z - self.z * other.y - self.xy * other.xz + self.xz * other.xy + self.yz * other.r + self.xyz * other.x,
            xyz: self.r * other.xyz + self.x * other.yz + self.y * other.xz + self.z * other.xy + self.xy * other.z - self.xz * other.y + self.yz * other.x + self.xyz * other.r,
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> MulAssign<MultiVec3<T>> for MultiVec3<T> {
    fn mul_assign(&mut self,other: MultiVec3<T>) {
        let r = self.r * other.r + self.x * other.x + self.y * other.y + self.z * other.z - self.xy * other.xy - self.xz * other.xz - self.yz * other.yz - self.xyz * other.xyz;
        let x = self.r * other.x + self.x * other.r - self.y * other.xy - self.z * other.xz + self.xy * other.y + self.xz * other.z - self.yz * other.xyz - self.xyz * other.yz;
        let y = self.r * other.y + self.x * other.xy + self.y * other.r - self.z * other.yz - self.xy * other.x + self.xz * other.xyz + self.yz * other.z + self.xyz * other.xz;
        let z = self.r * other.z + self.x * other.xz + self.y * other.yz + self.z * other.r - self.xy * other.xyz - self.xz * other.x - self.yz * other.y - self.xyz * other.xy;
        let xy = self.r * other.xy + self.x * other.y - self.y * other.x + self.z * other.xyz + self.xy * other.r - self.xz * other.yz + self.yz * other.xz + self.xyz * other.z;
        let xz = self.r * other.xz + self.x * other.z - self.y * other.xyz - self.z * other.x + self.xy * other.yz + self.xz * other.r - self.yz * other.xy - self.xyz * other.y;
        let yz = self.r * other.yz + self.x * other.xyz + self.y * other.z - self.z * other.y - self.xy * other.xz + self.xz * other.xy + self.yz * other.r + self.xyz * other.x;
        let xyz = self.r * other.xyz + self.x * other.yz + self.y * other.xz + self.z * other.xy + self.xy * other.z - self.xz * other.y + self.yz * other.x + self.xyz * other.r;
        self.r = r;
        self.x = x; self.y = y; self.z = z;
        self.xy = xy; self.xz = xz; self.yz = yz;
        self.xyz = xyz;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for MultiVec3<T> {
    type Output = MultiVec3<T>;
    fn div(self,other: T) -> Self {
        MultiVec3 {
            r: self.r / other,
            x: self.x / other,y: self.y / other,z: self.z / other,
            xy: self.xy / other,xz: self.xz / other,yz: self.yz / other,
            xyz: self.xyz,
        }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for MultiVec3<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.x /= other; self.y /= other; self.z /= other;
        self.xy /= other; self.xz /= other; self.yz /= other;
        self.xyz /= other;
    }
}

impl<T: Neg<Output=T>> Neg for MultiVec3<T> {
    type Output = MultiVec3<T>;
    fn neg(self) -> MultiVec3<T> {
        MultiVec3 {
            r: -self.r,
            x: -self.x,y: -self.y,z: -self.z,
            xy: -self.xy,xz: -self.xz,yz: -self.yz,
            xyz: -self.xyz,
        }
    }
}

impl<T: Zero> From<T> for MultiVec3<T> {
    fn from(v: T) -> MultiVec3<T> {
        MultiVec3 {
            r: v,
            x: T::ZERO,y: T::ZERO,z: T::ZERO,
            xy: T::ZERO,xz: T::ZERO,yz: T::ZERO,
            xyz: T::ZERO,
        }
    }
}

impl<T: Zero> From<Complex<T>> for MultiVec3<T> {
    fn from(v: Complex<T>) -> MultiVec3<T> {
        MultiVec3 {
            r: v.r,
            x: T::ZERO,y: T::ZERO,z: T::ZERO,
            xy: T::ZERO,xz: T::ZERO,yz: T::ZERO,
            xyz: v.i,
        }
    }
}

impl<T: Zero> From<Quat<T>> for MultiVec3<T> {
    fn from(v: Quat<T>) -> MultiVec3<T> {
        MultiVec3 {
            r: v.r,
            x: T::ZERO,y: T::ZERO,z: T::ZERO,
            xy: v.i,xz: v.j,yz: v.k,
            xyz: T::ZERO,
        }
    }
}

impl<T: Zero> From<Vec3<T>> for MultiVec3<T> {
    fn from(v: Vec3<T>) -> MultiVec3<T> {
        MultiVec3 {
            r: T::ZERO,
            x: v.x,y: v.y,z: v.z,
            xy: T::ZERO,xz: T::ZERO,yz: T::ZERO,
            xyz: T::ZERO,
        }
    }
}
