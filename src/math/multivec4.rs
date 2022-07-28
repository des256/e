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
pub struct MultiVec4<T> {
    pub r: T,
    pub x: T,pub y: T,pub z: T,pub w: T,
    pub xy: T,pub xz: T,pub xw: T,pub yz: T,pub yw: T,pub zw: T,
    pub xyz: T,pub xzw: T,pub xyw: T,pub yzw: T,
    pub xyzw: T,
}

impl<T> MultiVec4<T> {
    pub fn new(
        r: T,
        x: T,y: T,z: T,w: T,
        xy: T,xz: T,xw: T,yz: T,yw: T,zw: T,
        xyz: T,xzw: T,xyw: T,yzw: T,
        xyzw: T
    ) -> Self {
        MultiVec4 {
            r: r,
            x: x,y: y,z: z,w: w,
            xy: xy,xz: xz,xw: xw,yz: yz,yw: yw,zw: zw,
            xyz: xyz,xyw: xyw,xzw: xzw,yzw: yzw,
            xyzw: xyzw,
        }
    }
}

impl<T: PartialEq> PartialEq for MultiVec4<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) &&
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w) &&
        (self.xy == other.xy) && (self.xz == other.xz) && (self.xw == other.xw) && (self.yz == other.yz) && (self.yw == other.yw) && (self.zw == other.zw) &&
        (self.xyz == other.xyz) && (self.xyw == other.xyw) && (self.xzw == other.xzw) && (self.yzw == other.yzw) &&
        (self.xyzw == other.xyzw)
    }
}

impl<T: Zero> Zero for MultiVec4<T> {
    const ZERO: Self = MultiVec4 { r: T::ZERO,x: T::ZERO,y: T::ZERO, z: T::ZERO, w: T::ZERO, xy: T::ZERO, xz: T::ZERO, xw: T::ZERO, yz: T::ZERO, yw: T::ZERO, zw: T::ZERO, xyz: T::ZERO, xyw: T::ZERO, xzw: T::ZERO, yzw: T::ZERO, xyzw: T::ZERO, };
}

impl<T: Zero + PartialOrd + Display> Display for MultiVec4<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        let sx = if self.x < T::ZERO {
            format!("{}x",self.x)
        } else {
            format!("+{}x",self.x)
        };
        let sy = if self.y < T::ZERO {
            format!("{}y",self.y)
        } else {
            format!("+{}y",self.y)
        };
        let sz = if self.z < T::ZERO {
            format!("{}z",self.z)
        } else {
            format!("+{}z",self.z)
        };
        let sw = if self.w < T::ZERO {
            format!("{}w",self.w)
        } else {
            format!("+{}w",self.w)
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
        let sxw = if self.xw < T::ZERO {
            format!("{}xw",self.xw)
        } else {
            format!("+{}xw",self.xw)
        };
        let syz = if self.yz < T::ZERO {
            format!("{}yz",self.yz)
        } else {
            format!("+{}yz",self.yz)
        };
        let syw = if self.yw < T::ZERO {
            format!("{}yw",self.yw)
        } else {
            format!("+{}yw",self.yw)
        };
        let szw = if self.zw < T::ZERO {
            format!("{}zw",self.zw)
        } else {
            format!("+{}zw",self.zw)
        };
        let sxyz = if self.xyz < T::ZERO {
            format!("{}xyz",self.xyz)
        } else {
            format!("+{}xyz",self.xyz)
        };
        let sxyw = if self.xyw < T::ZERO {
            format!("{}xyw",self.xyw)
        } else {
            format!("+{}xyw",self.xyw)
        };
        let sxzw = if self.xzw < T::ZERO {
            format!("{}xzw",self.xzw)
        } else {
            format!("+{}xzw",self.xzw)
        };
        let syzw = if self.yzw < T::ZERO {
            format!("{}yzw",self.yzw)
        } else {
            format!("+{}yzw",self.yzw)
        };
        let sxyzw = if self.xyzw < T::ZERO {
            format!("{}xyzw",self.xyzw)
        } else {
            format!("+{}xyzw",self.xyzw)
        };
        write!(f,"{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}",self.r,sx,sy,sz,sw,sxy,sxz,sxw,syz,syw,szw,sxyz,sxyw,sxzw,syzw,sxyzw)
    }
}

impl<T: Add<T,Output=T>> Add<MultiVec4<T>> for MultiVec4<T> {
    type Output = Self;
    fn add(self,other: MultiVec4<T>) -> Self {
        MultiVec4 {
            r: self.r + other.r,
            x: self.x + other.x,y: self.y + other.y,z: self.z + other.z,w: self.w + other.w,
            xy: self.xy + other.xy,xz: self.xz + other.xz,xw: self.xw + other.xw,yz: self.yz + other.yz,yw: self.yw + other.yw,zw: self.zw + other.zw,
            xyz: self.xyz + other.xyz,xyw: self.xyw + other.xyw,xzw: self.xzw + other.xzw,yzw: self.yzw + other.yzw,
            xyzw: self.xyzw + other.xyzw,
        }
    }
}

impl<T: AddAssign<T>> AddAssign<MultiVec4<T>> for MultiVec4<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.x += other.x; self.y += other.y; self.z += other.z; self.w += other.w;
        self.xy += other.xy; self.xz += other.xz; self.xw += other.xw; self.yz += other.yz; self.yw += other.yw; self.zw += other.zw;
        self.xyz += other.xyz; self.xyw += other.xyw; self.xzw += other.xzw; self.yzw += other.yzw;
        self.xyzw += other.xyzw;
    }
}

impl<T: Sub<T,Output=T>> Sub<MultiVec4<T>> for MultiVec4<T> {
    type Output = Self;
    fn sub(self,other: MultiVec4<T>) -> Self {
        MultiVec4 {
            r: self.r - other.r,
            x: self.x - other.x,y: self.y - other.y,z: self.z - other.z,w: self.w - other.w,
            xy: self.xy - other.xy,xz: self.xz - other.xz,xw: self.xw - other.xw,yz: self.yz - other.yz,yw: self.yw - other.yw,zw: self.zw - other.zw,
            xyz: self.xyz - other.xyz,xyw: self.xyw - other.xyw,xzw: self.xzw - other.xzw,yzw: self.yzw - other.yzw,
            xyzw: self.xyzw - other.xyzw,
        }
    }
}

impl<T: SubAssign<T>> SubAssign<MultiVec4<T>> for MultiVec4<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.x -= other.x; self.y -= other.y; self.z -= other.z; self.w -= other.w;
        self.xy -= other.xy; self.xz -= other.xz; self.xw -= other.xw; self.yz -= other.yz; self.yw -= other.yw; self.zw -= other.zw;
        self.xyz -= other.xyz; self.xyw -= other.xyw; self.xzw -= other.xzw; self.yzw -= other.yzw;
        self.xyzw -= other.xyzw;
    }
}

macro_rules! scalar_multivec4_mul {
    ($t:ty) => {
        impl Mul<MultiVec4<$t>> for $t {
            type Output = MultiVec4<$t>;
            fn mul(self,other: MultiVec4<$t>) -> MultiVec4<$t> {
                MultiVec4 {
                    r: self * other.r,
                    x: self * other.x,y: self * other.y,z: self * other.z,w: self * other.w,
                    xy: self * other.xy,xz: self * other.xz,xw: self * other.xw,yz: self * other.yz,yw: self * other.yw,zw: self * other.zw,
                    xyz: self * other.xyz,xyw: self * other.xyw,xzw: self * other.xzw,yzw: self * other.yzw,
                    xyzw: self * other.xyzw,
                }
            }
        }        
    }
}

scalar_multivec4_mul!(f32);
scalar_multivec4_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for MultiVec4<T> {
    type Output = MultiVec4<T>;
    fn mul(self,other: T) -> Self {
        MultiVec4 {
            r: self.r * other,
            x: self.x * other,y: self.y * other,z: self.z * other,w: self.w * other,
            xy: self.xy * other,xz: self.xz * other,xw: self.xw * other,yz: self.yz * other,yw: self.yw * other,zw: self.zw * other,
            xyz: self.xyz * other,xyw: self.xyw * other,xzw: self.xzw * other,yzw: self.yzw * other,
            xyzw: self.xyzw * other,
        }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for MultiVec4<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.x *= other; self.y *= other; self.z *= other; self.w *= other;
        self.xy *= other; self.xz *= other; self.xw *= other; self.yz *= other; self.yw *= other; self.zw *= other;
        self.xyz *= other; self.xyw *= other; self.xzw *= other; self.yzw *= other;
        self.xyzw *= other;
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> Mul<MultiVec4<T>> for MultiVec4<T> {
    type Output = MultiVec4<T>;
    fn mul(self,other: MultiVec4<T>) -> Self {
        MultiVec4 {
            r: self.r * other.r + self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w - self.xy * other.xy - self.xz * other.xz - self.xw * other.xw - self.yz * other.yz - self.yw * other.yw - self.zw * other.zw - self.xyz * other.xyz - self.xyw * other.xyw - self.xzw * other.xzw - self.yzw * other.yzw + self.xyzw * other.xyzw,
            x: self.r * other.x + self.x * other.r - self.y * other.xy - self.z * other.xz - self.w * other.xw + self.xy * other.y + self.xz * other.z + self.xw * other.w - self.yz * other.xyz - self.yw * other.xyw - self.zw * other.xzw - self.xyz * other.yz - self.xyw * other.yw - self.xzw * other.zw + self.yzw * other.xyzw - self.xyzw * other.yzw,
            y: self.r * other.y + self.x * other.xy + self.y * other.r - self.z * other.yz - self.w * other.yw - self.xy * other.x + self.xz * other.xyz + self.xw * other.xyw + self.yz * other.z + self.yw * other.w - self.zw * other.yzw + self.xyz * other.xz + self.xyw * other.xw - self.xzw * other.xyzw - self.yzw * other.zw + self.xyzw * other.xzw,
            z: self.r * other.z + self.x * other.xz + self.y * other.yz + self.z * other.r - self.w * other.zw - self.xy * other.xyz - self.xz * other.x + self.xw * other.xzw - self.yz * other.y + self.yw * other.yzw + self.zw * other.w - self.xyz * other.xy + self.xyw * other.xyzw + self.xzw * other.xw + self.yzw * other.yw - self.xyzw * other.xyw,
            w: self.r * other.w + self.x * other.xw + self.y * other.yw + self.z * other.zw + self.w * other.r - self.xy * other.xyw - self.xz * other.xzw - self.xw * other.x - self.yz * other.yzw - self.yw * other.y - self.zw * other.z - self.xyz * other.xyzw - self.xyw * other.xy - self.xzw * other.xz - self.yzw * other.yz + self.xyzw * other.xyz,
            xy: self.r * other.xy + self.x * other.y - self.y * other.x + self.z * other.xyz + self.w * other.xyw + self.xy * other.r - self.xz * other.yz - self.xw * other.yw + self.yz * other.xz + self.yw * other.xw - self.zw * other.xyzw + self.xyz * other.z + self.xyw * other.w - self.xzw * other.yzw + self.yzw * other.xzw - self.xyzw * other.zw,
            xz: self.r * other.xz + self.x * other.z - self.y * other.xyz - self.z * other.x + self.w * other.xzw + self.xy * other.yz + self.xz * other.r - self.xw * other.zw - self.yz * other.xy + self.yw * other.xyzw + self.zw * other.xw - self.xyz * other.y + self.xyw * other.yzw + self.xzw * other.w - self.yzw * other.xyw + self.xyzw * other.yw,
            xw: self.r * other.xw + self.x * other.w - self.y * other.xyw - self.z * other.xzw - self.w * other.x + self.xy * other.yw + self.xz * other.zw + self.xw * other.r - self.yz * other.xyzw - self.yw * other.xy - self.zw * other.xz - self.xyz * other.yzw - self.xyw * other.y - self.xzw * other.z + self.yzw * other.xyz - self.xyzw * other.yz,
            yz: self.r * other.yz + self.x * other.xyz + self.y * other.z - self.z * other.y + self.w * other.yzw - self.xy * other.xz + self.xz * other.xy - self.xw * other.xyzw + self.yz * other.r - self.yw * other.zw + self.zw * other.yw + self.xyz * other.x - self.xyw * other.xzw + self.xzw * other.xyw + self.yzw * other.w - self.xyzw * other.xw,
            yw: self.r * other.yw + self.x * other.xyw + self.y * other.w - self.z * other.yzw - self.w * other.y - self.xy * other.xw + self.xz * other.xyzw + self.xw * other.xy + self.yz * other.zw + self.yw * other.r - self.zw * other.yz + self.xyz * other.xzw + self.xyw * other.x - self.xzw * other.xyz - self.yzw * other.z + self.xyzw * other.xz,
            zw: self.r * other.zw + self.x * other.xzw + self.y * other.yzw + self.z * other.w - self.w * other.z - self.xy * other.xyzw - self.xz * other.xw + self.xw * other.xz - self.yz * other.yw + self.yw * other.yz + self.zw * other.r - self.xyz * other.xyw + self.xyw * other.xyz + self.xzw * other.x + self.yzw * other.y - self.xyzw * other.xy,
            xyz: self.r * other.xyz + self.x * other.yz - self.y * other.xz + self.z * other.xy - self.w * other.xyzw + self.xy * other.z - self.xz * other.y + self.xw * other.yzw + self.yz * other.x - self.yw * other.xzw + self.zw * other.xyw + self.xyz * other.r - self.xyw * other.zw + self.xzw * other.yw - self.yzw * other.xw + self.xyzw * other.w,
            xyw: self.r * other.xyw + self.x * other.yw - self.y * other.xw + self.z * other.xyzw + self.w * other.xy + self.xy * other.w - self.xz * other.yzw - self.xw * other.y + self.yz * other.xzw + self.yw * other.x - self.zw * other.xyz + self.xyz * other.zw + self.xyw * other.r - self.xzw * other.yz + self.yzw * other.xz - self.xyzw * other.z,
            xzw: self.r * other.xzw + self.x * other.zw - self.y * other.xyzw - self.z * other.xw + self.w * other.xz + self.xy * other.yzw + self.xz * other.w - self.xw * other.z - self.yz * other.xyw + self.yw * other.xyz + self.zw * other.x - self.xyz * other.yw + self.xyw * other.yz + self.xzw * other.r - self.yzw * other.xy + self.xyzw * other.y,
            yzw: self.r * other.yzw + self.x * other.xyzw + self.y * other.zw - self.z * other.yw + self.w * other.yz - self.xy * other.xzw + self.xz * other.xyw - self.xw * other.xyz + self.yz * other.w - self.yw * other.z + self.zw * other.y + self.xyz * other.xw - self.xyw * other.xz + self.xzw * other.xy + self.yzw * other.r - self.xyzw * other.x,
            xyzw: self.r * other.xyzw + self.x * other.yzw - self.y * other.xzw + self.z * other.xyw - self.w * other.xyz + self.xy * other.zw - self.xz * other.yw + self.xw * other.yz + self.yz * other.xw - self.yw * other.xz + self.zw * other.xy + self.xyz * other.w - self.xyw * other.z + self.xzw * other.y - self.yzw * other.x + self.xyzw * other.r,
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> MulAssign<MultiVec4<T>> for MultiVec4<T> {
    fn mul_assign(&mut self,other: MultiVec4<T>) {
        let r = self.r * other.r + self.x * other.x + self.y * other.y + self.z * other.z + self.w * other.w - self.xy * other.xy - self.xz * other.xz - self.xw * other.xw - self.yz * other.yz - self.yw * other.yw - self.zw * other.zw - self.xyz * other.xyz - self.xyw * other.xyw - self.xzw * other.xzw - self.yzw * other.yzw + self.xyzw * other.xyzw;
        let x = self.r * other.x + self.x * other.r - self.y * other.xy - self.z * other.xz - self.w * other.xw + self.xy * other.y + self.xz * other.z + self.xw * other.w - self.yz * other.xyz - self.yw * other.xyw - self.zw * other.xzw - self.xyz * other.yz - self.xyw * other.yw - self.xzw * other.zw + self.yzw * other.xyzw - self.xyzw * other.yzw;
        let y = self.r * other.y + self.x * other.xy + self.y * other.r - self.z * other.yz - self.w * other.yw - self.xy * other.x + self.xz * other.xyz + self.xw * other.xyw + self.yz * other.z + self.yw * other.w - self.zw * other.yzw + self.xyz * other.xz + self.xyw * other.xw - self.xzw * other.xyzw - self.yzw * other.zw + self.xyzw * other.xzw;
        let z = self.r * other.z + self.x * other.xz + self.y * other.yz + self.z * other.r - self.w * other.zw - self.xy * other.xyz - self.xz * other.x + self.xw * other.xzw - self.yz * other.y + self.yw * other.yzw + self.zw * other.w - self.xyz * other.xy + self.xyw * other.xyzw + self.xzw * other.xw + self.yzw * other.yw - self.xyzw * other.xyw;
        let w = self.r * other.w + self.x * other.xw + self.y * other.yw + self.z * other.zw + self.w * other.r - self.xy * other.xyw - self.xz * other.xzw - self.xw * other.x - self.yz * other.yzw - self.yw * other.y - self.zw * other.z - self.xyz * other.xyzw - self.xyw * other.xy - self.xzw * other.xz - self.yzw * other.yz + self.xyzw * other.xyz;
        let xy = self.r * other.xy + self.x * other.y - self.y * other.x + self.z * other.xyz + self.w * other.xyw + self.xy * other.r - self.xz * other.yz - self.xw * other.yw + self.yz * other.xz + self.yw * other.xw - self.zw * other.xyzw + self.xyz * other.z + self.xyw * other.w - self.xzw * other.yzw + self.yzw * other.xzw - self.xyzw * other.zw;
        let xz = self.r * other.xz + self.x * other.z - self.y * other.xyz - self.z * other.x + self.w * other.xzw + self.xy * other.yz + self.xz * other.r - self.xw * other.zw - self.yz * other.xy + self.yw * other.xyzw + self.zw * other.xw - self.xyz * other.y + self.xyw * other.yzw + self.xzw * other.w - self.yzw * other.xyw + self.xyzw * other.yw;
        let xw = self.r * other.xw + self.x * other.w - self.y * other.xyw - self.z * other.xzw - self.w * other.x + self.xy * other.yw + self.xz * other.zw + self.xw * other.r - self.yz * other.xyzw - self.yw * other.xy - self.zw * other.xz - self.xyz * other.yzw - self.xyw * other.y - self.xzw * other.z + self.yzw * other.xyz - self.xyzw * other.yz;
        let yz = self.r * other.yz + self.x * other.xyz + self.y * other.z - self.z * other.y + self.w * other.yzw - self.xy * other.xz + self.xz * other.xy - self.xw * other.xyzw + self.yz * other.r - self.yw * other.zw + self.zw * other.yw + self.xyz * other.x - self.xyw * other.xzw + self.xzw * other.xyw + self.yzw * other.w - self.xyzw * other.xw;
        let yw = self.r * other.yw + self.x * other.xyw + self.y * other.w - self.z * other.yzw - self.w * other.y - self.xy * other.xw + self.xz * other.xyzw + self.xw * other.xy + self.yz * other.zw + self.yw * other.r - self.zw * other.yz + self.xyz * other.xzw + self.xyw * other.x - self.xzw * other.xyz - self.yzw * other.z + self.xyzw * other.xz;
        let zw = self.r * other.zw + self.x * other.xzw + self.y * other.yzw + self.z * other.w - self.w * other.z - self.xy * other.xyzw - self.xz * other.xw + self.xw * other.xz - self.yz * other.yw + self.yw * other.yz + self.zw * other.r - self.xyz * other.xyw + self.xyw * other.xyz + self.xzw * other.x + self.yzw * other.y - self.xyzw * other.xy;
        let xyz = self.r * other.xyz + self.x * other.yz - self.y * other.xz + self.z * other.xy - self.w * other.xyzw + self.xy * other.z - self.xz * other.y + self.xw * other.yzw + self.yz * other.x - self.yw * other.xzw + self.zw * other.xyw + self.xyz * other.r - self.xyw * other.zw + self.xzw * other.yw - self.yzw * other.xw + self.xyzw * other.w;
        let xyw = self.r * other.xyw + self.x * other.yw - self.y * other.xw + self.z * other.xyzw + self.w * other.xy + self.xy * other.w - self.xz * other.yzw - self.xw * other.y + self.yz * other.xzw + self.yw * other.x - self.zw * other.xyz + self.xyz * other.zw + self.xyw * other.r - self.xzw * other.yz + self.yzw * other.xz - self.xyzw * other.z;
        let xzw = self.r * other.xzw + self.x * other.zw - self.y * other.xyzw - self.z * other.xw + self.w * other.xz + self.xy * other.yzw + self.xz * other.w - self.xw * other.z - self.yz * other.xyw + self.yw * other.xyz + self.zw * other.x - self.xyz * other.yw + self.xyw * other.yz + self.xzw * other.r - self.yzw * other.xy + self.xyzw * other.y;
        let yzw = self.r * other.yzw + self.x * other.xyzw + self.y * other.zw - self.z * other.yw + self.w * other.yz - self.xy * other.xzw + self.xz * other.xyw - self.xw * other.xyz + self.yz * other.w - self.yw * other.z + self.zw * other.y + self.xyz * other.xw - self.xyw * other.xz + self.xzw * other.xy + self.yzw * other.r - self.xyzw * other.x;
        let xyzw = self.r * other.xyzw + self.x * other.yzw - self.y * other.xzw + self.z * other.xyw - self.w * other.xyz + self.xy * other.zw - self.xz * other.yw + self.xw * other.yz + self.yz * other.xw - self.yw * other.xz + self.zw * other.xy + self.xyz * other.w - self.xyw * other.z + self.xzw * other.y - self.yzw * other.x + self.xyzw * other.r;
        self.r = r;
        self.x = x; self.y = y; self.z = z; self.w = w;
        self.xy = xy; self.xz = xz;self.xw = xw; self.yz = yz; self.yw = yw; self.zw = zw;
        self.xyz = xyz; self.xyw = xyw; self.xzw = xzw; self.yzw = yzw;
        self.xyzw = xyzw;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for MultiVec4<T> {
    type Output = MultiVec4<T>;
    fn div(self,other: T) -> Self {
        MultiVec4 {
            r: self.r / other,
            x: self.x / other,y: self.y / other,z: self.z / other,w: self.w / other,
            xy: self.xy / other,xz: self.xz / other,xw: self.xw / other,yz: self.yz / other,yw: self.yw / other,zw: self.zw / other,
            xyz: self.xyz / other,xyw: self.xyw / other,xzw: self.xzw / other,yzw: self.yzw / other,
            xyzw: self.xyzw / other,
        }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for MultiVec4<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.x /= other; self.y /= other; self.z /= other; self.w /= other;
        self.xy /= other; self.xz /= other; self.xw /= other; self.yz /= other; self.yw /= other; self.zw /= other;
        self.xyz /= other; self.xyw /= other; self.xzw /= other; self.yzw /= other;
        self.xyzw /= other;
    }
}

impl<T: Neg<Output=T>> Neg for MultiVec4<T> {
    type Output = MultiVec4<T>;
    fn neg(self) -> MultiVec4<T> {
        MultiVec4 {
            r: -self.r,
            x: -self.x,y: -self.y,z: -self.z,w: -self.w,
            xy: -self.xy,xz: -self.xz,xw: -self.xw,yz: -self.yz,yw: -self.yw,zw: -self.zw,
            xyz: -self.xyz,xyw: -self.xyw,xzw: -self.xzw,yzw: -self.yzw,
            xyzw: -self.xyzw,
        }
    }
}

impl<T: Zero> From<T> for MultiVec4<T> {
    fn from(v: T) -> MultiVec4<T> {
        MultiVec4 {
            r: v,
            x: T::ZERO,y: T::ZERO,z: T::ZERO,w: T::ZERO,
            xy: T::ZERO,xz: T::ZERO,xw: T::ZERO,yz: T::ZERO,yw: T::ZERO,zw: T::ZERO,
            xyz: T::ZERO,xyw: T::ZERO,xzw: T::ZERO,yzw: T::ZERO,
            xyzw: T::ZERO,
        }
    }
}

impl<T: Zero> From<Vec4<T>> for MultiVec4<T> {
    fn from(v: Vec4<T>) -> MultiVec4<T> {
        MultiVec4 {
            r: T::ZERO,
            x: v.x,y: v.y,z: v.z,w: v.w,
            xy: T::ZERO,xz: T::ZERO,xw: T::ZERO,yz: T::ZERO,yw: T::ZERO,zw: T::ZERO,
            xyz: T::ZERO,xyw: T::ZERO,xzw: T::ZERO,yzw: T::ZERO,
            xyzw: T::ZERO,
        }
    }
}
