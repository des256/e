use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Formatter,
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
pub struct Mat4x3<T> {
    pub x: Vec3<T>,
    pub y: Vec3<T>,
    pub z: Vec3<T>,
    pub w: Vec3<T>,
}

impl<T: Copy> Mat4x3<T> {
    pub fn new(
        xx: T,xy: T,xz: T,
        yx: T,yy: T,yz: T,
        zx: T,zy: T,zz: T,
        wx: T,wy: T,wz: T,
    ) -> Self {
        Mat4x3 {
            x: Vec3 { x: xx,y: xy,z: xz, },
            y: Vec3 { x: yx,y: yy,z: yz, },
            z: Vec3 { x: zx,y: zy,z: zz, },
            w: Vec3 { x: wx,y: wy,z: wz, },
        }
    }
}

impl<T: Copy> From<[Vec3<T>; 4]> for Mat4x3<T> {
    fn from(array: [Vec3<T>; 4]) -> Self {
        Mat4x3 {
            x: array[0],
            y: array[1],
            z: array[2],
            w: array[3],
        }
    }
}

impl<T: Copy> From<&[Vec3<T>; 4]> for Mat4x3<T> {
    fn from(slice: &[Vec3<T>; 4]) -> Self {
        Mat4x3 {
            x: slice[0],
            y: slice[1],
            z: slice[2],
            w: slice[3],
        }
    }
}

impl<T: Copy> From<[T; 12]> for Mat4x3<T> {
    fn from(array: [T; 12]) -> Self {
        Mat4x3 {
            x: Vec3 { x: array[0],y: array[1],z: array[2], },
            y: Vec3 { x: array[3],y: array[4],z: array[5], },
            z: Vec3 { x: array[6],y: array[7],z: array[8], },
            w: Vec3 { x: array[9],y: array[10],z: array[11], },
        }
    }
}

impl<T: Copy> From<&[T; 12]> for Mat4x3<T> {
    fn from(slice: &[T; 12]) -> Self {
        Mat4x3 {
            x: Vec3 { x: slice[0],y: slice[1],z: slice[2], },
            y: Vec3 { x: slice[3],y: slice[4],z: slice[5], },
            z: Vec3 { x: slice[6],y: slice[7],z: slice[8], },
            w: Vec3 { x: slice[9],y: slice[10],z: slice[11], },
        }
    }
}

impl<T: PartialEq> PartialEq for Mat4x3<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w)
    }
}

impl<T: Display> Display for Mat4x3<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"[{},{},{},{}]",self.x,self.y,self.z,self.w)
    }
}

// matrix + matrix
impl<T: Add<T,Output=T>> Add<Mat4x3<T>> for Mat4x3<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Mat4x3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

// matrix += matrix
impl<T: AddAssign<T>> AddAssign<Mat4x3<T>> for Mat4x3<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

// matrix - matrix
impl<T: Sub<T,Output=T>> Sub<Mat4x3<T>> for Mat4x3<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Mat4x3 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

// matrix -= matrix
impl<T: SubAssign<T>> SubAssign<Mat4x3<T>> for Mat4x3<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

/*

A00 A01 ... A0n
A10 A11 ... A1n
 :           :
Am0 Am1 ... Amn

times

B00 B01 ... B0p
B10 B11 ... B1p
 :           :
Bn0 Bn1 ... Bnp

equals

C00 C01 ... C0p
C10 C11 ... C1p
 :           :
Cm0 Cm1 ... Cmp

such that

Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..n,Aik Bkj)

so multiplying an mxn-dimensional matrix by an n-dimensional column vector:

Ci = Ai0 B0 + Ai1 B1 + ... + Ain Bn = sum(k = 0..n,Aik Bk)

and multiplying an m-dimensional row vector by an nxp-dimensional matrix:

row vector Cj = A0 B0j + A1 B1j + ... + An Bnj = sum(k = 0..n,Ak Bkj)

*/

// scalar * matrix
macro_rules! scalar_mat4x3_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Mat4x3<$t>> for $t {
                type Output = Mat4x3<$t>;
                fn mul(self,other: Mat4x3<$t>) -> Mat4x3<$t> {
                    Mat4x3 {
                        x: self * other.x,
                        y: self * other.y,
                        z: self * other.z,
                        w: self * other.w,
                    }
                }
            }
        )+
    }
}

scalar_mat4x3_mul!(f32 f64);

// matrix * scalar
impl<T: Copy + Mul<T,Output=T>> Mul<T> for Mat4x3<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Mat4x3 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

// matrix * vector
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec3<T>> for Mat4x3<T> {
    type Output = Vec4<T>;
    fn mul(self,other: Vec3<T>) -> Vec4<T> {
        Vec4 {
            x: self.x.x * other.x + self.x.y * other.y + self.x.z * other.z,
            y: self.y.x * other.x + self.y.y * other.y + self.y.z * other.z,
            z: self.z.x * other.x + self.z.y * other.y + self.z.z * other.z,
            w: self.w.x * other.x + self.w.y * other.y + self.w.z * other.z,
        }
    }
}

// matrix * matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat3x2<T>> for Mat4x3<T> {
    type Output = Mat4x2<T>;
    fn mul(self,other: Mat3x2<T>) -> Mat4x2<T> {
        Mat4x2 {
            x: Vec2 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y,
            },
            y: Vec2 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y,
            },
            z: Vec2 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y,
            },
            w: Vec2 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat3x3<T>> for Mat4x3<T> {
    type Output = Mat4x3<T>;
    fn mul(self,other: Mat3x3<T>) -> Mat4x3<T> {
        Mat4x3 {
            x: Vec3 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y,
                z: self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z,
            },
            y: Vec3 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y,
                z: self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z,
            },
            z: Vec3 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y,
                z: self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z,
            },
            w: Vec3 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y,
                z: self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat3x4<T>> for Mat4x3<T> {
    type Output = Mat4x4<T>;
    fn mul(self,other: Mat3x4<T>) -> Mat4x4<T> {
        Mat4x4 {
            x: Vec4 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y,
                z: self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z,
                w: self.x.x * other.x.w + self.x.y * other.y.w + self.x.z * other.z.w,
            },
            y: Vec4 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y,
                z: self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z,
                w: self.y.x * other.x.w + self.y.y * other.y.w + self.y.z * other.z.w,
            },
            z: Vec4 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y,
                z: self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z,
                w: self.z.x * other.x.w + self.z.y * other.y.w + self.z.z * other.z.w,
            },
            w: Vec4 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y,
                z: self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z,
                w: self.w.x * other.x.w + self.w.y * other.y.w + self.w.z * other.z.w,
            },
        }
    }
}

// matrix *= scalar
impl<T: Copy + MulAssign<T>> MulAssign<T> for Mat4x3<T> {
    fn mul_assign(&mut self,other: T) {
        self.x.x *= other;
        self.x.y *= other;
        self.x.z *= other;
        self.y.x *= other;
        self.y.y *= other;
        self.y.z *= other;
        self.z.x *= other;
        self.z.y *= other;
        self.z.z *= other;
        self.w.x *= other;
        self.w.y *= other;
        self.w.z *= other;
    }
}

// matrix *= matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> MulAssign<Mat3x3<T>> for Mat4x3<T> {
    fn mul_assign(&mut self,other: Mat3x3<T>) {
        let xx = self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x;
        let xy = self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y;
        let xz = self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z;
        self.x = Vec3 { x: xx,y: xy,z: xz, };
        let yx = self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x;
        let yy = self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y;
        let yz = self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z;
        self.y = Vec3 { x: yx,y: yy,z: yz, };
        let zx = self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x;
        let zy = self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y;
        let zz = self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z;
        self.z = Vec3 { x: zx,y: zy,z: zz, };
        let wx = self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x;
        let wy = self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y;
        let wz = self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z;
        self.w = Vec3 { x: wx,y: wy,z: wz, };
    }
}

// matrix / scalar
impl<T: Copy + Div<T,Output=T>> Div<T> for Mat4x3<T> {
    type Output = Mat4x3<T>;
    fn div(self,other: T) -> Mat4x3<T> {
        Mat4x3 {
            x: Vec3 { x: self.x.x / other,y: self.x.y / other,z: self.x.z / other, },
            y: Vec3 { x: self.y.x / other,y: self.y.y / other,z: self.y.z / other, },
            z: Vec3 { x: self.z.x / other,y: self.z.y / other,z: self.z.z / other, },
            w: Vec3 { x: self.w.x / other,y: self.w.y / other,z: self.w.z / other, },
        }
    }
}

// matrix /= scalar
impl<T: Copy + DivAssign<T>> DivAssign<T> for Mat4x3<T> {
    fn div_assign(&mut self,other: T) {
        self.x.x /= other;
        self.x.y /= other;
        self.x.z /= other;
        self.y.x /= other;
        self.y.y /= other;
        self.y.z /= other;
        self.z.x /= other;
        self.z.y /= other;
        self.z.z /= other;
        self.w.x /= other;
        self.w.y /= other;
        self.w.z /= other;
    }
}

// -matrix
impl<T: Neg<Output=T>> Neg for Mat4x3<T> {
    type Output = Mat4x3<T>;
    fn neg(self) -> Mat4x3<T> {
        Mat4x3 {
            x: Vec3 { x: -self.x.x,y: -self.x.y,z: -self.x.z, },
            y: Vec3 { x: -self.y.x,y: -self.y.y,z: -self.y.z, },
            z: Vec3 { x: -self.z.x,y: -self.z.y,z: -self.z.z, },
            w: Vec3 { x: -self.w.x,y: -self.w.y,z: -self.w.z, },
        }
    }
}

macro_rules! mat4x3_float {
    ($($t:ty)+) => {
        $(
            impl Mat4x3<$t> {
                pub fn transpose(self) -> Mat3x4<$t> {
                    Mat3x4 {
                        x: Vec4 { x: self.x.x,y: self.y.x,z: self.z.x,w: self.w.x, },
                        y: Vec4 { x: self.x.y,y: self.y.y,z: self.z.y,w: self.w.y, },
                        z: Vec4 { x: self.x.z,y: self.y.z,z: self.z.z,w: self.w.z, },
                    }
                }
            }
        )+
    }
}

mat4x3_float!(f32 f64);
