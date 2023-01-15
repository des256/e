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
pub struct Mat2x4<T> {
    pub x: Vec4<T>,
    pub y: Vec4<T>,
}

impl<T: Copy> Mat2x4<T> {
    pub fn new(
        xx: T,xy: T,xz: T,xw: T,
        yx: T,yy: T,yz: T,yw: T,
    ) -> Self {
        Mat2x4 {
            x: Vec4 { x: xx,y: xy,z: xz,w: xw, },
            y: Vec4 { x: yx,y: yy,z: yz,w: yw, },
        }
    }
}

impl<T: Copy> From<[Vec4<T>; 2]> for Mat2x4<T> {
    fn from(array: [Vec4<T>; 2]) -> Self {
        Mat2x4 {
            x: array[0],
            y: array[1],
        }
    }
}

impl<T: Copy> From<&[Vec4<T>; 2]> for Mat2x4<T> {
    fn from(slice: &[Vec4<T>; 2]) -> Self {
        Mat2x4 {
            x: slice[0],
            y: slice[1],
        }
    }
}

impl<T: Copy> From<[T; 8]> for Mat2x4<T> {
    fn from(array: [T; 8]) -> Self {
        Mat2x4 {
            x: Vec4 { x: array[0],y: array[1],z: array[2],w: array[3], },
            y: Vec4 { x: array[4],y: array[5],z: array[6],w: array[7], },
        }
    }
}

impl<T: Copy> From<&[T; 8]> for Mat2x4<T> {
    fn from(slice: &[T; 8]) -> Self {
        Mat2x4 {
            x: Vec4 { x: slice[0],y: slice[1],z: slice[2],w: slice[3], },
            y: Vec4 { x: slice[4],y: slice[5],z: slice[6],w: slice[7], },
        }
    }
}

impl<T: PartialEq> PartialEq for Mat2x4<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y)
    }
}

impl<T: Display> Display for Mat2x4<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"[{},{}]",self.x,self.y)
    }
}

// matrix + matrix
impl<T: Add<T,Output=T>> Add<Mat2x4<T>> for Mat2x4<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Mat2x4 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// matrix += matrix
impl<T: AddAssign<T>> AddAssign<Mat2x4<T>> for Mat2x4<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

// matrix - matrix
impl<T: Sub<T,Output=T>> Sub<Mat2x4<T>> for Mat2x4<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Mat2x4 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

// matrix -= matrix
impl<T: SubAssign<T>> SubAssign<Mat2x4<T>> for Mat2x4<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
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

Amn * Bnp = Cmp, where: Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..n,Aik Bkj)

A3x4 * B4 = C3, where: Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..n,Aik Bkj)
A3x4 * B4x3 = C3x3, where: Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..4,Aik Bkj)
A3x4 * B4x4 = C3x4, where: Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..4,Aik Bkj)
A3x4 * B4x2 = C3x2, where: Cij = Ai0 B0j + Ai1 B1j + ... + Ain Bnj = sum(k = 0..4,Aik Bkj)

*/

// scalar * matrix
macro_rules! scalar_mat2x4_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Mat2x4<$t>> for $t {
                type Output = Mat2x4<$t>;
                fn mul(self,other: Mat2x4<$t>) -> Mat2x4<$t> {
                    Mat2x4 {
                        x: self * other.x,
                        y: self * other.y,
                    }
                }
            }
        )+
    }
}

scalar_mat2x4_mul!(f32 f64);

// matrix * scalar
impl<T: Copy + Mul<T,Output=T>> Mul<T> for Mat2x4<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Mat2x4 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

// matrix * vector
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec4<T>> for Mat2x4<T> {
    type Output = Vec2<T>;
    fn mul(self,other: Vec4<T>) -> Vec2<T> {
        Vec2 {
            x: self.x.x * other.x + self.x.y * other.y + self.x.z * other.z + self.x.w * other.w,
            y: self.y.x * other.x + self.y.y * other.y + self.y.z * other.z + self.y.w * other.w,
        }
    }
}

// matrix * matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x2<T>> for Mat2x4<T> {
    type Output = Mat2x2<T>;
    fn mul(self,other: Mat4x2<T>) -> Mat2x2<T> {
        Mat2x2 {
            x: Vec2 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y,
            },
            y: Vec2 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x3<T>> for Mat2x4<T> {
    type Output = Mat2x3<T>;
    fn mul(self,other: Mat4x3<T>) -> Mat2x3<T> {
        Mat2x3 {
            x: Vec3 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y,
                z: self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z + self.x.w * other.w.z,
            },
            y: Vec3 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y,
                z: self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z + self.y.w * other.w.z,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x4<T>> for Mat2x4<T> {
    type Output = Mat2x4<T>;
    fn mul(self,other: Mat4x4<T>) -> Mat2x4<T> {
        Mat2x4 {
            x: Vec4 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y,
                z: self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z + self.x.w * other.w.z,
                w: self.x.x * other.x.w + self.x.y * other.y.w + self.x.z * other.z.w + self.x.w * other.w.w,
            },
            y: Vec4 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y,
                z: self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z + self.y.w * other.w.z,
                w: self.y.x * other.x.w + self.y.y * other.y.w + self.y.z * other.z.w + self.y.w * other.w.w,
            },
        }
    }
}

// matrix *= scalar
impl<T: Copy + MulAssign<T>> MulAssign<T> for Mat2x4<T> {
    fn mul_assign(&mut self,other: T) {
        self.x.x *= other;
        self.x.y *= other;
        self.x.z *= other;
        self.x.w *= other;
        self.y.x *= other;
        self.y.y *= other;
        self.y.z *= other;
        self.y.w *= other;
    }
}

// matrix *= matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> MulAssign<Mat4x4<T>> for Mat2x4<T> {
    fn mul_assign(&mut self,other: Mat4x4<T>) {
        let xx = self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x;
        let xy = self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y;
        let xz = self.x.x * other.x.z + self.x.y * other.y.z + self.x.z * other.z.z + self.x.w * other.w.z;
        let xw = self.x.x * other.x.w + self.x.y * other.y.w + self.x.z * other.z.w + self.x.w * other.w.w;
        self.x = Vec4 { x: xx,y: xy,z: xz,w: xw, };
        let yx = self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x;
        let yy = self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y;
        let yz = self.y.x * other.x.z + self.y.y * other.y.z + self.y.z * other.z.z + self.y.w * other.w.z;
        let yw = self.y.x * other.x.w + self.y.y * other.y.w + self.y.z * other.z.w + self.y.w * other.w.w;
        self.y = Vec4 { x: yx,y: yy,z: yz,w: yw, };
    }
}

// matrix / scalar
impl<T: Copy + Div<T,Output=T>> Div<T> for Mat2x4<T> {
    type Output = Mat2x4<T>;
    fn div(self,other: T) -> Mat2x4<T> {
        Mat2x4 {
            x: Vec4 { x: self.x.x / other,y: self.x.y / other,z: self.x.z / other,w: self.x.w / other, },
            y: Vec4 { x: self.y.x / other,y: self.y.y / other,z: self.y.z / other,w: self.y.w / other, },
        }
    }
}

// matrix /= scalar
impl<T: Copy + DivAssign<T>> DivAssign<T> for Mat2x4<T> {
    fn div_assign(&mut self,other: T) {
        self.x.x /= other;
        self.x.y /= other;
        self.x.z /= other;
        self.x.w /= other;
        self.y.x /= other;
        self.y.y /= other;
        self.y.z /= other;
        self.y.w /= other;
    }
}

// -matrix
impl<T: Neg<Output=T>> Neg for Mat2x4<T> {
    type Output = Mat2x4<T>;
    fn neg(self) -> Mat2x4<T> {
        Mat2x4 {
            x: Vec4 { x: -self.x.x,y: -self.x.y,z: -self.x.z,w: -self.x.w, },
            y: Vec4 { x: -self.y.x,y: -self.y.y,z: -self.y.z,w: -self.y.w, },
        }
    }
}

macro_rules! mat2x4_float {
    ($($t:ty)+) => {
        $(
            impl Mat2x4<$t> {
                pub fn transpose(self) -> Mat4x2<$t> {
                    Mat4x2 {
                        x: Vec2 { x: self.x.x,y: self.y.x, },
                        y: Vec2 { x: self.x.y,y: self.y.y, },
                        z: Vec2 { x: self.x.z,y: self.y.z, },
                        w: Vec2 { x: self.x.w,y: self.y.w, },
                    }
                }
            }
        )+
    }
}

mat2x4_float!(f32 f64);
