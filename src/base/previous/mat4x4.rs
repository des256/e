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
pub struct Mat4x4<T> {
    pub x: Vec4<T>,
    pub y: Vec4<T>,
    pub z: Vec4<T>,
    pub w: Vec4<T>,
}

impl<T: Copy> Mat4x4<T> {
    pub fn new(
        xx: T,xy: T,xz: T,xw: T,
        yx: T,yy: T,yz: T,yw: T,
        zx: T,zy: T,zz: T,zw: T,
        wx: T,wy: T,wz: T,ww: T,
    ) -> Self {
        Mat4x4 {
            x: Vec4 { x: xx,y: xy,z: xz,w: xw, },
            y: Vec4 { x: yx,y: yy,z: yz,w: yw, },
            z: Vec4 { x: zx,y: zy,z: zz,w: zw, },
            w: Vec4 { x: wx,y: wy,z: wz,w: ww, },
        }
    }
}

impl<T: Copy> From<[Vec4<T>; 4]> for Mat4x4<T> {
    fn from(array: [Vec4<T>; 4]) -> Self {
        Mat4x4 {
            x: array[0],
            y: array[1],
            z: array[2],
            w: array[3],
        }
    }
}

impl<T: Copy> From<&[Vec4<T>; 4]> for Mat4x4<T> {
    fn from(slice: &[Vec4<T>; 4]) -> Self {
        Mat4x4 {
            x: slice[0],
            y: slice[1],
            z: slice[2],
            w: slice[3],
        }
    }
}

impl<T: Copy> From<[T; 16]> for Mat4x4<T> {
    fn from(array: [T; 16]) -> Self {
        Mat4x4 {
            x: Vec4 { x: array[0],y: array[1],z: array[2],w: array[3], },
            y: Vec4 { x: array[4],y: array[5],z: array[6],w: array[7], },
            z: Vec4 { x: array[8],y: array[9],z: array[10],w: array[11], },
            w: Vec4 { x: array[12],y: array[13],z: array[14],w: array[15], },
        }
    }
}

impl<T: Copy> From<&[T; 16]> for Mat4x4<T> {
    fn from(slice: &[T; 16]) -> Self {
        Mat4x4 {
            x: Vec4 { x: slice[0],y: slice[1],z: slice[2],w: slice[3], },
            y: Vec4 { x: slice[4],y: slice[5],z: slice[6],w: slice[7], },
            z: Vec4 { x: slice[8],y: slice[9],z: slice[10],w: slice[11], },
            w: Vec4 { x: slice[12],y: slice[13],z: slice[14],w: slice[15], },
        }
    }
}

impl<T: PartialEq> PartialEq for Mat4x4<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w)
    }
}

impl<T: Display> Display for Mat4x4<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"[{},{},{},{}]",self.x,self.y,self.z,self.w)
    }
}

// matrix + matrix
impl<T: Add<T,Output=T>> Add<Mat4x4<T>> for Mat4x4<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Mat4x4 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
            w: self.w + other.w,
        }
    }
}

// matrix += matrix
impl<T: AddAssign<T>> AddAssign<Mat4x4<T>> for Mat4x4<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

// matrix - matrix
impl<T: Sub<T,Output=T>> Sub<Mat4x4<T>> for Mat4x4<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Mat4x4 {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
            w: self.w - other.w,
        }
    }
}

// matrix -= matrix
impl<T: SubAssign<T>> SubAssign<Mat4x4<T>> for Mat4x4<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

/*

4x4 * 4x2 = 4x2
4x4 * 4x3 = 4x3
4x4 * 4x4 = 4x4

3x4 * 4x2 = 3x2
3x4 * 4x3 = 3x3
3x4 * 4x4 = 3x4

2x4 * 4x2 = 2x2
2x4 * 4x3 = 2x3
2x4 * 4x4 = 2x4


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
macro_rules! scalar_mat4x4_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Mat4x4<$t>> for $t {
                type Output = Mat4x4<$t>;
                fn mul(self,other: Mat4x4<$t>) -> Mat4x4<$t> {
                    Mat4x4 {
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

scalar_mat4x4_mul!(f32 f64);

// matrix * scalar
impl<T: Copy + Mul<T,Output=T>> Mul<T> for Mat4x4<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Mat4x4 {
            x: self.x * other,
            y: self.y * other,
            z: self.z * other,
            w: self.w * other,
        }
    }
}

// matrix * vector
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec4<T>> for Mat4x4<T> {
    type Output = Vec4<T>;
    fn mul(self,other: Vec4<T>) -> Vec4<T> {
        Vec4 {
            x: self.x.x * other.x + self.x.y * other.y + self.x.z * other.z + self.x.w * other.w,
            y: self.y.x * other.x + self.y.y * other.y + self.y.z * other.z + self.y.w * other.w,
            z: self.z.x * other.x + self.z.y * other.y + self.z.z * other.z + self.z.w * other.w,
            w: self.w.x * other.x + self.w.y * other.y + self.w.z * other.z + self.w.w * other.w,
        }
    }
}

// matrix * matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x2<T>> for Mat4x4<T> {
    type Output = Mat4x2<T>;
    fn mul(self,other: Mat4x2<T>) -> Mat4x2<T> {
        Mat4x2 {
            x: Vec2 {
                x: self.x.x * other.x.x + self.x.y * other.y.x + self.x.z * other.z.x + self.x.w * other.w.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y + self.x.z * other.z.y + self.x.w * other.w.y,
            },
            y: Vec2 {
                x: self.y.x * other.x.x + self.y.y * other.y.x + self.y.z * other.z.x + self.y.w * other.w.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y + self.y.z * other.z.y + self.y.w * other.w.y,
            },
            z: Vec2 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y,
            },
            w: Vec2 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x3<T>> for Mat4x4<T> {
    type Output = Mat4x3<T>;
    fn mul(self,other: Mat4x3<T>) -> Mat4x3<T> {
        Mat4x3 {
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
            z: Vec3 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y,
                z: self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z + self.z.w * other.w.z,
            },
            w: Vec3 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y,
                z: self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z + self.w.w * other.w.z,
            },
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat4x4<T>> for Mat4x4<T> {
    type Output = Mat4x4<T>;
    fn mul(self,other: Mat4x4<T>) -> Mat4x4<T> {
        Mat4x4 {
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
            z: Vec4 {
                x: self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x,
                y: self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y,
                z: self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z + self.z.w * other.w.z,
                w: self.z.x * other.x.w + self.z.y * other.y.w + self.z.z * other.z.w + self.z.w * other.w.w,
            },
            w: Vec4 {
                x: self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x,
                y: self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y,
                z: self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z + self.w.w * other.w.z,
                w: self.w.x * other.x.w + self.w.y * other.y.w + self.w.z * other.z.w + self.w.w * other.w.w,
            },
        }
    }
}

// matrix *= scalar
impl<T: Copy + MulAssign<T>> MulAssign<T> for Mat4x4<T> {
    fn mul_assign(&mut self,other: T) {
        self.x.x *= other;
        self.x.y *= other;
        self.x.z *= other;
        self.x.w *= other;
        self.y.x *= other;
        self.y.y *= other;
        self.y.z *= other;
        self.y.w *= other;
        self.z.x *= other;
        self.z.y *= other;
        self.z.z *= other;
        self.z.w *= other;
        self.w.x *= other;
        self.w.y *= other;
        self.w.z *= other;
        self.w.w *= other;
    }
}

// matrix *= matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> MulAssign<Mat4x4<T>> for Mat4x4<T> {
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
        let zx = self.z.x * other.x.x + self.z.y * other.y.x + self.z.z * other.z.x + self.z.w * other.w.x;
        let zy = self.z.x * other.x.y + self.z.y * other.y.y + self.z.z * other.z.y + self.z.w * other.w.y;
        let zz = self.z.x * other.x.z + self.z.y * other.y.z + self.z.z * other.z.z + self.z.w * other.w.z;
        let zw = self.z.x * other.x.w + self.z.y * other.y.w + self.z.z * other.z.w + self.z.w * other.w.w;
        self.z = Vec4 { x: zx,y: zy,z: zz,w: zw, };
        let wx = self.w.x * other.x.x + self.w.y * other.y.x + self.w.z * other.z.x + self.w.w * other.w.x;
        let wy = self.w.x * other.x.y + self.w.y * other.y.y + self.w.z * other.z.y + self.w.w * other.w.y;
        let wz = self.w.x * other.x.z + self.w.y * other.y.z + self.w.z * other.z.z + self.w.w * other.w.z;
        let ww = self.w.x * other.x.w + self.w.y * other.y.w + self.w.z * other.z.w + self.w.w * other.w.w;
        self.w = Vec4 { x: wx,y: wy,z: wz,w: ww, };
    }
}

// matrix / scalar
impl<T: Copy + Div<T,Output=T>> Div<T> for Mat4x4<T> {
    type Output = Mat4x4<T>;
    fn div(self,other: T) -> Mat4x4<T> {
        Mat4x4 {
            x: Vec4 { x: self.x.x / other,y: self.x.y / other,z: self.x.z / other,w: self.x.w / other, },
            y: Vec4 { x: self.y.x / other,y: self.y.y / other,z: self.y.z / other,w: self.y.w / other, },
            z: Vec4 { x: self.z.x / other,y: self.z.y / other,z: self.z.z / other,w: self.z.w / other, },
            w: Vec4 { x: self.w.x / other,y: self.w.y / other,z: self.w.z / other,w: self.w.w / other, },
        }
    }
}

// matrix /= scalar
impl<T: Copy + DivAssign<T>> DivAssign<T> for Mat4x4<T> {
    fn div_assign(&mut self,other: T) {
        self.x.x /= other;
        self.x.y /= other;
        self.x.z /= other;
        self.x.w /= other;
        self.y.x /= other;
        self.y.y /= other;
        self.y.z /= other;
        self.y.w /= other;
        self.z.x /= other;
        self.z.y /= other;
        self.z.z /= other;
        self.z.w /= other;
        self.w.x /= other;
        self.w.y /= other;
        self.w.z /= other;
        self.w.w /= other;
    }
}

// -matrix
impl<T: Neg<Output=T>> Neg for Mat4x4<T> {
    type Output = Mat4x4<T>;
    fn neg(self) -> Mat4x4<T> {
        Mat4x4 {
            x: Vec4 { x: -self.x.x,y: -self.x.y,z: -self.x.z,w: -self.x.w, },
            y: Vec4 { x: -self.y.x,y: -self.y.y,z: -self.y.z,w: -self.y.w, },
            z: Vec4 { x: -self.z.x,y: -self.z.y,z: -self.z.z,w: -self.z.w, },
            w: Vec4 { x: -self.w.x,y: -self.w.y,z: -self.w.z,w: -self.w.w, },
        }
    }
}

macro_rules! mat4x4_float {
    ($($t:ty)+) => {
        $(
            impl Mat4x4<$t> {
                pub fn transpose(self) -> Mat4x4<$t> {
                    Mat4x4 {
                        x: Vec4 { x: self.x.x,y: self.y.x,z: self.z.x,w: self.w.x, },
                        y: Vec4 { x: self.x.y,y: self.y.y,z: self.z.y,w: self.w.y, },
                        z: Vec4 { x: self.x.z,y: self.y.z,z: self.z.z,w: self.w.z, },
                        w: Vec4 { x: self.x.w,y: self.y.w,z: self.z.w,w: self.w.w, },
                    }
                }

                pub fn determinant(self) -> $t {

                    // xx  yx  zx  wx
                    // xy  yy  zy  wy
                    // xz  yz  zz  wz
                    // xw  yw  zw  ww
                    let xx = self.x.x;
                    let xy = self.x.y;
                    let xz = self.x.z;
                    let xw = self.x.w;
                    let yx = self.y.x;
                    let yy = self.y.y;
                    let yz = self.y.z;
                    let yw = self.y.w;
                    let zx = self.z.x;
                    let zy = self.z.y;
                    let zz = self.z.z;
                    let zw = self.z.w;
                    let wx = self.w.x;
                    let wy = self.w.y;
                    let wz = self.w.z;
                    let ww = self.w.w;

                    // adjoint of first column
                    // yy  zy  wy
                    // yz  zz  wz
                    // yw  zw  ww
                    let axx = yy * (zz * ww - wz * zw) - yz * (zw * wy - ww * zy) + yw * (zy * wz - wy * zz);

                    // yz  zz  wz
                    // yw  zw  ww
                    // yx  zx  wx
                    let axy = -(yz * (zw * wx - ww * zx) - yw * (zx * wz - wx * zz) + yx * (zz * ww - wz * zw));

                    // yw  zw  ww
                    // yx  zx  wx
                    // yy  zy  wy
                    let axz = yw * (zx * wy - wx * zy) - yx * (zy * ww - wy * zw) + yy * (zw * wx - ww * zx);

                    // yx  zx  wx
                    // yy  zy  wy
                    // yz  zz  wz
                    let axw = -(yx * (zy * wz - wy * zz) - yy * (zz * wx - wz * wx) + yz * (wx * wy - wx * zy));

                    // determinant
                    xx * axx + xy * axy + xz * axz + xw * axw
                }

                pub fn inverse(self) -> Self {
                    // xx  yx  zx  wx
                    // xy  yy  zy  wy
                    // xz  yz  zz  wz
                    // xw  yw  zw  ww
                    let xx = self.x.x;
                    let xy = self.x.y;
                    let xz = self.x.z;
                    let xw = self.x.w;
                    let yx = self.y.x;
                    let yy = self.y.y;
                    let yz = self.y.z;
                    let yw = self.y.w;
                    let zx = self.z.x;
                    let zy = self.z.y;
                    let zz = self.z.z;
                    let zw = self.z.w;
                    let wx = self.w.x;
                    let wy = self.w.y;
                    let wz = self.w.z;
                    let ww = self.w.w;

                    // adjoint of first column
                    // yy  zy  wy
                    // yz  zz  wz
                    // yw  zw  ww
                    let axx = yy * (zz * ww - wz * zw) - yz * (zw * wy - ww * zy) + yw * (zy * wz - wy * zz);

                    // yz  zz  wz
                    // yw  zw  ww
                    // yx  zx  wx
                    let axy = -(yz * (zw * wx - ww * zx) - yw * (zx * wz - wx * zz) + yx * (zz * ww - wz * zw));

                    // yw  zw  ww
                    // yx  zx  wx
                    // yy  zy  wy
                    let axz = yw * (zx * wy - wx * zy) - yx * (zy * ww - wy * zw) + yy * (zw * wx - ww * zx);

                    // yx  zx  wx
                    // yy  zy  wy
                    // yz  zz  wz
                    let axw = -(yx * (zy * wz - wy * zz) - yy * (zz * wx - wz * wx) + yz * (wx * wy - wx * zy));

                    // determinant
                    let det = xx * axx + xy * axy + xz * axz + xw * axw;
                    if det == 0.0 {
                        return self;
                    }
                    
                    // rest of adjoint

                    // yx  zx  wx  xx
                    // yy  zy  wy  xy
                    // yz  zz  wz  xz
                    // yw  zw  ww  xw

                    // zy  wy  xy
                    // zz  wz  xz
                    // zw  ww  xw
                    let ayx = -(zy * (wz * xw - xz * ww) - zz * (ww * xy - xw * wy) + zw * (wy * xz - xy * wz));

                    // zz  wz  xz
                    // zw  ww  xw
                    // zx  wx  xx
                    let ayy = zz * (ww * xx - xw * wx) - zw * (wx * xz - xx * wz) + zx * (wz * xw - xz * ww);

                    // zw  ww  xw
                    // zx  wx  xx
                    // zy  wy  xy
                    let ayz = -(zw * (wx * xy - xx * wy) - zx * (wy * xw - xy * ww) + zy * (ww * xx - xw * wx));

                    // zx  wx  xx
                    // zy  wy  xy
                    // zz  wz  xz
                    let ayw = zx * (wy * xz - xy * wz) - zy * (wz * xx - xz * wx) + zz * (wx * xy - xx * wy);

                    // zx  wx  xx  yx
                    // zy  wy  xy  yy
                    // zz  wz  xz  yz
                    // zw  ww  xw  yw

                    // wy  xy  yy
                    // wz  xz  yz
                    // ww  xw  yw
                    let azx = wy * (xz * yw - yz * xw) - wz * (xw * yy - yz * xy) + ww * (xy * yz - yy * xz);

                    // wz  xz  yz
                    // ww  xw  yw
                    // wx  xx  yx
                    let azy = -(wz * (xw * yx - yw * xx) - ww * (xx * yz - yx * xz) + wx * (xz * yw - yz * xw));

                    // ww  xw  yw
                    // wx  xx  yx
                    // wy  xy  yy
                    let azz = ww * (xx * yy - yx * xy) - wx * (xy * yw - yy * xw) + wy * (xw * yx - yw * xx);

                    // wx  xx  yx
                    // wy  xy  yy
                    // wz  xz  yz
                    let azw = -(wx * (xy * yz - yy * xz) - wy * (xz * yx - yz * xx) + wz * (xx * yy - yx * xy));

                    // wx  xx  yx  zx
                    // wy  xy  yy  zy
                    // wz  xz  yz  zz
                    // ww  xw  yw  zw

                    // xy  yy  zy
                    // xz  yz  zz
                    // xw  yw  zw
                    let awx = -(xy * (yz * zw - zz * yw) - xz * (yw * zy - zw * yy) + xw * (yy * zz - zy * yz));

                    // xz  yz  zz
                    // xw  yw  zw                    
                    // xx  yx  zx
                    let awy = xz * (yw * zx - zw * yx) - xw * (yx * zz - zx * yz) + xx * (yz * zw - zz * yw);

                    // xw  yw  zw
                    // xx  yx  zx
                    // xy  yy  zy
                    let awz = -(xw * (yx * zy - zx * yy) - xx * (yy * zw - zy * yw) + xy * (yw * zx - zw * yx));

                    // xx  yx  zx
                    // xy  yy  zy
                    // xz  yz  zz
                    let aww = xx * (yy * zz - zy * yz) - xy * (yz * zx - zz * yx) + xz * (yx * zy - zx * yy);

                    // transpose of adjoint divided by determinant
                    Mat4x4::new(
                        axx,ayx,azx,awx,
                        axy,ayy,azy,awy,
                        axz,ayz,azz,awz,
                        axw,ayw,azw,aww,
                    ) / det
                }
            }
        )+
    }
}

mat4x4_float!(f32 f64);
