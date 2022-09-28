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
pub struct Mat2x2<T> {
    pub x: Vec2<T>,
    pub y: Vec2<T>,
}

impl<T: Copy> Mat2x2<T> {
    pub fn new(xx: T,xy: T,yx: T,yy: T) -> Self {
        Mat2x2 {
            x: Vec2 { x: xx, y: xy, },
            y: Vec2 { x: yx, y: yy, },
        }
    }
}

impl<T: Copy> From<[Vec2<T>; 2]> for Mat2x2<T> {
    fn from(array: [Vec2<T>; 2]) -> Self {
        Mat2x2 {
            x: array[0],
            y: array[1],
        }
    }
}

impl<T: Copy> From<&[Vec2<T>; 2]> for Mat2x2<T> {
    fn from(slice: &[Vec2<T>; 2]) -> Self {
        Mat2x2 {
            x: slice[0],
            y: slice[1],
        }
    }
}

impl<T: Copy> From<[T; 4]> for Mat2x2<T> {
    fn from(array: [T; 4]) -> Self {
        Mat2x2 {
            x: Vec2 { x: array[0],y: array[1], },
            y: Vec2 { x: array[2],y: array[2], },
        }
    }
}

impl<T: Copy> From<&[T; 4]> for Mat2x2<T> {
    fn from(slice: &[T; 4]) -> Self {
        Mat2x2 {
            x: Vec2 { x: slice[0],y: slice[1], },
            y: Vec2 { x: slice[2],y: slice[2], },
        }
    }
}

impl<T: PartialEq> PartialEq for Mat2x2<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y)
    }
}

impl<T: Display> Display for Mat2x2<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"[{},{}]",self.x,self.y)
    }
}

// matrix + matrix
impl<T: Add<T,Output=T>> Add<Mat2x2<T>> for Mat2x2<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Mat2x2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

// matrix += matrix
impl<T: AddAssign<T>> AddAssign<Mat2x2<T>> for Mat2x2<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
    }
}

// matrix - matrix
impl<T: Sub<T,Output=T>> Sub<Mat2x2<T>> for Mat2x2<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Mat2x2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

// matrix -= matrix
impl<T: SubAssign<T>> SubAssign<Mat2x2<T>> for Mat2x2<T> {
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

*/

// scalar * matrix
macro_rules! scalar_mat2x2_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Mat2x2<$t>> for $t {
                type Output = Mat2x2<$t>;
                fn mul(self,other: Mat2x2<$t>) -> Mat2x2<$t> {
                    Mat2x2 { x: self * other.x,y: self * other.y, }
                }
            }
        )+
    }
}

scalar_mat2x2_mul!(f32 f64);

// matrix * scalar
impl<T: Copy + Mul<T,Output=T>> Mul<T> for Mat2x2<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Mat2x2 {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

// matrix * vector
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Vec2<T>> for Mat2x2<T> {
    type Output = Vec2<T>;
    fn mul(self,other: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x.x * other.x + self.x.y * other.y,
            y: self.y.x * other.x + self.y.y * other.y,
        }
    }
}

// matrix * matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> Mul<Mat2x2<T>> for Mat2x2<T> {
    type Output = Mat2x2<T>;
    fn mul(self,other: Mat2x2<T>) -> Mat2x2<T> {
        Mat2x2 {
            x: Vec2 {
                x: self.x.x * other.x.x + self.x.y * other.y.x,
                y: self.x.x * other.x.y + self.x.y * other.y.y,
            },
            y: Vec2 {
                x: self.y.x * other.x.x + self.y.y * other.y.x,
                y: self.y.x * other.x.y + self.y.y * other.y.y,
            },
        }
    }
}

// matrix *= scalar
impl<T: Copy + MulAssign<T>> MulAssign<T> for Mat2x2<T> {
    fn mul_assign(&mut self,other: T) {
        self.x.x *= other;
        self.x.y *= other;
        self.y.x *= other;
        self.y.y *= other;
    }
}

// matrix *= matrix
impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T>> MulAssign<Mat2x2<T>> for Mat2x2<T> {
    fn mul_assign(&mut self,other: Mat2x2<T>) {
        let xx = self.x.x * other.x.x + self.x.y * other.y.x;
        let xy = self.x.x * other.x.y + self.x.y * other.y.y;
        self.x = Vec2 { x: xx, y: xy, };
        let yx = self.y.x * other.x.x + self.y.y * other.y.x;
        let yy = self.y.x * other.x.y + self.y.y * other.y.y;
        self.y = Vec2 { x: yx, y: yy, };
    }
}

// matrix / scalar
impl<T: Copy + Div<T,Output=T>> Div<T> for Mat2x2<T> {
    type Output = Mat2x2<T>;
    fn div(self,other: T) -> Mat2x2<T> {
        Mat2x2 {
            x: Vec2 { x: self.x.x / other,y: self.x.y / other, },
            y: Vec2 { x: self.y.x / other,y: self.y.y / other, }
        }
    }
}

// matrix /= scalar
impl<T: Copy + DivAssign<T>> DivAssign<T> for Mat2x2<T> {
    fn div_assign(&mut self,other: T) {
        self.x.x /= other;
        self.x.y /= other;
        self.y.x /= other;
        self.y.y /= other;
    }
}

// -matrix
impl<T: Neg<Output=T>> Neg for Mat2x2<T> {
    type Output = Mat2x2<T>;
    fn neg(self) -> Mat2x2<T> {
        Mat2x2 {
            x: Vec2 { x: -self.x.x,y: -self.x.y, },
            y: Vec2 { x: -self.y.x,y: -self.y.y, },
        }
    }
}
