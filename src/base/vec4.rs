use {
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Debug,
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
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

impl<T> Vec4<T> {
    pub fn new(x: T,y: T,z: T,w: T) -> Self {
        Vec4 { x: x,y: y,z: z,w: w, }
    }
}

impl<T: PartialEq> PartialEq for Vec4<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.x == other.x) && (self.y == other.y) && (self.z == other.z) && (self.w == other.w)
    }
}

impl<T: Display> Display for Vec4<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{},{})",self.x,self.y,self.z,self.w)
    }
}

impl<T: Add<T,Output=T>> Add<Vec4<T>> for Vec4<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Vec4 { x: self.x + other.x,y: self.y + other.y,z: self.z + other.z,w: self.w + other.w, }
    }
}

impl<T: AddAssign<T>> AddAssign<Vec4<T>> for Vec4<T> {
    fn add_assign(&mut self,other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
        self.w += other.w;
    }
}

impl<T: Sub<T,Output=T>> Sub<Vec4<T>> for Vec4<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Vec4 { x: self.x - other.x,y: self.y - other.y,z: self.z - other.z,w: self.w - other.w, }
    }
}

impl<T: SubAssign<T>> SubAssign<Vec4<T>> for Vec4<T> {
    fn sub_assign(&mut self,other: Self) {
        self.x -= other.x;
        self.y -= other.y;
        self.z -= other.z;
        self.w -= other.w;
    }
}

macro_rules! scalar_vec4_mul {
    ($t:ty) => {
        impl Mul<Vec4<$t>> for $t {
            type Output = Vec4<$t>;
            fn mul(self,other: Vec4<$t>) -> Vec4<$t> {
                Vec4 { x: self * other.x,y: self * other.y,z: self * other.z,w: self * other.w, }
            }
        }
    }
}

scalar_vec4_mul!(u8);
scalar_vec4_mul!(u16);
scalar_vec4_mul!(u32);
scalar_vec4_mul!(u64);
scalar_vec4_mul!(u128);
scalar_vec4_mul!(usize);
scalar_vec4_mul!(i8);
scalar_vec4_mul!(i16);
scalar_vec4_mul!(i32);
scalar_vec4_mul!(i64);
scalar_vec4_mul!(i128);
scalar_vec4_mul!(isize);
scalar_vec4_mul!(f32);
scalar_vec4_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Vec4<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Vec4 { x: self.x * other,y: self.y * other,z: self.z * other,w: self.w * other, }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Vec4<T> {
    fn mul_assign(&mut self,other: T) {
        self.x *= other;
        self.y *= other;
        self.z *= other;
        self.w *= other;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Vec4<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Vec4 { x: self.x / other,y: self.y / other,z: self.z / other,w: self.w / other, }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for Vec4<T> {
    fn div_assign(&mut self,other: T) {
        self.x /= other;
        self.y /= other;
        self.z /= other;
        self.w /= other;
    }
}

impl<T: Neg<Output=T>> Neg for Vec4<T> {
    type Output = Self;
    fn neg(self) -> Self {
        Vec4 { x: -self.x,y: -self.y,z: -self.z,w: -self.w, }
    }
}

macro_rules! vec4_float {
    ($t:ty) => {
        impl Vec4<$t> {
            pub fn dot(a: Self,b: Self) -> $t {
                a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
            }

            pub fn abs(&self) -> $t {
                (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
            }

            pub fn normalize(&self) -> Self {
                let d = self.abs();
                if d != 0 as $t {
                    *self / d
                }
                else {
                    *self
                }
            }
        }
    }
}

vec4_float!(f32);
vec4_float!(f64);

#[macro_export]
macro_rules! vec4 {
    ($x:expr,$y:expr,$z:expr,$w:expr) => { Vec2::new($x,$y,$z,$w) };
}
