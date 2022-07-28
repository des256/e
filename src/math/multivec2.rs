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
pub struct MultiVec2<T> {
    pub r: T,
    pub x: T,pub y: T,
    pub xy: T,
}

impl<T> MultiVec2<T> {
    pub fn new(
        r: T,
        x: T,y: T,
        xy: T
    ) -> Self {
        MultiVec2 {
            r: r,
            x: x,y: y,
            xy: xy,
        }
    }
}

impl<T: PartialEq> PartialEq for MultiVec2<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) &&
        (self.x == other.x) && (self.y == other.y) &&
        (self.xy == other.xy)
    }
}

impl<T: Zero> Zero for MultiVec2<T> {
    const ZERO: Self = MultiVec2 { r: T::ZERO,x: T::ZERO,y: T::ZERO, xy: T::ZERO, };
}

impl<T: Zero + PartialOrd + Display> Display for MultiVec2<T> {
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
        let sxy = if self.xy < T::ZERO {
            format!("{}xy",self.xy)
        } else {
            format!("+{}xy",self.xy)
        };
        write!(f,"{}{}{}{}",self.r,sx,sy,sxy)
    }
}

impl<T: Add<T,Output=T>> Add<MultiVec2<T>> for MultiVec2<T> {
    type Output = Self;
    fn add(self,other: MultiVec2<T>) -> Self {
        MultiVec2 {
            r: self.r + other.r,
            x: self.x + other.x,y: self.y + other.y,
            xy: self.xy + other.xy,
        }
    }
}

impl<T: AddAssign<T>> AddAssign<MultiVec2<T>> for MultiVec2<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.x += other.x; self.y += other.y;
        self.xy += other.xy;
    }
}

impl<T: Sub<T,Output=T>> Sub<MultiVec2<T>> for MultiVec2<T> {
    type Output = Self;
    fn sub(self,other: MultiVec2<T>) -> Self {
        MultiVec2 {
            r: self.r - other.r,
            x: self.x - other.x,y: self.y - other.y,
            xy: self.xy - other.xy,
        }
    }
}

impl<T: SubAssign<T>> SubAssign<MultiVec2<T>> for MultiVec2<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.x -= other.x; self.y -= other.y;
        self.xy -= other.xy;
    }
}

macro_rules! scalar_multivec2_mul {
    ($t:ty) => {
        impl Mul<MultiVec2<$t>> for $t {
            type Output = MultiVec2<$t>;
            fn mul(self,other: MultiVec2<$t>) -> MultiVec2<$t> {
                MultiVec2 {
                    r: self * other.r,
                    x: self * other.x,y: self * other.y,
                    xy: self * other.xy,
                }
            }
        }        
    }
}

scalar_multivec2_mul!(f32);
scalar_multivec2_mul!(f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for MultiVec2<T> {
    type Output = MultiVec2<T>;
    fn mul(self,other: T) -> Self {
        MultiVec2 {
            r: self.r * other,
            x: self.x * other,y: self.y * other,
            xy: self.xy * other,
        }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for MultiVec2<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.x *= other; self.y *= other;
        self.xy *= other;
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> Mul<MultiVec2<T>> for MultiVec2<T> {
    type Output = MultiVec2<T>;
    fn mul(self,other: MultiVec2<T>) -> Self {
        MultiVec2 {
            r: self.r * other.r - self.x * other.x - self.y * other.y - self.xy * other.xy,
            x: self.r * other.x + self.x * other.r + self.y * other.xy - self.xy * other.y,
            y: self.r * other.y + self.y * other.r - self.x * other.xy + self.xy * other.x,
            xy: self.r * other.xy + self.xy * other.r + self.x * other.y - self.y * other.x,
        }
    }
}

impl<T: Copy + Mul<T,Output=T> + Add<T,Output=T> + Sub<T,Output=T>> MulAssign<MultiVec2<T>> for MultiVec2<T> {
    fn mul_assign(&mut self,other: MultiVec2<T>) {
        let r = self.r * other.r - self.x * other.x - self.y * other.y - self.xy * other.xy;
        let x = self.r * other.x + self.x * other.r + self.y * other.xy - self.xy * other.y;
        let y = self.r * other.y + self.y * other.r - self.x * other.xy + self.xy * other.x;
        let xy = self.r * other.xy + self.xy * other.r + self.x * other.y - self.y * other.x;
        self.r = r;
        self.x = x; self.y = y;
        self.xy = xy;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for MultiVec2<T> {
    type Output = MultiVec2<T>;
    fn div(self,other: T) -> Self {
        MultiVec2 {
            r: self.r / other,
            x: self.x / other,y: self.y / other,
            xy: self.xy / other,
        }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for MultiVec2<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.x /= other; self.y /= other;
        self.xy /= other;
    }
}

impl<T: Neg<Output=T>> Neg for MultiVec2<T> {
    type Output = MultiVec2<T>;
    fn neg(self) -> MultiVec2<T> {
        MultiVec2 {
            r: -self.r,
            x: -self.x,y: -self.y,
            xy: -self.xy,
        }
    }
}

impl<T: Zero> From<T> for MultiVec2<T> {
    fn from(v: T) -> MultiVec2<T> {
        MultiVec2::new(v,T::ZERO,T::ZERO,T::ZERO)
    }
}

impl<T: Zero> From<Vec2<T>> for MultiVec2<T> {
    fn from(v: Vec2<T>) -> MultiVec2<T> {
        MultiVec2::new(T::ZERO,v.x,v.y,T::ZERO)
    }
}

impl<T: Zero> From<Complex<T>> for MultiVec2<T> {
    fn from(v: Complex<T>) -> MultiVec2<T> {
        MultiVec2::new(v.r,T::ZERO,T::ZERO,v.i)
    }
}
