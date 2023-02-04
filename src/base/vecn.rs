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
pub struct VecN<N,T>([T; N]);

impl<N: usize,T> VecN<N,T> {
    pub fn dot(self,other: VecN<N,T>) -> T {
        let mut a = T::ZERO;
        for i in 0..N {
            a += self.0[i] * other.0[i];
        }
        a
    }

    pub fn scale(&self,other: &VecN<N,T>) -> Self {
        let mut a: VecN<N,T> = self;
        for i in 0..N {
            a.0[i] *= other.0[i];
        }
        a
    }
}

impl Display for VecN<N,T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({}",self.0[0])?;
        if N > 1 {
            for i in 1..N {
                write!(f,",{}",self.0[i])?;
            }
        }
        write!(f,")")
    }
}

impl PartialEq for VecN<N,T> {
    fn eq(&self,other: &Self) -> bool {
        for i in 0..N {
            if self.0[i] != other.0[i] {
                return false;
            }
        }
        true
    }
}

impl Zero for VecN<N,T> {
    const ZERO: VecN<N,T> = VecN([T::ZERO; N]);
}

// vector + vector
impl Add<VecN<N,T>> for VecN<N,T> {
    type Output = VecN<N,T>;
    fn add(self,other: Self) -> Self {
        let mut a: VecN<N,T> = self;
        for i in 0..N {
            a.0[i] += other.0[i];
        }
        a
    }
}

// vector += vector
impl AddAssign<VecN<N,T>> for VecN<N,T> {
    fn add_assign(&mut self,other: Self) {
        for i in 0..N {
            self.0[i] += other.0[i];
        }
    }
}

// vector - vector
impl Sub<VecN<N,T>> for VecN<N,T> {
    type Output = VecN<N,T>;
    fn sub(self,other: Self) -> Self {
        let mut a: VecN<N,T> = self;
        for i in 0..N {
            a.0[i] -= other.0[i];
        }
        a
    }
}

// vector -= vector
impl SubAssign<VecN<N,T>> for VecN<N,T> {
    fn sub_assign(&mut self,other: Self) {
        for i in 0..N {
            self.0[i] -= other.0[i];
        }
    }
}

// scalar * vector
impl Mul<VecN<N,T>> for T {
    type Output = VecN<N,T>;
    fn mul(self,other: VecN<N,T>) -> Self::Output {
        let mut a: VecN<N,T> = other;
        for i in 0..N {
            a.0[i] *= self;
        }
        a
    }
}

// vector * scalar
impl Mul<T> for VecN<N,T> {
    type Output = VecN<N,T>;
    fn mul(self,other: T) -> Self::Output {
        let mut a: VecN<N,T> = self;
        for i in 0..N {
            a.0[i] *= other;
        }
        a
    }
}

// vector *= scalar
impl MulAssign<T> for VecN<N,T> {
    fn mul_assign(&mut self,other: T) {
        for i in 0..N {
            self.0[i] *= other;
        }
    }
}

// vector / scalar
impl Div<T> for VecN<N,T> {
    type Output = VecN<N,T>;
    fn div(self,other: T) -> Self::Output {
        let mut a: VecN<N,T> = self;
        for i in 0..N {
            a.0[i] /= other;
        }
        a
    }
}

// vector /= scalar
impl DivAssign<T> for VecN<N,T> {
    fn div_assign(&mut self,other: T) {
        for i in 0..N {
            self.0[i] /= other;
        }
    }
}

// -vector
impl Neg for VecN<N,T> {
    type Output = VecN<N,T>;
    fn neg(self) -> Self::Output {
        let mut a: VecN<N,T> = VecN([T::ZERO; N]);
        for i in 0..N {
            a.0[i] = -self.0[i];
        }
        a
    }
}
