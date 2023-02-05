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

impl<T> Vec<T> {
    pub fn dot(self,other: Vec<T>) -> T {
        let mut a = T::ZERO;
        for i in 0..self.len() {
            a += self.0[i] * other.0[i];
        }
        a
    }

    pub fn scale(&self,other: &Vec<T>) -> Self {
        if self.len() != other.len() {
            panic!("vectors should be same length");
        }
        let mut a = self;
        for i in 0..self.len() {
            a[i] *= other[i];
        }
        a
    }
}

// vector + vector
impl Add<Vec<T>> for Vec<T> {
    type Output = Vec<T>;
    fn add(self,other: Self) -> Self {
        if self.len() != other.len() {
            panic!("vectors should be same length");
        }
        let mut a = self;
        for i in 0..self.len() {
            a[i] += other[i];
        }
        a
    }
}

// vector += vector
impl AddAssign<Vec<T>> for Vec<T> {
    fn add_assign(&mut self,other: Self) {
        if self.len() != other.len() {
            panic!("vectors should be same length");
        }
        for i in 0..self.len() {
            self[i] += other[i];
        }
    }
}

// vector - vector
impl Sub<Vec<T>> for Vec<T> {
    type Output = Vec<T>;
    fn sub(self,other: Self) -> Self {
        if self.len() != other.len() {
            panic!("vectors should be same length");
        }
        let mut a: Vec<T> = self;
        for i in 0..self.len() {
            a[i] -= other[i];
        }
        a
    }
}

// vector -= vector
impl SubAssign<Vec<T>> for Vec<T> {
    fn sub_assign(&mut self,other: Self) {
        if self.len() != other.len() {
            panic!("vectors should be same length");
        }
        for i in 0..self.len() {
            self[i] -= other[i];
        }
    }
}

// scalar * vector
impl Mul<Vec<T>> for T {
    type Output = Vec<T>;
    fn mul(self,other: Vec<T>) -> Self::Output {
        let mut a: Vec<T> = other;
        for i in 0..other.len() {
            a[i] *= self;
        }
        a
    }
}

// vector * scalar
impl Mul<T> for Vec<T> {
    type Output = Vec<T>;
    fn mul(self,other: T) -> Self::Output {
        let mut a: Vec<T> = self;
        for i in 0..self.len() {
            a[i] *= other;
        }
        a
    }
}

// vector *= scalar
impl MulAssign<T> for Vec<T> {
    fn mul_assign(&mut self,other: T) {
        for i in 0..self.len() {
            self[i] *= other;
        }
    }
}

// vector / scalar
impl Div<T> for Vec<T> {
    type Output = Vec<T>;
    fn div(self,other: T) -> Self::Output {
        let mut a: Vec<T> = self;
        for i in 0..self.len() {
            a[i] /= other;
        }
        a
    }
}

// vector /= scalar
impl DivAssign<T> for VecN<N,T> {
    fn div_assign(&mut self,other: T) {
        for i in 0..self.len() {
            self[i] /= other;
        }
    }
}

// -vector
impl Neg for Vec<T> {
    type Output = Vec<T>;
    fn neg(self) -> Self::Output {
        let mut a = vec![T::ZERO; N];
        for i in 0..self.len() {
            a[i] = -self[i];
        }
        a
    }
}
