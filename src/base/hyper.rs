use {
    crate::*,
    std::{
        fmt::{
            Display,
            Debug,
            Formatter,
            Result
        },
    },
};

/// Hyperrectangle.
#[derive(Copy,Clone,Debug)]
pub struct Hyper<I,U> {
    pub o: Vec3<I>,
    pub s: Vec3<U>,
}

impl<I,U> Hyper<I,U> {
    pub fn new(o: Vec3<I>,s: Vec3<U>) -> Hyper<I,U> {
        Hyper { o: o,s: s, }
    }
}

impl<I: Display,U: Display> Display for Hyper<I,U> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{} {}x{}x{})",self.o.x,self.o.y,self.o.z,self.s.x,self.s.y,self.s.z)
    }
}

macro_rules! hyper_contains {
    ($i:ty,$u:ty) => {
        impl Hyper<$i,$u> {
            pub fn contains(&self,p: &Vec3<$i>) -> bool {
                (p.x >= self.o.x) &&
                (p.y >= self.o.y) &&
                (p.z >= self.o.z) &&
                (p.x < self.o.x + self.s.x as $i) &&
                (p.y < self.o.y + self.s.y as $i) &&
                (p.z < self.o.z + self.s.z as $i)
            }
        }
    }
}

hyper_contains!(i8,u8);
hyper_contains!(i16,u16);
hyper_contains!(i32,u32);
hyper_contains!(i64,u64);
hyper_contains!(i128,u128);
hyper_contains!(isize,usize);
hyper_contains!(f32,f32);
hyper_contains!(f64,f64);

#[allow(non_camel_case_types)]
pub type i8h = Hyper<i8,u8>;
#[allow(non_camel_case_types)]
pub type i16h = Hyper<i16,u16>;
#[allow(non_camel_case_types)]
pub type i32h = Hyper<i32,u32>;
#[allow(non_camel_case_types)]
pub type i64h = Hyper<i64,u64>;
#[allow(non_camel_case_types)]
pub type i128h = Hyper<i128,u128>;
#[allow(non_camel_case_types)]
pub type isizeh = Hyper<isize,usize>;
#[allow(non_camel_case_types)]
pub type f32h = Hyper<f32,f32>;
#[allow(non_camel_case_types)]
pub type f64h = Hyper<f64,f64>;
