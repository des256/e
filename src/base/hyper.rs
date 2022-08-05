use {
    crate::*,
    std::{
        fmt::{
            Display,
            Debug,
            Formatter,
            Result
        },
        ops::Add,
    },
};

/// Hyperrectangle.
#[derive(Copy,Clone,Debug)]
pub struct Hyper<T> {
    pub o: Vec3<T>,
    pub s: Vec3<T>,
}

impl<T: Copy + PartialOrd + Add<T,Output=T>> Hyper<T> {
    pub fn new(ox: T,oy: T,oz: T,sx: T,sy: T,sz: T) -> Hyper<T> {
        Hyper { o: vec3!(ox,oy,oz),s: vec3!(sx,sy,sz), }
    }

    pub fn contains(&self,p: &Vec3<T>) -> bool {
        (p.x >= self.o.x) &&
        (p.y >= self.o.y) &&
        (p.z >= self.o.z) &&
        (p.x < self.o.x + self.s.x) &&
        (p.y < self.o.y + self.s.y) &&
        (p.z < self.o.z + self.s.z)
    }
}

impl<T: Display> Display for Hyper<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{} {}x{}x{})",self.o.x,self.o.y,self.o.z,self.s.x,self.s.y,self.s.z)
    }
}

#[macro_export]
macro_rules! hyper {
    ($ox:expr,$oy:expr,$oz:expr,$sx:expr,$sy:expr,$sz:expr) => { Hyper::new($ox,$oy,$oz,$sx,$sy,$sz) };
}
