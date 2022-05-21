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

/// Rectangle.
#[derive(Copy,Clone,Debug)]
pub struct Rect<T> {
    pub o: Vec2<T>,
    pub s: Vec2<T>,
}

impl<T: Copy + PartialOrd + Add<T,Output=T>> Rect<T> {
    pub fn new(ox: T,oy: T,sx: T,sy: T) -> Rect<T> {
        Rect { o: vec2!(ox,oy),s: vec2!(sx,sy), }
    }

    pub fn contains(&self,p: &Vec2<T>) -> bool {
        (p.x >= self.o.x) &&
        (p.y >= self.o.y) &&
        (p.x < self.o.x + self.s.x) &&
        (p.y < self.o.y + self.s.y)
    }
}

impl<T: Display> Display for Rect<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{} {}x{})",self.o.x,self.o.y,self.s.x,self.s.y)
    }
}

#[macro_export]
macro_rules! rect {
    ($ox:expr,$oy:expr,$sx:expr,$sy:expr) => { Rect::new($ox,$oy,$sx,$sy) };
}
