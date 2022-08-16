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

/// Rectangle.
#[derive(Copy,Clone,Debug)]
pub struct Rect<I,U> {
    pub o: Vec2<I>,
    pub s: Vec2<U>,
}


impl<I,U> Rect<I,U> {
    pub fn new(o: Vec2<I>,s: Vec2<U>) -> Rect<I,U> {
        Rect { o: o,s: s, }
    }
}

impl<I: Display,U: Display> Display for Rect<I,U> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{} {}x{})",self.o.x,self.o.y,self.s.x,self.s.y)
    }
}

macro_rules! rect_contains {
    ($i:ty,$u:ty) => {
        impl Rect<$i,$u> {
            pub fn contains(&self,p: Vec2<$i>) -> bool {
                (p.x >= self.o.x) &&
                (p.y >= self.o.y) &&
                (p.x < self.o.x + self.s.x as $i) &&
                (p.y < self.o.y + self.s.y as $i)
            }    
        }
    }
}

rect_contains!(i8,u8);
rect_contains!(i16,u16);
rect_contains!(i32,u32);
rect_contains!(i64,u64);
rect_contains!(i128,u128);
rect_contains!(isize,usize);
rect_contains!(f32,f32);
rect_contains!(f64,f64);

#[allow(non_camel_case_types)]
pub type i8r = Rect<i8,u8>;
#[allow(non_camel_case_types)]
pub type i16r = Rect<i16,u16>;
#[allow(non_camel_case_types)]
pub type i32r = Rect<i32,u32>;
#[allow(non_camel_case_types)]
pub type i64r = Rect<i64,u64>;
#[allow(non_camel_case_types)]
pub type i128r = Rect<i128,u128>;
#[allow(non_camel_case_types)]
pub type isizer = Rect<isize,usize>;
#[allow(non_camel_case_types)]
pub type f32r = Rect<f32,f32>;
#[allow(non_camel_case_types)]
pub type f64r = Rect<f64,f64>;
