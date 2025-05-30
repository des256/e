use {
    crate::*,
    std::{
        cmp::PartialOrd,
        fmt::{Debug, Display, Formatter, Result},
        ops::Add,
    },
};

/// Rectangle.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect<T> {
    pub o: Vec2<T>,
    pub s: Vec2<T>,
}

impl<T> Display for Rect<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({},{} {}x{})", self.o.x, self.o.y, self.s.x, self.s.y)
    }
}

impl<T> Rect<T>
where
    T: Copy + PartialOrd + Add<Output = T>,
{
    /// Test if point is inside rectangle.
    pub fn contains(&self, p: Vec2<T>) -> bool {
        (p.x >= self.o.x)
            && (p.y >= self.o.y)
            && (p.x < self.o.x + self.s.x)
            && (p.y < self.o.y + self.s.y)
    }
}

// if `T as U` exists, `Rect<U>::from(Rect<T>)` should also exist
// generic implementation doesn't work because `From<T> for T` is already defined, so instantiate all of them
macro_rules! rect_from_impl {
    ($(($t:ty,$u:ty))+) => {
        $(
            impl From<Rect<$t>> for Rect<$u> {
                fn from(value: Rect<$t>) -> Self { Rect { o: value.o.into(),s: value.s.into(), } }
            }
        )+
    }
}

rect_from_impl! { (isize,i8) (isize,i16) (isize,i32) (isize,i64) (isize,i128) (isize,f32) (isize,f64) }
rect_from_impl! { (i8,isize) (i8,u16) (i8,i16) (i8,i32) (i8,i64) (i8,i128) (i8,f32) (i8,f64) }
rect_from_impl! { (i16,isize) (i16,i8) (i16,i32) (i16,i64) (i16,i128) (i16,f32) (i16,f64) }
rect_from_impl! { (i32,isize) (i32,i8) (i32,i16) (i32,i64) (i32,i128) (i32,f32) (i32,f64) }
rect_from_impl! { (i64,isize) (i64,i8) (i64,i16) (i64,i32) (i64,i128) (i64,f32) (i64,f64) }
rect_from_impl! { (i128,isize) (i128,i8) (i128,i16) (i128,i32) (i128,i64) (i128,f32) (i128,f64) }
rect_from_impl! { (f32,isize) (f32,i8) (f32,i16) (f32,i32) (f32,i64) (f32,i128) (f32,f64) }
rect_from_impl! { (f64,isize) (f64,i8) (f64,i16) (f64,i32) (f64,i64)(f64,i128) (f64,f32) }
