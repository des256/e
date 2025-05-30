use std::fmt::{Debug, Display, Formatter, Result};

/// FOV specification.
#[derive(Copy, Clone, Debug)]
pub struct Fov<T> {
    pub l: T,
    pub r: T,
    pub b: T,
    pub t: T,
}

impl<T> Display for Fov<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({}..{}, {}..{})", self.l, self.r, self.b, self.t)
    }
}
