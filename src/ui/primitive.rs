use {
    crate::*,
};

pub trait Primitive {
    fn layout(&self,constraints: Constraints);
}
