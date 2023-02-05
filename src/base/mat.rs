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
        }
    }
};

#[derive(Copy,Clone,Debug)]
pub struct Mat<T> {
    r: usize,
    c: usize,
    v: Vec<T>,
}
