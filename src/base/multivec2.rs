use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Debug,
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

/// 2D Multivector template for geometric algebra.
/// 
/// A 2D Multivector describes the linear combination of a scalar `r`, a vector with components `x` and `y` (like ['Vec2']),
/// and a bivector `xy` that describes an orientation or area, or imaginary number (`r` and `xy` together are like [`Complex`]).
#[derive(Copy,Clone,Debug)]
pub struct MultiVec2<T> {
    pub r: T,
    pub x: T,
    pub y: T,
    pub xy: T,
}
