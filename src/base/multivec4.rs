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

/// 4D Multivector template for geometric algebra.
/// 
/// A 4D Multivector describes the linear combination of a scalar `r`, four vectors `x`, `y`, `z` and `w` that describe
/// directions, six bivectors `xy`, `xz`, `xw`, `yz`, `yw` and `zw` that each describe an orientation on a surface, four
/// pseudovectors `xyz`, `xyw`, `xzw` and `yzw` which describe oriented volumes, and a pseudoscalar `xyzw` that describes ...
#[derive(Copy,Clone,Debug)]
pub struct MultiVec4<T> {
    pub r: T,
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
    pub xy: T,
    pub xz: T,
    pub xw: T,
    pub yz: T,
    pub yw: T,
    pub zw: T,
    pub xyz: T,
    pub xzw: T,
    pub xyw: T,
    pub yzw: T,
    pub xyzw: T,
}
