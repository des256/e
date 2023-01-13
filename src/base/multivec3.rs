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

/// 3D Multivector template for geometric algebra.
/// 
/// A 3D Multivector describes the linear combination of a scalar `r`, a vector with three components `x`, `y` and `z` (like
/// [`Vec3`]), three bivectors `xy`, `xz` and `yz` that each describe orientations in orthogonal planes (like [`Quaternion`]),
/// and a trivector `xyz` that describes an oriented volume or imaginary number.
/// 
/// Uses include:
/// 
/// * Various quantities in physics, analysis of electrodynamics, mechanics, torque, angular momentum.
/// * Mathematical equivalents, complex numbers, quaternions.
#[derive(Copy,Clone,Debug)]
pub struct MultiVec3<T> {
    pub r: T, // scalar, weight, etc.
    pub x: T, // position, speed, acceleration, momentum, force, etc.
    pub y: T,
    pub z: T,
    pub xy: T, // orientation, rotation, rotor, torque, angular momentum, magnetic field, etc.
    pub xz: T,
    pub yz: T,
    pub xyz: T, // imaginary number, magnetic flux, etc.
}
