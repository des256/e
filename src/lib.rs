#![feature(const_trait_impl)]
#![feature(const_fn_floating_point_arithmetic)]

//! # E
//! 
//! This is E.
//!
//! ## Base Structs and Traits
//! 
//! [`Rect`]
//! 
//! ### Numbers
//! 
//! [`Unsigned`] numbers, [`Signed`] numbers,
//! 
//! [`Rational`] numbers,
//! 
//! [`Real`] numbers ([`Float`], [`Fixed`]),
//! 
//! [`Complex`] numbers.
//! 
//! ### Helper Traits
//! 
//! Additive identity [`Zero`], multiplicative identity [`One`].
//! 
//! Color, Pixel encodings, SIMD support.
//! 
//! ### Vectors
//! 
//! Generic [`VecN`].
//! 
//! [`Vec2`], [`Vec3`], [`Vec4`].
//! 
//! ### Matrices
//! 
//! Generic [`Mat`].
//! 
//! [`Mat2x2`], [`Mat2x3`], [`Mat2x4`],
//! 
//! [`Mat3x2`], [`Mat3x3`], [`Mat3x4`],
//!
//! [`Mat4x2`], [`Mat4x3`], [`Mat4x4`].
//! 
//! ### Tensors
//! 
//! Generic [`Ten`]
//! 
//! ### More Advanced Algebra
//! 
//! Orientation 3D/VR [`Quaternion`], [`Pose`].
//! 
//! Geometric/Clifford [`MultiVec2`], [`MultiVec3`], [`MultiVec4`].
//! 
//! ## System Functionality
//! 
//! The system accessor [`open_system`], and [`System`] struct.
//! 
//! ## Executors
//! 
//! ### Implementation Details
//! 
//! Linux
//! Windows
//! Android
//! iOS
//! MacOS
//! (Web)
//! 
//! ## GPU
//! 
//! The GPU accessor [`open_system_gpu`], and [`GPU`] struct.
//! 
//! ## Codecs
//! 
//! ### Image Codecs
//! 
//! ### Audio Codecs
//! 
//! ### Video Codecs
//! 
//! ## User Interface
//! 

#[cfg(build="debug")]
#[macro_export]
macro_rules! dprintln {
    ($($arg:tt)*) => { println!("DEBUG: {}",std::format_args!($($arg)*)) };
}

#[cfg(build="release")]
#[macro_export]
macro_rules! dprintln {
    ($($arg:tt)*) => { };
}

#[doc(hidden)]
mod sys;

mod base;
pub use base::*;

mod system;
pub use system::*;

mod gpu;
pub use gpu::*;

mod codecs;
pub use codecs::*;

mod ui;
pub use ui::*;
