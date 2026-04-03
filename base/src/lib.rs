//! Foundation crate: async executor, channels, logging, and math.
//!
//! # Async Runtime
//!
//! A single-threaded async executor ([`Executor`]) and multi-producer
//! multi-consumer channel ([`channel`], [`Sender`], [`Receiver`]).
//!
//! # Logging
//!
//! The [`log`] module provides logging macros.
//!
//! # Math
//!
//! Generic math types for graphics, simulation, and geometric algebra.
//!
//! All core types ([`Vec2`], [`Vec3`], [`Vec4`], [`Mat2x2`], [`Mat3x3`], [`Mat4x4`],
//! [`Complex`], [`Quat`], [`Dual`]) are generic over their component type `T`.
//! Operator overloads provide natural arithmetic syntax; concrete `f32`/`f64`
//! impls add transcendental operations (length, normalize, trig, etc.).
//!
//! ## Geometric Algebra
//!
//! [`MultiVec201`] and [`MultiVec301`] implement 2D and 3D projective geometric
//! algebra (PGA). Operators map to PGA products:
//! `*` (geometric), `|` (inner), `^` (outer/wedge), `&` (regressive/join),
//! `!` (dual).
//!
//! ## Rotations
//!
//! 3D rotations are represented by [`Quat`] (quaternion). Construction from
//! axis-angle or [`Euler`] angles is supported, with conversions to [`Mat3x3`]
//! and [`Mat4x4`]. [`Pose`] combines position and orientation.
//!
//! ## Utilities
//!
//! - Interpolation: [`lerp`], [`Quat::slerp`], [`Quat::nlerp`]
//! - Clamping: [`clamp`], [`saturate_f32`], [`smoothstep_f32`], [`remap_f32`]
//! - Trig helpers: [`radians_f32`], [`degrees_f32`], [`sincos_f32`]
//! - Bounding volumes: [`Aabb2`], [`Aabb3`]

// -- async runtime --

/// Logging macros and utilities.
pub mod log;

mod executor;
pub use executor::*;

mod channel;
pub use channel::*;

mod epoch;
pub use epoch::*;

// -- serial port --

mod serial;
pub use serial::*;

// -- math: algebraic identity traits --

mod zero;
pub use zero::*;

mod one;
pub use one::*;

// -- number types --

mod f16;
pub use f16::*;

// -- math: number types --

mod complex;
pub use complex::*;

mod quat;
pub use quat::*;

mod euler;
pub use euler::*;

mod dual;
pub use dual::*;

// -- math: vectors --

mod vec2;
pub use vec2::*;

mod vec3;
pub use vec3::*;

mod vec4;
pub use vec4::*;

// -- math: matrices --

mod mat2x2;
pub use mat2x2::*;

mod mat3x3;
pub use mat3x3::*;

mod mat4x4;
pub use mat4x4::*;

// -- math: geometric algebra --

mod multivec201;
pub use multivec201::*;

mod multivec301;
pub use multivec301::*;

// -- math: geometry --

mod pose;
pub use pose::*;

mod fov;
pub use fov::*;

mod aabb2;
pub use aabb2::*;

mod aabb3;
pub use aabb3::*;

// -- tensors --

mod tensor;
pub use tensor::*;

// -- math: utilities --

mod interp;
pub use interp::*;

mod clamp;
pub use clamp::*;

mod trig;
pub use trig::*;
