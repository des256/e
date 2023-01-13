mod half;
pub use half::*;

pub trait Zero {
    const ZERO: Self;
}

impl Zero for u8 { const ZERO: u8 = 0; }
impl Zero for u16 { const ZERO: u16 = 0; }
impl Zero for u32 { const ZERO: u32 = 0; }
impl Zero for u64 { const ZERO: u64 = 0; }
impl Zero for u128 { const ZERO: u128 = 0; }
impl Zero for usize { const ZERO: usize = 0; }
impl Zero for i8 { const ZERO: i8 = 0; }
impl Zero for i16 { const ZERO: i16 = 0; }
impl Zero for i32 { const ZERO: i32 = 0; }
impl Zero for i64 { const ZERO: i64 = 0; }
impl Zero for i128 { const ZERO: i128 = 0; }
impl Zero for isize { const ZERO: isize = 0; }
impl Zero for f16 { const ZERO: f16 = f16(0); }
impl Zero for f32 { const ZERO: f32 = 0.0; }
impl Zero for f64 { const ZERO: f64 = 0.0; }

pub trait One {
    const ONE: Self;
}

impl One for u8 { const ONE: u8 = 1; }
impl One for u16 { const ONE: u16 = 1; }
impl One for u32 { const ONE: u32 = 1; }
impl One for u64 { const ONE: u64 = 1; }
impl One for u128 { const ONE: u128 = 1; }
impl One for usize { const ONE: usize = 1; }
impl One for i8 { const ONE: i8 = 1; }
impl One for i16 { const ONE: i16 = 1; }
impl One for i32 { const ONE: i32 = 1; }
impl One for i64 { const ONE: i64 = 1; }
impl One for i128 { const ONE: i128 = 1; }
impl One for isize { const ONE: isize = 1; }
impl One for f16 { const ONE: f16 = f16(1); }
impl One for f32 { const ONE: f32 = 1.0; }
impl One for f64 { const ONE: f64 = 1.0; }

mod unsigned;
pub use unsigned::*;

mod signed;
pub use signed::*;

mod rational;
pub use rational::*;

mod real;
pub use real::*;

mod float;
pub use float::*;

mod fixed;
pub use fixed::*;

mod complex;
pub use complex::*;

mod mat;
pub use mat::*;

mod ten;
pub use ten::*;

mod vec2;
pub use vec2::*;

mod vec3;
pub use vec3::*;

mod vec4;
pub use vec4::*;

mod mat2x2;
pub use mat2x2::*;

mod mat2x3;
pub use mat2x3::*;

mod mat2x4;
pub use mat2x4::*;

mod mat3x2;
pub use mat3x2::*;

mod mat3x3;
pub use mat3x3::*;

mod mat3x4;
pub use mat3x4::*;

mod mat4x2;
pub use mat4x2::*;

mod mat4x3;
pub use mat4x3::*;

mod mat4x4;
pub use mat4x4::*;

mod multivec2;
pub use multivec2::*;

mod multivec3;
pub use multivec3::*;

mod multivec4;
pub use multivec4::*;

mod color;
pub use color::*;

mod rect;
pub use rect::*;

mod hyper;
pub use hyper::*;

mod pose;
pub use pose::*;

mod quaternion;
pub use quaternion::*;

mod executor;
pub use executor::*;

mod timer;
pub use timer::*;

mod pixel;
pub use pixel::*;
