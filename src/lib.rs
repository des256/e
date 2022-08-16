#![feature(const_trait_impl)]
#![feature(const_fn_floating_point_arithmetic)]

mod sys {
    #![allow(non_camel_case_types)]
    #![allow(non_upper_case_globals)]
    #![allow(non_snake_case)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"),"/sys.rs"));
}

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

mod base;
pub use base::*;

mod math;
pub use math::*;

mod system;
pub use system::*;

mod gpu;
pub use gpu::*;

mod codecs;
pub use codecs::*;

mod ui;
pub use ui::*;
