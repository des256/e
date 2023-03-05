use {
    crate::sys,
    crate::system::*,
    crate::base::*,
};

mod gpu;
pub use gpu::*;

mod surface;
pub use surface::*;

mod vertexshader;
pub use vertexshader::*;

mod fragmentshader;
pub use fragmentshader::*;

mod vertexbuffer;
pub use vertexbuffer::*;

mod indexbuffer;
pub use indexbuffer::*;

mod pipelinelayout;
pub use pipelinelayout::*;

mod graphicspipeline;
pub use graphicspipeline::*;

mod computepipeline;
pub use computepipeline::*;

mod commandbuffer;
pub use commandbuffer::*;

#[cfg(build="debug")]
#[macro_export]
macro_rules! checkgl {
    ($call:expr) => { { let result = $call; let error = sys::glGetError(); if error != sys::GL_NO_ERROR { dprintln!("DEBUG {}:{}:{}: {} => {}",file!(),line!(),column!(),stringify!($call),gl_error_to_string(error)); } result } };
}

#[cfg(build="release")]
#[macro_export]
macro_rules! checkgl {
    ($call:expr) => { $call };
}

pub fn gl_error_to_string(error: sys::GLenum) -> &'static str {
    match error {
        sys::GL_INVALID_ENUM => "GL_INVALID_ENUM",
        sys::GL_INVALID_VALUE => "GL_INVALID_VALUE",
        sys::GL_INVALID_OPERATION => "GL_INVALID_OPERATION",
        sys::GL_STACK_OVERFLOW => "GL_STACK_OVERFLOW",
        sys::GL_STACK_UNDERFLOW => "GL_STACK_UNDERFLOW",
        sys::GL_OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        sys::GL_INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
        sys::GL_CONTEXT_LOST => "GL_CONTEXT_LOST",
        sys::GL_TABLE_TOO_LARGE => "GL_TABLE_TOO_LARGE",
        _ => "(unknown)",
    }
}
