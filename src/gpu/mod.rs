#[derive(Clone,Copy)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Clone,Copy)]
pub enum TextureWrap {
    Black,
    Edge,
    Repeat,
    Mirror,
}

#[derive(Clone,Copy)]
pub enum BlendMode {
    _Replace,
    _Over,
}

#[cfg(feature="gpu_vulkan")]
mod vulkan;
#[cfg(feature="gpu_vulkan")]
pub use vulkan::*;

#[cfg(feature="gpu_opengl")]
mod opengl;
#[cfg(feature="gpu_opengl")]
pub use opengl::*;
