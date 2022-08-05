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

#[cfg(gpu="vulkan")]
mod vulkan;
#[cfg(gpu="vulkan")]
pub use vulkan::*;

#[cfg(gpu="opengl")]
mod opengl;
#[cfg(gpu="opengl")]
pub use opengl::*;
