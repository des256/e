mod detuplify;
pub use detuplify::*;

mod findtype;
pub use findtype::*;

#[cfg(any(gpu="opengl"))]
mod glsl;
#[cfg(any(gpu="opengl"))]
pub use glsl::*;

#[cfg(any(gpu="vulkan"))]
mod spirv;
#[cfg(any(gpu="vulkan"))]
pub use spirv::*;