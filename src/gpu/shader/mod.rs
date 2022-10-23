pub mod ast;

mod astdisplay;

mod stdlib;
pub use stdlib::*;

mod destructure;
pub use destructure::*;

mod findtype;
pub use findtype::*;

mod evaluate;
pub use evaluate::*;

mod resolvesymbols;
pub use resolvesymbols::*;

#[cfg(any(gpu="opengl"))]
mod glsl;
#[cfg(any(gpu="opengl"))]
pub use glsl::*;

#[cfg(any(gpu="vulkan"))]
mod spirv;
#[cfg(any(gpu="vulkan"))]
pub use spirv::*;

pub fn translate_source(source: ast::Source) -> ast::Source {

    // TODO: destructure
    // TODO: convert named tuples, convert anonymous tuples, eliminate aliases, convert enums

    source
}
