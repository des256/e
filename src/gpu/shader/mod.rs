pub mod ast;

mod astdisplay;

mod stdlib;
pub use stdlib::*;

mod context;
pub use context::*;

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

pub fn translate_source(source: ast::Source) -> ast::Module {

    println!("resolve into C-like");

    let mut resolver = Resolver::new(&source);
    let module = resolver.resolve(source);

    // now the shader is ready to be translated to the target language

    module
}
