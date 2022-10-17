pub mod ast;

mod astdisplay;

mod stdlib;
pub use stdlib::*;

mod findtype;
pub use findtype::*;

mod resolver;
pub use resolver::*;

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
