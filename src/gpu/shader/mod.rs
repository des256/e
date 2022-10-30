pub mod ast;

mod astdisplay;

mod stdlib;
pub use stdlib::*;

mod destructure;
pub use destructure::*;

mod convert;
pub use convert::*;

mod resolve;
pub use resolve::*;

#[cfg(any(gpu="opengl"))]
mod glsl;
#[cfg(any(gpu="opengl"))]
pub use glsl::*;

#[cfg(any(gpu="vulkan"))]
mod spirv;
#[cfg(any(gpu="vulkan"))]
pub use spirv::*;

pub fn translate_module(module: ast::RustModule) -> ast::Module {

    // destructure all pattern nodes into regular boolean and field expressions
    let module = destructure_module(module);

    // convert tuples and enums to structs
    // convert aliases to their types
    let module = convert_module(module);

    // resolve anonymous tuples
    // resolve Expr:Discriptor and Expr::Destructure nodes
    let module = resolve_module(module);

    // what's left has no more patterns, tuples or enums and should be easily translatable into the target language
    module
}
