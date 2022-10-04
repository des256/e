use {
    sr::*,
};

mod findtype;
pub use findtype::*;

mod resolvesymbols;
pub use resolvesymbols::*;

mod detuplify;
pub use detuplify::*;

mod destructure;
pub use destructure::*;

mod disenumify;
pub use disenumify::*;

#[cfg(any(gpu="opengl"))]
mod glsl;
#[cfg(any(gpu="opengl"))]
pub use glsl::*;

#[cfg(any(gpu="vulkan"))]
mod spirv;
#[cfg(any(gpu="vulkan"))]
pub use spirv::*;

pub fn translate_module(module: ast::Module) -> ast::Module {

    // resolve all symbol references
    let mut resolver = SymbolResolver::new(&module);
    let module = resolver.resolve_module(module);

    // turn tuples and anonymous tuples into structs
    let mut detuplifier = Detuplifier::new();
    let module = detuplifier.detuplify_module(module);
    panic!("after detuplification:\n{}",module);

    // destructure patterns into local variable declarations
    //let mut destructurer = Destructurer::new();
    //let module = destructurer.destructure_module(module);

    // turn enums into structs
    //let mut disenumifier = Disenumifier::new();
    //disenumifier.disenumify_module(module)

    // now the shader is ready to be translated to the target language
}
