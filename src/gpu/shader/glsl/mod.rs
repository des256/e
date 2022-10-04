use {
    crate::*,
    sr::*,
};

mod glsl;

mod buildast;
pub use buildast::*;

mod optimize;
pub use optimize::*;

mod render;
pub use render::*;

pub fn compile_module(mut module: ast::Module) -> Option<Vec<u8>> {
    
    // resolve symbols, detuplify, destructure patterns and disenumify
    module = translate_module(module);

    /*
    // translate to GLSL-specific AST
    let builder = Builder::new();
    let module = builder.build_module(module);

    // optimize GLSL
    let optimizer = Optimizer::new();
    let module = optimizer.optimize_module(module);

    // render to text output
    let renderer = Renderer::new();
    renderer.render_module(module)
    */
    None
}
