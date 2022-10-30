use super::*;

mod glsl;

mod buildast;
pub use buildast::*;

mod optimize;
pub use optimize::*;

mod render;
pub use render::*;

pub fn compile_module(module: ast::RustModule) -> Option<Vec<u8>> {
    
    println!("module right after parsing:\n{}",module);
    
    // destructure patterns, resolve symbols, translate tuples, translate enums, translate aliases
    let module = translate_module(module);

    println!("module before conversion to GLSL:\n{}",module);
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
