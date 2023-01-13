use {
    crate::*,
    sr::*,
};

mod spirv;

mod buildast;
pub use buildast::*;

mod optimize;
pub use optimize::*;

mod render;
pub use render::*;

pub fn compile_module(mut module: ast::Module) -> Option<Vec<u8>> {
    module = translate_module(module);
    let builder = Builder::new();
    let module = builder.build_module(module);
    let optimizer = Optimizer::new();
    let module = optimizer.optimize_module(module);
    let renderer = Renderer::new();
    renderer.render_module(module)
}
