use {
    sr::*,
};

pub fn translate_module(mut module: ast::Module) -> ast::Module {
    // TODO: resolve symbols
    // TODO: translate tuples to structs
    // TODO: destructure patterns
    // TODO: translate enums to structs
    module
}
