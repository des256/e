use {
    crate::*,
};

pub fn create_vertex_shader(item: ast::Item,uniforms: String,vertex: String,varying: String) -> String {
    let mut shader = String::new();
    shader.add("#version 450\n");
    // TODO: put uniform struct in uniforms
    // TODO: put vertex format struct in ins
    // TODO: put varying format struct in outs
    // TODO: put other functions in functions
    shader.add("void main() {\n");
    // TODO: render main function
    // TODO: map the result to the right variables (outs and gl_Position)
    shader.add("}\n");
    shader
}
