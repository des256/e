use {
    crate::*,
    std::rc::Rc,
};

pub fn compile_vertex_shader(system: &Rc<System>,items: Vec<sr::Item>,_vertex: &'static str,_vertex_types: Vec<sr::BaseType>) -> Option<VertexShader> {
    println!("create_vertex_shader called with:");
    for item in items {
        println!("{}",item);
    }
    let r: Vec<u32> = vec![0];
    system.create_vertex_shader(&r)
}

pub fn compile_fragment_shader(system: &Rc<System>,items: Vec<sr::Item>) -> Option<FragmentShader> {
    println!("create_fragment_shader called with:");
    for item in items {
        println!("{}",item);
    }
    let r: Vec<u32> = vec![0];
    system.create_fragment_shader(&r)
}
