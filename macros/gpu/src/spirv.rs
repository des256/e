use crate::*;

pub fn compile_vertex_shader(item: sr::Item,vertex: String) -> String {
    "[0x07230203,0x00010000,0x00080001]".to_string()
}

pub fn compile_fragment_shader(item: sr::Item) -> String {
    "[0x07230203,0x00010000,0x00080001]".to_string()
}
