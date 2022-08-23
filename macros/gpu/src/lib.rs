use {
    proc_macro::{
        TokenStream,
        TokenTree,
        Delimiter,
        token_stream::IntoIter,
    },
};

mod basetype;
use basetype::*;

mod sr;

mod parser;
use parser::*;

mod types;

mod pats;

mod exprs;

mod stats;

mod items;

#[cfg(gpu="vulkan")]
mod spirv;
#[cfg(gpu="vulkan")]
use spirv::*;

#[cfg(gpu="opengl")]
mod glsl;
#[cfg(gpu="opengl")]
use glsl::*;

mod render;
use render::*;

#[proc_macro_derive(Vertex)]
pub fn derive_vertex(stream: TokenStream) -> TokenStream {
    let item = Parser::new(stream).parse_item();
    render_vertex_trait(item).parse().unwrap()
}

#[proc_macro_attribute]
pub fn vertex_shader(attr_stream: TokenStream,item_stream: TokenStream) -> TokenStream {
    let vertex = Parser::new(attr_stream).any_ident().expect("vertex attribute expected");
    let item = Parser::new(item_stream).parse_item();
    let compiled = compile_vertex_shader(item,vertex);
    panic!("DONE: {}",compiled);
    //compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let item = Parser::new(item_stream).parse_item();
    let compiled = compile_fragment_shader(item);
    panic!("DONE: {}",compiled);
    //compiled.parse().unwrap()
}
