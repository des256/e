use {
    proc_macro::{
        TokenStream,
        TokenTree,
        Delimiter,
        token_stream::IntoIter,
    },
    sr,
};

mod parser;
use parser::*;

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
    let compiled = render_vertex_shader(item,vertex);
    //panic!("DONE: {}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let item = Parser::new(item_stream).parse_item();
    let compiled = render_fragment_shader(item);
    //panic!("DONE: {}",compiled);
    compiled.parse().unwrap()
}
