use proc_macro::{
    TokenStream,
    TokenTree,
    Delimiter,
    token_stream::IntoIter,
};

mod ast;
use ast::*;

mod parser;
use parser::*;

mod types;

mod pats;

mod exprs;

mod render;
use render::*;

mod items;

mod resolveidents;
use resolveidents::*;

#[proc_macro_derive(Vertex)]
pub fn derive_vertex(stream: TokenStream) -> TokenStream {
    let (ident,fields) = Parser::new(stream).parse_struct();
    let compiled = render_vertex_trait(&ident,&fields);
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn vertex_shader(attr_stream: TokenStream,item_stream: TokenStream) -> TokenStream {
    let vertex = Parser::new(attr_stream).ident().expect("vertex attribute expected");
    let module = Parser::new(item_stream).parse_module();
    let compiled = render_vertex_shader(module,&vertex);
    panic!("DONE:\n{}",compiled);
    //compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let module = Parser::new(item_stream).parse_module();
    let compiled = render_fragment_shader(module);
    panic!("DONE:\n{}",compiled);
    //compiled.parse().unwrap()
}
