#![feature(proc_macro_span)]
use {
    ast::*,
    std::collections::HashMap,
    proc_macro::{
        TokenStream,
        TokenTree,
        Delimiter,
        token_stream::IntoIter,
    }
};

mod parser;
use parser::*;

mod types;

mod pats;

mod exprs;

mod items;

mod render;
use render::*;

#[proc_macro_derive(Vertex)]
pub fn derive_vertex(stream: TokenStream) -> TokenStream {
    let struct_ = Parser::new(stream).struct_();
    let compiled = format!("impl Vertex for {} {{ fn ast() -> ast::Struct {{ use ast::*; {} }} }}",struct_.ident,struct_.render());
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn vertex_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let module = Parser::new(item_stream).module();
    //panic!("DONE:\n{}",module.render());
    let compiled = format!("pub mod {} {{ pub fn ast() -> ast::Module {{ use ast::*; {} }} }}",module.ident,module.render());
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let module = Parser::new(item_stream).module();
    //panic!("DONE:\n{}",module.render());
    let compiled = format!("pub mod {} {{ pub fn ast() -> ast::Module {{ use ast::*; {} }} }}",module.ident,module.render());
    compiled.parse().unwrap()
}
