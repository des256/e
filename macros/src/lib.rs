#![feature(proc_macro_span)]
use proc_macro::{
    TokenStream,
    TokenTree,
    Delimiter,
    token_stream::IntoIter,
};

mod ast;
use ast::*;

mod astdisplay;

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
    let struct_ = Parser::new(stream).struct_().unwrap();
    let compiled = format!("impl Vertex for {} {{ fn ast() -> e::sc::Struct {{ use e::sc::*; {} }} }}",struct_.ident,struct_.render());
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn vertex_shader(arg_stream: TokenStream,item_stream: TokenStream) -> TokenStream {
    let module = Parser::new(item_stream).module().unwrap();
    //panic!("DONE:\n{}",module.render());
    let compiled = format!("pub mod {} {{ pub fn ast() -> e::sc::Module {{ use e::sc::*; {} }} }}",module.ident,module.render_root(arg_stream));
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(arg_stream: TokenStream,item_stream: TokenStream) -> TokenStream {
    let module = Parser::new(item_stream).module().unwrap();
    //panic!("DONE:\n{}",module.render());
    let compiled = format!("pub mod {} {{ pub fn ast() -> e::sc::Module {{ use e::sc::*; {} }} }}",module.ident,module.render_root(arg_stream));
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}
