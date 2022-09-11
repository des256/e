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

mod render;
use render::*;

mod items;

mod resolveconsts;
use resolveconsts::*;

mod unfoldpatterns;
use unfoldpatterns::*;

#[proc_macro_derive(Vertex)]
pub fn derive_vertex(stream: TokenStream) -> TokenStream {
    let (ident,fields) = Parser::new(stream).parse_struct();
    let compiled = render_vertex_trait(&ident,&fields);
    //panic!("DONE:\n{}",compiled);
    compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn vertex_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let mut module = Parser::new(item_stream).module();
    resolve_consts(&mut module);
    unfold_patterns(&mut module);
    panic!("DONE: {}",module);
    //let compiled = render_vertex_shader(module);
    //panic!("DONE:\n{}",compiled);
    //compiled.parse().unwrap()
}

#[proc_macro_attribute]
pub fn fragment_shader(_: TokenStream,item_stream: TokenStream) -> TokenStream {
    let mut module = Parser::new(item_stream).module();
    resolve_consts(&mut module);
    unfold_patterns(&mut module);
    panic!("DONE: {}",module);
    //let compiled = render_fragment_shader(module);
    //panic!("DONE:\n{}",compiled);
    //compiled.parse().unwrap()
}
