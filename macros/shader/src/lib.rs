use {
    proc_macro::{
        TokenStream,
        TokenTree,
        Delimiter,
        token_stream::IntoIter,
    },
};

mod ast;

mod parser;
use parser::*;

mod types;
use types::*;

mod pats;
use pats::*;

mod exprs;
use exprs::*;

mod stats;
use stats::*;

mod items;
use items::*;

#[proc_macro_attribute]
pub fn vertex_shader(attr_stream: TokenStream,item_stream: TokenStream) -> TokenStream {

    // get uniform structure, vertex struct and varying struct
    let mut attr_parser = Parser::new(attr_stream);
    let uniforms = attr_parser.any_ident().expect("uniform expected");
    attr_parser.punct(',');
    let vertex = attr_parser.any_ident().expect("vertex expected");
    attr_parser.punct(',');
    let varying = attr_parser.any_ident().expect("varying expected");

    // parse module or function into AST
    let item = Parser::new(item_stream).parse_item();

    panic!("done: {}",item);
}
