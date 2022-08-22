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

mod render;
use render::*;

#[proc_macro_attribute]
pub fn shader(attr_stream: TokenStream,item_stream: TokenStream) -> TokenStream {

    let mut parser = Parser::new(attr_stream);

    let shader_type = parser.any_ident().expect("shader type (vertex, fragment or geometry) expected");
    parser.punct(',');
    match shader_type.as_str() {
        "vertex" => {
            let uniform = parser.any_ident().expect("uniform type expected");
            parser.punct(',');
            let vertex = parser.any_ident().expect("vertex type expected");
            parser.punct(',');
            let varying = parser.any_ident().expect("varying type expected");
            let item = Parser::new(item_stream).parse_item();
            panic!(format!("item: {}",item));
            render_vertex_shader(item,uniform,vertex,varying).parse().unwrap()
        },
        _ => {
            panic!("only vertex shader supported for now");
        },
    }
}
