use {
    proc_macro::{
        TokenStream,
        TokenTree,
        Delimiter,
        token_stream::IntoIter,
        Spacing,
        Group,
    },
    std::fmt,
};

mod ast;
use ast::*;

mod display;
use display::*;

mod lexer;
use lexer::*;

mod segs;
use segs::*;

mod types;
use types::*;

mod pats;
use pats::*;

mod exprs;
use exprs::*;

mod items;
use items::*;

#[proc_macro_attribute]
pub fn shader(attr: TokenStream,item: TokenStream) -> TokenStream {
    let lexer = Lexer::new(item);
    let item = parse_item(&lexer).expect("item expected");
    panic!("done, debug: {}",item);
}
