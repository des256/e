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

mod grammar;
use grammar::*;

mod display;
use display::*;

mod lexer;
use lexer::*;

mod visattr;
use visattr::*;

mod item;
use item::*;

mod expr;
use expr::*;

mod r#type;
use r#type::*;

#[proc_macro_attribute]
pub fn shader(attr: TokenStream,item: TokenStream) -> TokenStream {
    let item = Lexer::new(item).parse_item().unwrap();
    panic!("done, debug: {}",item);
}
