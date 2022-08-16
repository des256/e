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

//mod grammar;
//use grammar::*;

mod lexer;
use lexer::*;

#[proc_macro_attribute]
pub fn shader(attr: TokenStream,item: TokenStream) -> TokenStream {
    let iterator = item.into_iter();
    for token in iterator {
        print_token(&token);
    }
    panic!("done, debug.");
}
