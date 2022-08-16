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

mod lexer;
use lexer::*;

mod r#struct;
use r#struct::*;

mod render;
use render::*;

#[proc_macro_derive(VertexFormat)]
pub fn derive_vertexformat(stream: TokenStream) -> TokenStream {
    let mut lexer = Lexer::new(stream);
    let item = lexer.is_struct();
    //panic!("RENDERED IMPLEMENTATION: {}",render_struct(&item));
    render_struct(&item).parse().unwrap()
}
