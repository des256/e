use {
    crate::*,
};

pub(crate) fn print_token(token: &TokenTree) {
    match token {
        TokenTree::Group(group) => {
            let delimiter = group.delimiter();
            let iterator = group.stream().into_iter();
            match delimiter {
                Delimiter::Parenthesis => print!("("),
                Delimiter::Brace => print!("{{"),
                Delimiter::Bracket => print!("["),
                Delimiter::None => { },
            }
            for token in iterator {
                print_token(&token);
            }
            match delimiter {
                Delimiter::Parenthesis => print!(")"),
                Delimiter::Brace => print!("}}"),
                Delimiter::Bracket => print!("]"),
                Delimiter::None => { },
            }
        },
        TokenTree::Ident(ident) => print!("{}",ident),
        TokenTree::Punct(punct) => print!("{}",punct),
        TokenTree::Literal(literal) => print!("{}",literal),
    }
}
