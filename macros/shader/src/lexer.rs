use {
    crate::*,
};

pub(crate) fn debug_token(token: &TokenTree) -> String {
    let mut debug = String::new();
    match token {
        TokenTree::Group(group) => {
            let delimiter = group.delimiter();
            let iterator = group.stream().into_iter();
            match delimiter {
                Delimiter::Parenthesis => debug += "(",
                Delimiter::Brace => debug += "{",
                Delimiter::Bracket => debug += "[",
                Delimiter::None => { },
            }
            for token in iterator {
                debug += &debug_token(&token);
            }
            match delimiter {
                Delimiter::Parenthesis => debug += ")",
                Delimiter::Brace => debug += "}",
                Delimiter::Bracket => debug += "]",
                Delimiter::None => { },
            }
        },
        TokenTree::Ident(ident) => debug += &ident.to_string(),
        TokenTree::Punct(punct) => debug += &punct.to_string(),
        TokenTree::Literal(literal) => debug += &literal.to_string(),
    }
    debug
}

pub(crate) struct Lexer {
    pub token: Option<TokenTree>,
    pub stream: std::iter::Peekable<IntoIter>,
}

impl Lexer {
    /// Create new Lexer around a TokenStream.
    pub(crate) fn new(stream: TokenStream) -> Lexer {
        let mut stream = stream.into_iter().peekable();
        let token = stream.next();
        Lexer {
            token,
            stream,
        }
    }

    /// Consume current token.
    pub(crate) fn consume(&mut self) {
        self.token = self.stream.next();
    }

    /// If the current token is a specific identifier, consume it and return true. Otherwise return false.
    pub(crate) fn parse_ident(&mut self,s: &str) -> bool {
        let result = if let Some(TokenTree::Ident(ident)) = &self.token {
            if ident.to_string() == s {
                true
            }
            else {
                false
            }
        }
        else {
            false
        };
        if result {
            self.consume();
        }
        result
    }

    /// If the current token is an identifier, consume it and return Some(String) containing the identifier. Otherwise return None.
    pub(crate) fn parse_any_ident(&mut self) -> Option<String> {
        let result = if let Some(TokenTree::Ident(ident)) = &self.token {
            Some(ident.to_string())
        }
        else {
            None
        };
        if let Some(_) = result {
            self.consume();
        }
        result
    }

    /// If the current token is a specific punctuation symbol, consume it and return true. Otherwise return false.
    pub(crate) fn parse_punct(&mut self,c: char) -> bool {
        let result = if let Some(TokenTree::Punct(punct)) = &self.token {
            if punct.as_char() == c {
                true
            }
            else {
                false
            }
        }
        else {
            false
        };
        if result {
            self.consume();
        }
        result
    }

    /// If the current token is two specific punctuation symbols, consume them and return true. Otherwise return false.
    pub(crate) fn parse_punct2(&mut self,c1: char,c2: char) -> bool {
        let result = if let Some(TokenTree::Punct(punct)) = &self.token {
            if punct.as_char() == c1 {
                if let Some(TokenTree::Punct(punct)) = &self.stream.peek() {
                    punct.as_char() == c2
                }
                else {
                    false
                }
            }
            else {
                false
            }
        }
        else {
            false
        };
        if result {
            self.consume();
            self.consume();
        }
        result
    }

    /// If the current token is the open parenthesis, consume it and return the group in Some(Group). Otherwise return None.
    pub(crate) fn parse_paren(&mut self) -> Option<Group> {
        let result = if let Some(TokenTree::Group(group)) = &self.token {
            if group.delimiter() == Delimiter::Parenthesis {
                Some(group.clone())
            }
            else {
                None
            }
        }
        else {
            None
        };
        if let Some(_) = result {
            self.consume();
        }
        result
    }

    /// If the current token is the open bracket, consume it and return the group in Some(Group). Otherwise return None.
    pub(crate) fn parse_bracket(&mut self) -> Option<Group> {
        let result = if let Some(TokenTree::Group(group)) = &self.token {
            if group.delimiter() == Delimiter::Bracket {
                Some(group.clone())
            }
            else {
                None
            }
        }
        else {
            None
        };
        if let Some(_) = result {
            self.consume();
        }
        result
    }

    /// If the current token is the open brace, consume it and return the group in Some(Group). Otherwise return None.
    pub(crate) fn parse_brace(&mut self) -> Option<Group> {
        let result = if let Some(TokenTree::Group(group)) = &self.token {
            if group.delimiter() == Delimiter::Brace {
                Some(group.clone())
            }
            else {
                None
            }
        }
        else {
            None
        };
        if let Some(_) = result {
            self.consume();
        }
        result
    }
}
