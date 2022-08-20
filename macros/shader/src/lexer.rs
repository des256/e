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
    pub stream: std::iter::Peekable<IntoIter>,
    pub current: Option<TokenTree>,
}

impl Lexer {

    /// Create new Lexer around a TokenStream.
    pub(crate) fn new(stream: TokenStream) -> Lexer {
        let mut stream = stream.into_iter().peekable();
        let current = stream.next();
        Lexer { stream,current, }
    }

    /// Consume current token.
    pub(crate) fn consume(&mut self) {
        self.current = self.stream.next();
    }

    /// Return whether or not the stream is done.
    pub(crate) fn done(&self) -> bool {
        if let None = self.current {
            true
        }
        else {
            false
        }
    }

    /// If the current token is a literal, consume it and return Some(String) containing the literal. Otherwise return None.
    pub(crate) fn literal(&mut self) -> Option<String> {
        let result = if let Some(TokenTree::Literal(literal)) = &self.current {
            Some(literal.to_string())
        }
        else {
            None
        };
        if let Some(_) = result {
            self.consume();
        }
        result
    }

    /// If the current token is a specific identifier, consume it and return true. Otherwise return false.
    pub(crate) fn ident(&mut self,s: &str) -> bool {
        let result = if let Some(TokenTree::Ident(ident)) = &self.current {
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
    pub(crate) fn any_ident(&mut self) -> Option<String> {
        let result = if let Some(TokenTree::Ident(ident)) = &self.current {
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
    pub(crate) fn punct(&mut self,c: char) -> bool {
        let result = if let Some(TokenTree::Punct(punct)) = &self.current {
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

    /// If the current token is two specific punctuation symbols, consume it and return true. Otherwise return false.
    pub(crate) fn punct2(&mut self,c1: char,c2: char) -> bool {
        let result = if let Some(TokenTree::Punct(punct)) = &self.current {
            if punct.as_char() == c1 {
                if let Some(TokenTree::Punct(punct)) = &self.stream.peek() {
                    if punct.as_char() == c2 {
                        true
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

    /// If the current token is parenthesis, bracket or brace, consume it and return the sub-lexer in Some(Lexer). Otherwise return None.
    pub(crate) fn group(&mut self,open_punct: char) -> Option<Lexer> {
        let result = if let Some(TokenTree::Group(group)) = &self.current {
            match open_punct {
                '(' => if group.delimiter() == Delimiter::Parenthesis {
                    Some(Lexer::new(group.stream()))
                }
                else {
                    None
                },
                '[' => if group.delimiter() == Delimiter::Bracket {
                    Some(Lexer::new(group.stream()))
                }
                else {
                    None
                },
                '{' => if group.delimiter() == Delimiter::Brace {
                    Some(Lexer::new(group.stream()))
                }
                else {
                    None
                },
                _ => None,
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
