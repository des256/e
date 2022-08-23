use {
    crate::*,
};

#[allow(dead_code)]
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

pub(crate) struct Parser {
    pub stream: std::iter::Peekable<IntoIter>,
    pub current: Option<TokenTree>,
}

impl Parser {

    /// Create new Parser around a TokenStream.
    pub(crate) fn new(stream: TokenStream) -> Parser {
        let mut stream = stream.into_iter().peekable();
        let current = stream.next();
        Parser { stream,current, }
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
        let result = self.peek_ident(s);
        if result {
            self.consume();
        }
        result
    }

    pub(crate) fn peek_ident(&mut self,s: &str) -> bool {
        if let Some(TokenTree::Ident(ident)) = &self.current {
            ident.to_string() == s
        }
        else {
            false
        }
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
        let result = self.peek_punct(c);
        if result {
            self.consume();
        }
        result
    }

    /// If the current token is a specific punctuation symbol, return true, otherwise return false.
    pub(crate) fn peek_punct(&self,c: char) -> bool {
        if let Some(TokenTree::Punct(punct)) = &self.current {
            punct.as_char() == c
        }
        else {
            false
        }
    }

    /// If the current token is two specific punctuation symbols, consume it and return true. Otherwise return false.
    pub(crate) fn punct2(&mut self,c1: char,c2: char) -> bool {
        let result = self.peek_punct2(c1,c2);
        if result {
            self.consume();
            self.consume();
        }
        result
    }

    /// If the current token is two specific punctuation symbols, return true. Otherwise return false.
    pub(crate) fn peek_punct2(&mut self,c1: char,c2: char) -> bool {
        if let Some(TokenTree::Punct(punct)) = &self.current {
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
        }
    }

    /// If the current token is parenthesis, bracket or brace, consume it and return the sub-lexer in Some(Parser). Otherwise return None.
    pub(crate) fn group(&mut self,open_punct: char) -> Option<Parser> {
        let result = if let Some(TokenTree::Group(group)) = &self.current {
            match open_punct {
                '(' => if group.delimiter() == Delimiter::Parenthesis {
                    Some(Parser::new(group.stream()))
                }
                else {
                    None
                },
                '[' => if group.delimiter() == Delimiter::Bracket {
                    Some(Parser::new(group.stream()))
                }
                else {
                    None
                },
                '{' => if group.delimiter() == Delimiter::Brace {
                    Some(Parser::new(group.stream()))
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

    /// If the current token is parenthesis, bracket or brace, return true, otherwise return false.
    pub(crate) fn peek_group(&self,open_punct: char) -> bool {
        if let Some(TokenTree::Group(group)) = &self.current {
            match open_punct {
                '(' => group.delimiter() == Delimiter::Parenthesis,
                '[' => group.delimiter() == Delimiter::Bracket,
                '{' => group.delimiter() == Delimiter::Brace,
                _ => false,
            }
        }
        else {
            false
        }
    }
}
