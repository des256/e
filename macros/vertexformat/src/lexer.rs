use {
    crate::*,
};

pub(crate) struct Lexer {
    pub token: Option<TokenTree>,
    pub stream: IntoIter,
}

impl Lexer {
    pub(crate) fn new(stream: TokenStream) -> Lexer {
        let mut stream = stream.into_iter();
        let token = stream.next();
        Lexer {
            token,
            stream,
        }
    }

    pub(crate) fn step(&mut self) {
        self.token = self.stream.next();
    }

    pub(crate) fn is_ident(&mut self,s: &str) -> bool {
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
            self.step();
        }
        result
    }

    pub(crate) fn is_some_ident(&mut self) -> Option<String> {
        let result = if let Some(TokenTree::Ident(ident)) = &self.token {
            Some(ident.to_string())
        }
        else {
            None
        };
        if let Some(_) = result {
            self.step();
        }
        result
    }

    pub(crate) fn is_punct(&mut self,c: char) -> bool {
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
            self.step();
        }
        result
    }

    pub(crate) fn is_paren_group(&mut self) -> Option<Group> {
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
            self.step();
        }
        result
    }

    pub(crate) fn is_bracket_group(&mut self) -> Option<Group> {
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
            self.step();
        }
        result
    }

    pub(crate) fn is_brace_group(&mut self) -> Option<Group> {
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
            self.step();
        }
        result
    }
}
