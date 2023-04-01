use {
    super::*,
    std::mem::swap,
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
    pub stream: IntoIter,
    pub current: Option<TokenTree>,
    pub next: Option<TokenTree>,
    pub nextnext: Option<TokenTree>,
}

impl Parser {

    /// Create new Parser around a TokenStream.
    pub(crate) fn new(item_stream: TokenStream) -> Parser {
        let mut stream = item_stream.into_iter();
        let current = stream.next();
        let next = stream.next();
        let nextnext = stream.next();
        Parser {
            stream,
            current,
            next,
            nextnext,
        }
    }

    /// Consume current token.
    pub(crate) fn consume(&mut self) {
        swap(&mut self.current,&mut self.next);
        swap(&mut self.next,&mut self.nextnext);
        self.nextnext = self.stream.next();
    }

    pub(crate) fn done(&self) -> bool {
        if let None = self.current {
            true
        }
        else {
            false
        }
    }

    pub(crate) fn err<T>(&self,message: &str) -> Result<T,String> {
        let current = Option::<TokenTree>::clone(&self.current).unwrap();
        let path = current.span().source_file().path();
        let source_file = path.to_str().unwrap();
        let start = current.span().start();
        Err(format!("{}:{},{}: {}",source_file,start.line,start.column,message))
    }

    pub(crate) fn peek_keyword(&mut self,s: &str) -> bool {
        if let Some(TokenTree::Ident(ident)) = &self.current {
            ident.to_string() == s
        }
        else {
            false
        }
    }

    pub(crate) fn keyword(&mut self,s: &str) -> bool {
        let result = self.peek_keyword(s);
        if result {
            self.consume();
        }
        result
    }

    pub(crate) fn ident(&mut self) -> Option<String> {
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

    pub(crate) fn peek_punct(&self,c: char) -> bool {
        if let Some(TokenTree::Punct(punct)) = &self.current {
            punct.as_char() == c
        }
        else {
            false
        }
    }

    pub(crate) fn punct(&mut self,c: char) -> bool {
        let result = self.peek_punct(c);
        if result {
            self.consume();
        }
        result
    }

    pub(crate) fn peek_punct2(&mut self,c1: char,c2: char) -> bool {
        if let Some(TokenTree::Punct(punct)) = &self.current {
            if punct.as_char() == c1 {
                if let Some(TokenTree::Punct(punct)) = &self.next {
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

    pub(crate) fn punct2(&mut self,c1: char,c2: char) -> bool {
        let result = self.peek_punct2(c1,c2);
        if result {
            self.consume();
            self.consume();
        }
        result
    }

    pub(crate) fn peek_punct3(&mut self,c1: char,c2: char,c3: char) -> bool {
        if let Some(TokenTree::Punct(punct)) = &self.current {
            if punct.as_char() == c1 {
                if let Some(TokenTree::Punct(punct)) = &self.next {
                    if punct.as_char() == c2 {
                        if let Some(TokenTree::Punct(punct)) = &self.nextnext {
                            punct.as_char() == c3
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
            }
        }
        else {
            false
        }
    }

    pub(crate) fn punct3(&mut self,c1: char,c2: char,c3: char) -> bool {
        let result = self.peek_punct3(c1,c2,c3);
        if result {
            self.consume();
            self.consume();
            self.consume();
        }
        result
    }

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

    // VARIOUS LISTS
    pub(crate) fn ident_types(&mut self) -> Result<Vec<(String,Type)>,String> {
        let mut ident_types: Vec<(String,Type)> = Vec::new();
        while !self.done() {
            self.keyword("pub");  // just skip it if it occurs here
            let ident = self.ident();
            if ident.is_none() {
                return self.err("identifier expected");
            }
            let ident = ident.unwrap();
            if !self.punct(':') {
                return self.err(": expected");
            }
            let type_ = self.type_()?;
            ident_types.push((ident,type_));
            self.punct(',');
        }
        Ok(ident_types)
    }

    pub(crate) fn paren_ident_types(&mut self) -> Result<Vec<(String,Type)>,String> {
        if let Some(mut parser) = self.group('(') {
            parser.ident_types()
        }
        else {
            self.err("( expected")
        }
    }

    pub(crate) fn brace_ident_types(&mut self) -> Result<Vec<(String,Type)>,String> {
        if let Some(mut parser) = self.group('{') {
            parser.ident_types()
        }
        else {
            self.err("{{ expected (brace_ident_types)")
        }
    }

    fn types(&mut self) -> Result<Vec<Type>,String> {
        let mut types: Vec<Type> = Vec::new();
        while !self.done() {
            let type_ = self.type_()?;
            types.push(type_);
            self.punct(',');
        }
        Ok(types)
    }

    pub(crate) fn paren_types(&mut self) -> Result<Option<Vec<Type>>,String> {
        if let Some(mut parser) = self.group('(') {
            Ok(Some(parser.types()?))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn exprs(&mut self) -> Result<Vec<Expr>,String> {
        let mut exprs: Vec<Expr> = Vec::new();
        while !self.done() {
            let expr = self.expr()?;
            exprs.push(expr);
            self.punct(',');
        }
        Ok(exprs)
    }

    pub(crate) fn paren_exprs(&mut self) -> Result<Option<Vec<Expr>>,String> {
        if let Some(mut parser) = self.group('(') {
            Ok(Some(parser.exprs()?))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn brace_ident_exprs(&mut self) -> Result<Option<Vec<(String,Expr)>>,String> {
        let mut ident_exprs: Vec<(String,Expr)> = Vec::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {
                let ident = parser.ident();
                if ident.is_none() {
                    return self.err("identifier expected");
                }
                let ident = ident.unwrap();
                if !parser.punct(':') {
                    return self.err(": expected");
                }
                let expr = parser.expr()?;
                ident_exprs.push((ident,expr));
                parser.punct(',');
            }
            Ok(Some(ident_exprs))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn brace_ident_pats(&mut self) -> Result<Option<Vec<FieldPat>>,String> {
        let mut ident_pats: Vec<FieldPat> = Vec::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {
                let ident_pat = if let Some(ident) = parser.ident() {
                    if parser.punct(':') {
                        FieldPat::IdentPat(ident,parser.pat()?)
                    }
                    else {
                        FieldPat::Ident(ident)
                    }
                }
                else if parser.punct('_') {
                    FieldPat::Wildcard
                }
                else if parser.punct2('.','.') {
                    FieldPat::Rest
                }
                else {
                    return self.err("identifier, _ or .. expected");
                };
                ident_pats.push(ident_pat);
                parser.punct(',');
            }
            Ok(Some(ident_pats))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn pats(&mut self) -> Result<Vec<Pat>,String> {
        let mut pats: Vec<Pat> = Vec::new();
        while !self.done() {
            pats.push(self.pat()?);
            self.punct(',');
        }
        Ok(pats)
    }

    pub(crate) fn paren_pats(&mut self) -> Result<Option<Vec<Pat>>,String> {
        if let Some(mut parser) = self.group('(') {
            Ok(Some(parser.pats()?))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn bracket_pats(&mut self) -> Result<Option<Vec<Pat>>,String> {
        if let Some(mut parser) = self.group('[') {
            Ok(Some(parser.pats()?))
        }
        else {
            Ok(None)
        }
    }

    pub(crate) fn boolean_literal(&mut self) -> Option<bool> {
        if self.keyword("true") {
            self.consume();
            Some(true)
        }
        else if self.keyword("false") {
            self.consume();
            Some(false)
        }
        else {
            None
        }
    }

    pub(crate) fn integer_literal(&mut self) -> Option<u64> {
        if let Some(TokenTree::Literal(literal)) = &self.current {
            if let Ok(value) = literal.to_string().parse::<u64>() {
                self.consume();
                Some(value)
            }
            else {
                None
            }
        }
        else {
            None
        }
    }

    pub(crate) fn float_literal(&mut self) -> Option<f64> {
        if let Some(TokenTree::Literal(literal)) = &self.current {
            if let Ok(value) = literal.to_string().parse::<f64>() {
                self.consume();
                Some(value)
            }
            else {
                None
            }
        }
        else {
            None
        }
    }

    /*
    pub(crate) fn make_anon_tuple_struct(&mut self,types: Vec<Type>) -> String {
        for (ident,fields) in &self.anon_tuple_structs {
            if fields.len() == types.len() {
                let mut all_types_match = true;
                for i in 0..types.len() {
                    if fields[i].1 != types[i] {
                        all_types_match = false;
                        break;
                    }
                }
                if all_types_match {
                    return ident.clone();
                }
            }
        }
        let mut fields: Vec<(String,Type)> = Vec::new();
        let mut i = 0usize;
        for ty in types {
            fields.push((format!("_{}",i),ty));
            i += 1;
        }
        let ident = format!("AnonTuple{}",self.anon_tuple_structs.len());
        self.anon_tuple_structs.insert(ident.clone(),fields);
        ident
    }
    */
}
