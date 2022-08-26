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

    // ITEMS

    pub fn parse_item(&mut self) -> sr::Item {
        if self.ident("pub") {
            self.group('(');
        }
        if self.ident("mod") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut items: Vec<sr::Item> = Vec::new();
            if let Some(mut parser) = self.group('{') {
                while !parser.done() {
                    items.push(parser.parse_item());
                }
            }
            else {
                self.punct(';');
            }
            sr::Item::Module(symbol,items)
        }
        else if self.ident("fn") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut params: Vec<(sr::Pat,Box<sr::Type>)> = Vec::new();
            if let Some(mut parser) = self.group('(') {
                while !parser.done() {
                    let pat = parser.parse_pat();
                    parser.punct(':');
                    let ty = parser.parse_type();
                    params.push((pat,Box::new(ty)));
                    parser.punct(';');
                }
            }
            let return_ty = if self.punct2('-','>') {
                Some(Box::new(self.parse_type()))
            }
            else {
                None
            };
            let mut stats: Vec<sr::Stat> = Vec::new();
            let mut parser = self.group('{').expect("{ expected");
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            sr::Item::Function(symbol,params,return_ty,stats)
        }
        else if self.ident("struct") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut fields: Vec<(String,Box<sr::Type>)> = Vec::new();
            let mut parser = self.group('{').expect("{ expected");
            while !parser.done() {
                if parser.ident("pub") {
                    parser.group('(');
                }
                let symbol = parser.any_ident().expect("identifier expected");
                parser.punct(':');
                let ty = parser.parse_type();
                fields.push((symbol,Box::new(ty)));
                parser.punct(',');
            }
            sr::Item::Struct(symbol,fields)
        }
        else {
            panic!("item not supported");
        }
    }

    // STATEMENTS

    pub fn parse_stat(&mut self) -> sr::Stat {
        if self.ident("let") {
            let pat = self.parse_pat();
            let ty = if self.punct(':') {
                Some(Box::new(self.parse_type()))
            }
            else {
                None
            };
            let expr = if self.punct('=') {
                Some(Box::new(self.parse_expr()))
            }
            else {
                None
            };
            sr::Stat::Let(pat,ty,expr)
        }
        else {
            let expr = self.parse_expr();
            self.punct(';');
            sr::Stat::Expr(Box::new(expr))
        }
    }

    // EXPRESSIONS

    pub fn parse_primary_expr(&mut self) -> sr::Expr {
        if let Some(literal) = self.literal() {
            sr::Expr::Literal(literal)
        }
        else if let Some(mut symbol) = self.any_ident() {
            // generics are only strictly allowed to address basic types like Vec3<f32>, Vec4<u8>, etc. Any ::'s are removed.
            self.punct2(':',':');
            if self.punct('<') {
                let subtype = self.any_ident().unwrap();
                self.punct('>');
                symbol = format!("{}<{}>",symbol,subtype);
            }
            if let Some(mut parser) = self.group('{') {
                let mut fields: Vec<(String,sr::Expr)> = Vec::new();
                while !parser.done() {
                    let symbol = parser.any_ident().expect("identifier expected");
                    parser.punct(':');
                    let expr = parser.parse_expr();
                    fields.push((symbol,expr));
                    parser.punct(',');
                }
                sr::Expr::Struct(symbol,fields)
            }
            else if let Some(mut parser) = self.group('(') {
                let mut exprs: Vec<sr::Expr> = Vec::new();
                while !parser.done() {
                    exprs.push(parser.parse_expr());
                    parser.punct(',');
                }
                sr::Expr::Tuple(symbol,exprs)
            }
            else {
                sr::Expr::Symbol(symbol)
            }
        }
        else if let Some(mut parser) = self.group('[') {
            let expr = parser.parse_expr();
            if parser.punct(';') {
                let expr2 = parser.parse_expr();
                sr::Expr::AnonCloned(Box::new(expr),Box::new(expr2))
            }
            else {
                let mut exprs: Vec<sr::Expr> = Vec::new();
                exprs.push(expr);
                parser.punct(',');
                while !parser.done() {
                    exprs.push(parser.parse_expr());
                    parser.punct(',');
                }
                sr::Expr::AnonArray(exprs)
            }
        }
        else if let Some(mut parser) = self.group('(') {
            let mut exprs: Vec<sr::Expr> = Vec::new();
            while !parser.done() {
                exprs.push(parser.parse_expr());
                parser.punct(',');
            }
            sr::Expr::AnonTuple(exprs)
        }
        else {
            panic!("literal, symbol or [ expected, instead of: {:?}",self.current);
        }
    }

    pub fn parse_post_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_primary_expr();
        while
            self.peek_punct('.') ||
            self.peek_punct('?')  ||
            self.peek_group('(') ||
            self.peek_group('[') ||
            self.peek_ident("as") {
            if self.punct('.') {
                let ident = self.any_ident().expect("identifier expected");
                result = sr::Expr::Field(Box::new(result),ident);
            }
            else if self.punct('?') {
                result = sr::Expr::Error(Box::new(result));
            }
            else if let Some(mut parser) = self.group('(') {
                let mut exprs: Vec<sr::Expr> = Vec::new();
                while !parser.done() {
                    exprs.push(parser.parse_expr());
                    parser.punct(',');
                }
                result = sr::Expr::Call(Box::new(result),exprs);
            }
            else if let Some(mut parser) = self.group('[') {
                let expr = parser.parse_expr();
                result = sr::Expr::Index(Box::new(result),Box::new(expr));
            }
            else if self.ident("as") {
                let ty = self.parse_type();
                result = sr::Expr::Cast(Box::new(result),Box::new(ty));
            }
            else {
                panic!("., ?, (, [, or as expected");
            }
        }
        result
    }

    pub fn parse_neg_expr(&mut self) -> sr::Expr {
        if self.punct('-') {
            sr::Expr::Neg(Box::new(self.parse_neg_expr()))
        }
        else {
            self.parse_post_expr()
        }
    }

    pub fn parse_not_expr(&mut self) -> sr::Expr {
        if self.punct('!') {
            sr::Expr::Not(Box::new(self.parse_not_expr()))
        }
        else {
            self.parse_neg_expr()
        }
    }

    pub fn parse_mul_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_not_expr();
        while self.peek_punct('*') || self.peek_punct('/') || self.peek_punct('%') {
            if self.punct('*') {
                let expr = self.parse_not_expr();
                result = sr::Expr::Mul(Box::new(result),Box::new(expr));
            }
            else if self.punct('/') {
                let expr = self.parse_not_expr();
                result = sr::Expr::Div(Box::new(result),Box::new(expr));
            }
            else if self.punct('%') {
                let expr = self.parse_not_expr();
                result = sr::Expr::Mod(Box::new(result),Box::new(expr));
            }
            else {
                panic!("*, / or % expected");
            }
        }
        result
    }

    pub fn parse_add_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_mul_expr();
        while self.peek_punct('+') || self.peek_punct('-') {
            if self.punct('+') {
                let expr = self.parse_mul_expr();
                result = sr::Expr::Add(Box::new(result),Box::new(expr));
            }
            else if self.punct('-') {
                let expr = self.parse_mul_expr();
                result = sr::Expr::Sub(Box::new(result),Box::new(expr));
            }
            else {
                panic!("+ or - expected");
            }
        }
        result
    }

    pub fn parse_shift_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_add_expr();
        while self.peek_punct2('<','<') || self.peek_punct2('>','>') {
            if self.punct2('<','<') {
                let expr = self.parse_add_expr();
                result = sr::Expr::Shl(Box::new(result),Box::new(expr));
            }
            else if self.punct2('>','>') {
                let expr = self.parse_add_expr();
                result = sr::Expr::Shr(Box::new(result),Box::new(expr));
            }
            else {
                panic!("<< or >> expected");
            }
        }
        result
    }

    pub fn parse_and_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_shift_expr();
        while self.peek_punct('&') {
            self.consume();
            let expr = self.parse_shift_expr();
            result = sr::Expr::And(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_xor_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_and_expr();
        while self.peek_punct('^') {
            self.consume();
            let expr = self.parse_and_expr();
            result = sr::Expr::Xor(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_or_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_xor_expr();
        while self.peek_punct('^') {
            self.consume();
            let expr = self.parse_xor_expr();
            result = sr::Expr::Or(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_comp_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_or_expr();
        while
            self.peek_punct2('=','=') ||
            self.peek_punct2('!','=') ||
            self.peek_punct('>') ||
            self.peek_punct2('<','=') ||
            self.peek_punct('<') ||
            self.peek_punct2('>','=') {
            if self.punct2('=','=') {
                let expr = self.parse_or_expr();
                result = sr::Expr::Eq(Box::new(result),Box::new(expr));
            }
            else if self.punct2('!','=') {
                let expr = self.parse_or_expr();
                result = sr::Expr::NotEq(Box::new(result),Box::new(expr));
            }
            else if self.punct('>') {
                let expr = self.parse_or_expr();
                result = sr::Expr::Gt(Box::new(result),Box::new(expr));
            }
            else if self.punct2('<','=') {
                let expr = self.parse_or_expr();
                result = sr::Expr::NotGt(Box::new(result),Box::new(expr));
            }
            else if self.punct('<') {
                let expr = self.parse_or_expr();
                result = sr::Expr::Lt(Box::new(result),Box::new(expr));
            }
            else if self.punct2('>','=') {
                let expr = self.parse_or_expr();
                result = sr::Expr::NotLt(Box::new(result),Box::new(expr));
            }
            else {
                panic!("==, !=, <, >=, > or <= expected");
            }                
        }
        result
    }

    pub fn parse_logand_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_comp_expr();
        while self.peek_punct2('&','&') {
            self.consume();
            let expr = self.parse_comp_expr();
            result = sr::Expr::LogAnd(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_logor_expr(&mut self) -> sr::Expr {
        let mut result = self.parse_logand_expr();
        while self.peek_punct2('|','|') {
            self.consume();
            let expr = self.parse_logand_expr();
            result = sr::Expr::LogOr(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_assign_expr(&mut self) -> sr::Expr {
        let lvalue = self.parse_logor_expr();
        if self.punct('=') {
            let expr = self.parse_expr();
            sr::Expr::Assign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('+','=') {
            let expr = self.parse_expr();
            sr::Expr::AddAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('=','=') {
            let expr = self.parse_expr();
            sr::Expr::SubAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('*','=') {
            let expr = self.parse_expr();
            sr::Expr::MulAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('/','=') {
            let expr = self.parse_expr();
            sr::Expr::DivAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('%','=') {
            let expr = self.parse_expr();
            sr::Expr::ModAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('&','=') {
            let expr = self.parse_expr();
            sr::Expr::AndAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('^','=') {
            let expr = self.parse_expr();
            sr::Expr::XorAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('|','=') {
            let expr = self.parse_expr();
            sr::Expr::OrAssign(Box::new(lvalue),Box::new(expr))
        }
        else {
            lvalue
        }
    }

    pub fn parse_else_expr(&mut self) -> Option<sr::Expr> {
        if let Some(mut parser) = self.group('{') {
            let mut stats: Vec<sr::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            Some(sr::Expr::Block(stats))
        }
        else if self.ident("if") {
            if self.ident("let") {
                let mut pats: Vec<sr::Pat> = Vec::new();
                self.punct('|');
                pats.push(self.parse_pat());
                while self.punct('|') {
                    pats.push(self.parse_pat());
                }
                self.punct('=');
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<sr::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                let else_expr = if self.ident("else") {
                    if let Some(expr) = self.parse_else_expr() {
                        Some(Box::new(expr))
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                };
                Some(sr::Expr::IfLet(pats,Box::new(expr),stats,else_expr))
            }
            else {
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<sr::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                let else_expr = if self.ident("else") {
                    if let Some(expr) = self.parse_else_expr() {
                        Some(Box::new(expr))
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                };
                Some(sr::Expr::If(Box::new(expr),stats,else_expr))
            }
        }
        else {
            None
        }
    }

    pub fn parse_expr(&mut self) -> sr::Expr {
        if self.ident("continue") {
            sr::Expr::Continue
        }
        else if self.ident("break") {
            if self.peek_punct(';') || self.done() {
                sr::Expr::Break(None)
            }
            else {
                let expr = self.parse_expr();
                sr::Expr::Break(Some(Box::new(expr)))
            }
        }
        else if self.ident("return") {
            if self.peek_punct(';') || self.done() {
                sr::Expr::Return(None)
            }
            else {
                let expr = self.parse_expr();
                sr::Expr::Return(Some(Box::new(expr)))
            }
        }
        else if self.ident("loop") {
            let mut parser = self.group('{').expect("{ expected");
            let mut stats: Vec<sr::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            sr::Expr::Loop(stats)
        }
        else if self.ident("for") {
            let pat = self.parse_pat();
            self.ident("in");
            let expr = self.parse_expr();
            let mut parser = self.group('{').expect("{ expected");
            let mut stats: Vec<sr::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            sr::Expr::For(pat,Box::new(expr),stats)
        }
        else if self.ident("while") {
            if self.ident("let") {
                let mut pats: Vec<sr::Pat> = Vec::new();
                self.punct('|');
                pats.push(self.parse_pat());
                while self.punct('|') {
                    pats.push(self.parse_pat());
                }
                self.punct('=');
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<sr::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                sr::Expr::WhileLet(pats,Box::new(expr),stats)
            }
            else {
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<sr::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                sr::Expr::While(Box::new(expr),stats)
            }
        }
        else if self.ident("match") {
            let expr = self.parse_expr();
            let mut parser = self.group('{').expect("{ expected");
            let mut arms: Vec<(Vec<sr::Pat>,Option<Box<sr::Expr>>,Box<sr::Expr>)> = Vec::new();
            while !parser.done() {
                let mut pats: Vec<sr::Pat> = Vec::new();
                parser.punct('|');
                pats.push(parser.parse_pat());
                while parser.punct('|') {
                    pats.push(parser.parse_pat());
                }
                let if_expr = if parser.ident("if") {
                    Some(Box::new(parser.parse_expr()))
                }
                else {
                    None
                };
                parser.punct2('=','>');
                let expr = parser.parse_expr();
                parser.punct(',');
                arms.push((pats,if_expr,Box::new(expr)));
            }
            sr::Expr::Match(Box::new(expr),arms)
        }
        else {
            if let Some(expr) = self.parse_else_expr() {
                expr
            }
            else {
                self.parse_assign_expr()
            }
        }
    }

    // PATTERNS

    pub fn parse_pat(&mut self) -> sr::Pat {
        if self.punct('_') {
            sr::Pat::Wildcard
        }
        else if self.punct2('.','.') {
            sr::Pat::Rest
        }
        else if let Some(literal) = self.literal() {
            sr::Pat::Literal(literal.to_string())
        }
        else if let Some(mut parser) = self.group('[') {
            let mut pats: Vec<sr::Pat> = Vec::new();
            while !parser.done() {
                pats.push(parser.parse_pat());
                parser.punct(',');
            }
            sr::Pat::Slice(pats)
        }
        else if let Some(symbol) = self.any_ident() {
            sr::Pat::Symbol(symbol)
        }
        else {
            panic!("pattern not supported");
        }
    }

    // TYPES
    
    pub fn parse_type(&mut self) -> sr::Type {
        let mut result = if self.punct('_') {
            sr::Type::Inferred
        }
        else if let Some(symbol) = self.any_ident() {
            // generics are only strictly allowed to address basic types like Vec3<f32>, Vec4<u8>, etc. Any ::'s are removed.
            self.punct2(':',':');
            if self.punct('<') {
                let subtype = self.any_ident().unwrap();
                self.punct('>');
                sr::Type::Symbol(format!("{}<{}>",symbol,subtype))
            }
            else {
                sr::Type::Symbol(symbol)
            }
        }
        else if let Some(mut parser) = self.group('(') {
            let mut types: Vec<sr::Type> = Vec::new();
            while !parser.done() {
                types.push(parser.parse_type());
                parser.punct(',');
            }
            sr::Type::Tuple(types)
        }
        else {
            panic!("type not supported: {:?}",self.current);
        };
        while let Some(mut parser) = self.group('[') {
            let expr = parser.parse_expr();
            result = sr::Type::Array(Box::new(result),Box::new(expr));
        }
        result
    }
}
