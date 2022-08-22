use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_primary_expr(&mut self) -> ast::Expr {
        if let Some(literal) = self.literal() {
            ast::Expr::Literal(literal)
        }
        else if let Some(symbol) = self.any_ident() {
            ast::Expr::Symbol(symbol)
        }
        else if let Some(mut parser) = self.group('[') {
            let expr = parser.parse_expr();
            if parser.punct(';') {
                let expr2 = parser.parse_expr();
                ast::Expr::Cloned(Box::new(expr),Box::new(expr2))
            }
            else {
                let mut exprs: Vec<ast::Expr> = Vec::new();
                exprs.push(expr);
                parser.punct(',');
                while !parser.done() {
                    exprs.push(parser.parse_expr());
                    parser.punct(',');
                }
                ast::Expr::Array(exprs)
            }
        }
        else {
            panic!("literal, symbol or [ expected");
        }
    }

    pub fn parse_post_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_primary_expr();
        while
            self.peek_punct('.') ||
            self.peek_punct('?')  ||
            self.peek_group('(') ||
            self.peek_group('[') ||
            self.peek_ident("as") {
            if self.punct('.') {
                let ident = self.any_ident().expect("identifier expected");
                result = ast::Expr::Field(Box::new(result),ident);
            }
            else if self.punct('?') {
                result = ast::Expr::Error(Box::new(result));
            }
            else if let Some(mut parser) = self.group('(') {
                let mut exprs: Vec<ast::Expr> = Vec::new();
                while !parser.done() {
                    exprs.push(parser.parse_expr());
                    parser.punct(',');
                }
                result = ast::Expr::Call(Box::new(result),exprs);
            }
            else if let Some(mut parser) = self.group('[') {
                let expr = parser.parse_expr();
                result = ast::Expr::Index(Box::new(result),Box::new(expr));
            }
            else if self.ident("as") {
                let ty = self.parse_type();
                result = ast::Expr::Cast(Box::new(result),Box::new(ty));
            }
            else {
                panic!("., ?, (, [, or as expected");
            }
        }
        result
    }

    pub fn parse_neg_expr(&mut self) -> ast::Expr {
        if self.punct('-') {
            ast::Expr::Neg(Box::new(self.parse_neg_expr()))
        }
        else {
            self.parse_post_expr()
        }
    }

    pub fn parse_not_expr(&mut self) -> ast::Expr {
        if self.punct('!') {
            ast::Expr::Not(Box::new(self.parse_not_expr()))
        }
        else {
            self.parse_neg_expr()
        }
    }

    pub fn parse_mul_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_not_expr();
        while self.peek_punct('*') || self.peek_punct('/') || self.peek_punct('%') {
            if self.punct('*') {
                let expr = self.parse_not_expr();
                result = ast::Expr::Mul(Box::new(result),Box::new(expr));
            }
            else if self.punct('/') {
                let expr = self.parse_not_expr();
                result = ast::Expr::Div(Box::new(result),Box::new(expr));
            }
            else if self.punct('%') {
                let expr = self.parse_not_expr();
                result = ast::Expr::Mod(Box::new(result),Box::new(expr));
            }
            else {
                panic!("*, / or % expected");
            }
        }
        result
    }

    pub fn parse_add_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_mul_expr();
        while self.peek_punct('+') || self.peek_punct('-') {
            if self.punct('+') {
                let expr = self.parse_mul_expr();
                result = ast::Expr::Add(Box::new(result),Box::new(expr));
            }
            else if self.punct('-') {
                let expr = self.parse_mul_expr();
                result = ast::Expr::Sub(Box::new(result),Box::new(expr));
            }
            else {
                panic!("+ or - expected");
            }
        }
        result
    }

    pub fn parse_shift_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_add_expr();
        while self.peek_punct2('<','<') || self.peek_punct2('>','>') {
            if self.punct2('<','<') {
                let expr = self.parse_add_expr();
                result = ast::Expr::Shl(Box::new(result),Box::new(expr));
            }
            else if self.punct2('>','>') {
                let expr = self.parse_add_expr();
                result = ast::Expr::Shr(Box::new(result),Box::new(expr));
            }
            else {
                panic!("<< or >> expected");
            }
        }
        result
    }

    pub fn parse_and_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_shift_expr();
        while self.peek_punct('&') {
            self.consume();
            let expr = self.parse_shift_expr();
            result = ast::Expr::And(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_xor_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_and_expr();
        while self.peek_punct('^') {
            self.consume();
            let expr = self.parse_and_expr();
            result = ast::Expr::Xor(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_or_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_xor_expr();
        while self.peek_punct('^') {
            self.consume();
            let expr = self.parse_xor_expr();
            result = ast::Expr::Or(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_comp_expr(&mut self) -> ast::Expr {
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
                result = ast::Expr::Eq(Box::new(result),Box::new(expr));
            }
            else if self.punct2('!','=') {
                let expr = self.parse_or_expr();
                result = ast::Expr::NotEq(Box::new(result),Box::new(expr));
            }
            else if self.punct('>') {
                let expr = self.parse_or_expr();
                result = ast::Expr::Gt(Box::new(result),Box::new(expr));
            }
            else if self.punct2('<','=') {
                let expr = self.parse_or_expr();
                result = ast::Expr::NotGt(Box::new(result),Box::new(expr));
            }
            else if self.punct('<') {
                let expr = self.parse_or_expr();
                result = ast::Expr::Lt(Box::new(result),Box::new(expr));
            }
            else if self.punct2('>','=') {
                let expr = self.parse_or_expr();
                result = ast::Expr::NotLt(Box::new(result),Box::new(expr));
            }
            else {
                panic!("==, !=, <, >=, > or <= expected");
            }                
        }
        result
    }

    pub fn parse_logand_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_comp_expr();
        while self.peek_punct2('&','&') {
            self.consume();
            let expr = self.parse_comp_expr();
            result = ast::Expr::LogAnd(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_logor_expr(&mut self) -> ast::Expr {
        let mut result = self.parse_logand_expr();
        while self.peek_punct2('|','|') {
            self.consume();
            let expr = self.parse_logand_expr();
            result = ast::Expr::LogOr(Box::new(result),Box::new(expr));
        }
        result
    }

    pub fn parse_assign_expr(&mut self) -> ast::Expr {
        let lvalue = self.parse_logor_expr();
        if self.punct('=') {
            let expr = self.parse_expr();
            ast::Expr::Assign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('+','=') {
            let expr = self.parse_expr();
            ast::Expr::AddAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('=','=') {
            let expr = self.parse_expr();
            ast::Expr::SubAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('*','=') {
            let expr = self.parse_expr();
            ast::Expr::MulAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('/','=') {
            let expr = self.parse_expr();
            ast::Expr::DivAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('%','=') {
            let expr = self.parse_expr();
            ast::Expr::ModAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('&','=') {
            let expr = self.parse_expr();
            ast::Expr::AndAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('^','=') {
            let expr = self.parse_expr();
            ast::Expr::XorAssign(Box::new(lvalue),Box::new(expr))
        }
        else if self.punct2('|','=') {
            let expr = self.parse_expr();
            ast::Expr::OrAssign(Box::new(lvalue),Box::new(expr))
        }
        else {
            lvalue
        }
    }

    pub fn parse_else_expr(&mut self) -> ast::Expr {
        if let Some(mut parser) = self.group('{') {
            let mut stats: Vec<ast::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            ast::Expr::Block(stats)
        }
        else if self.ident("if") {
            if self.ident("let") {
                let mut pats: Vec<ast::Pat> = Vec::new();
                self.punct('|');
                pats.push(self.parse_pat());
                while self.punct('|') {
                    pats.push(self.parse_pat());
                }
                self.punct('=');
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<ast::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                let else_expr = if self.ident("else") {
                    Some(Box::new(self.parse_else_expr()))
                }
                else {
                    None
                };
                ast::Expr::IfLet(pats,Box::new(expr),stats,else_expr)
            }
            else {
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<ast::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                let else_expr = if self.ident("else") {
                    Some(Box::new(self.parse_else_expr()))
                }
                else {
                    None
                };
                ast::Expr::If(Box::new(expr),stats,else_expr)
            }
        }
        else {
            panic!("{ or if expected");
        }
    }

    pub fn parse_expr(&mut self) -> ast::Expr {
        if self.ident("continue") {
            ast::Expr::Continue
        }
        else if self.ident("break") {
            if self.peek_punct(';') || self.done() {
                ast::Expr::Break(None)
            }
            else {
                let expr = self.parse_expr();
                ast::Expr::Break(Some(Box::new(expr)))
            }
        }
        else if self.ident("return") {
            if self.peek_punct(';') || self.done() {
                ast::Expr::Return(None)
            }
            else {
                let expr = self.parse_expr();
                ast::Expr::Return(Some(Box::new(expr)))
            }
        }
        else if self.ident("loop") {
            let mut parser = self.group('{').expect("{ expected");
            let mut stats: Vec<ast::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            ast::Expr::Loop(stats)
        }
        else if self.ident("for") {
            let pat = self.parse_pat();
            self.ident("in");
            let expr = self.parse_expr();
            let mut parser = self.group('{').expect("{ expected");
            let mut stats: Vec<ast::Stat> = Vec::new();
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            ast::Expr::For(pat,Box::new(expr),stats)
        }
        else if self.ident("while") {
            if self.ident("let") {
                let mut pats: Vec<ast::Pat> = Vec::new();
                self.punct('|');
                pats.push(self.parse_pat());
                while self.punct('|') {
                    pats.push(self.parse_pat());
                }
                self.punct('=');
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<ast::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                ast::Expr::WhileLet(pats,Box::new(expr),stats)
            }
            else {
                let expr = self.parse_expr();
                let mut parser = self.group('{').expect("{ expected");
                let mut stats: Vec<ast::Stat> = Vec::new();
                while !parser.done() {
                    stats.push(parser.parse_stat());
                }
                ast::Expr::While(Box::new(expr),stats)
            }
        }
        else if self.ident("match") {
            let expr = self.parse_expr();
            let mut parser = self.group('{').expect("{ expected");
            let mut arms: Vec<(Vec<ast::Pat>,Option<Box<ast::Expr>>,Box<ast::Expr>)> = Vec::new();
            while !parser.done() {
                let mut pats: Vec<ast::Pat> = Vec::new();
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
            ast::Expr::Match(Box::new(expr),arms)
        }
        else {
            self.parse_else_expr()
        }
    }
}

impl Display for ast::Expr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            ast::Expr::Literal(literal) => write!(f,"{}",literal),
            ast::Expr::Symbol(symbol) => write!(f,"{}",symbol),
            ast::Expr::Array(exprs) => {
                write!(f,"[")?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr);
                    first_expr = false;
                }
                write!(f,"]")
            },
            ast::Expr::Cloned(expr,expr2) => write!(f,"[{}; {}]",expr,expr2),
            ast::Expr::Field(expr,field) => write!(f,"{}.{}",expr,field),
            ast::Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            ast::Expr::Call(expr,exprs) => {
                write!(f,"{}(",expr)?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr);
                    first_expr = false;
                }
                write!(f,")")
            },
            ast::Expr::Error(expr) => write!(f,"{}?",expr),
            ast::Expr::Cast(expr,ty) => write!(f,"{} as {}",expr,ty),
            ast::Expr::Neg(expr) => write!(f,"-{}",expr),
            ast::Expr::Not(expr) => write!(f,"!{}",expr),
            ast::Expr::Mul(expr,expr2) => write!(f,"{} * {}",expr,expr2),
            ast::Expr::Div(expr,expr2) => write!(f,"{} / {}",expr,expr2),
            ast::Expr::Mod(expr,expr2) => write!(f,"{} % {}",expr,expr2),
            ast::Expr::Add(expr,expr2) => write!(f,"{} + {}",expr,expr2),
            ast::Expr::Sub(expr,expr2) => write!(f,"{} - {}",expr,expr2),
            ast::Expr::Shl(expr,expr2) => write!(f,"{} << {}",expr,expr2),
            ast::Expr::Shr(expr,expr2) => write!(f,"{} >> {}",expr,expr2),
            ast::Expr::And(expr,expr2) => write!(f,"{} & {}",expr,expr2),
            ast::Expr::Xor(expr,expr2) => write!(f,"{} ^ {}",expr,expr2),
            ast::Expr::Or(expr,expr2) => write!(f,"{} | {}",expr,expr2),
            ast::Expr::Eq(expr,expr2) => write!(f,"{} == {}",expr,expr2),
            ast::Expr::NotEq(expr,expr2) => write!(f,"{} != {}",expr,expr2),
            ast::Expr::Gt(expr,expr2) => write!(f,"{} > {}",expr,expr2),
            ast::Expr::NotGt(expr,expr2) => write!(f,"{} <= {}",expr,expr2),
            ast::Expr::Lt(expr,expr2) => write!(f,"{} < {}",expr,expr2),
            ast::Expr::NotLt(expr,expr2) => write!(f,"{} >= {}",expr,expr2),
            ast::Expr::LogAnd(expr,expr2) => write!(f,"{} && {}",expr,expr2),
            ast::Expr::LogOr(expr,expr2) => write!(f,"{} || {}",expr,expr2),
            ast::Expr::Assign(expr,expr2) => write!(f,"{} = {}",expr,expr2),
            ast::Expr::AddAssign(expr,expr2) => write!(f,"{} += {}",expr,expr2),
            ast::Expr::SubAssign(expr,expr2) => write!(f,"{} -= {}",expr,expr2),
            ast::Expr::MulAssign(expr,expr2) => write!(f,"{} *= {}",expr,expr2),
            ast::Expr::DivAssign(expr,expr2) => write!(f,"{} /= {}",expr,expr2),
            ast::Expr::ModAssign(expr,expr2) => write!(f,"{} %= {}",expr,expr2),
            ast::Expr::AndAssign(expr,expr2) => write!(f,"{} &= {}",expr,expr2),
            ast::Expr::XorAssign(expr,expr2) => write!(f,"{} ^= {}",expr,expr2),
            ast::Expr::OrAssign(expr,expr2) => write!(f,"{} |= {}",expr,expr2),
            ast::Expr::Block(stats) => {
                write!(f,"{{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            ast::Expr::Continue => write!(f,"continue"),
            ast::Expr::Break(expr) => {
                write!(f,"break")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            ast::Expr::Return(expr) => {
                write!(f,"return")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            ast::Expr::Loop(stats) => {
                write!(f,"loop {{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            ast::Expr::For(pat,expr,stats) => {
                write!(f,"for {} in {} {{ ",pat,expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            ast::Expr::If(expr,stats,else_expr) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}");
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr);
                }
                write!(f,"")
            },
            ast::Expr::IfLet(pats,expr,stats,else_expr) => {
                write!(f,"if let ")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first_pat = false;
                }
                write!(f," = {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}");
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr);
                }
                write!(f,"")
            },
            ast::Expr::While(expr,stats) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            ast::Expr::WhileLet(pats,expr,stats) => {
                write!(f,"while let ")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first_pat = false;
                }
                write!(f," = {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            ast::Expr::Match(expr,arms) => {
                write!(f,"match {} {{ ",expr)?;
                for (pats,if_expr,expr) in arms {
                    let mut first_pat = true;
                    for pat in pats {
                        if !first_pat {
                            write!(f,"| ")?;
                        }
                        write!(f,"{} ",pat)?;
                        first_pat = false;
                    }
                    if let Some(if_expr) = if_expr {
                        write!(f,"if {} ",if_expr)?;
                    }
                    write!(f,"=> {},",expr)?;
                }
                write!(f,"}}")
            },
        }
    }
}
