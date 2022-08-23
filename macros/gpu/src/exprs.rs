use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
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
}

impl Display for sr::Expr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            sr::Expr::Literal(literal) => write!(f,"{}",literal),
            sr::Expr::Symbol(symbol) => write!(f,"{}",symbol),
            sr::Expr::AnonArray(exprs) => {
                write!(f,"[")?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,"]")
            },
            sr::Expr::AnonTuple(exprs) => {
                write!(f,"(")?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            sr::Expr::AnonCloned(expr,expr2) => write!(f,"[{}; {}]",expr,expr2),
            sr::Expr::Struct(symbol,fields) => {
                write!(f,"{} {{ ",symbol)?;
                let mut first_field = true;
                for (symbol,expr) in fields {
                    if !first_field {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",symbol,expr)?;
                    first_field = false;
                }
                write!(f," }}")
            },
            sr::Expr::Tuple(symbol,exprs) => {
                write!(f,"{}(",symbol)?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            sr::Expr::Field(expr,field) => write!(f,"{}.{}",expr,field),
            sr::Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            sr::Expr::Call(expr,exprs) => {
                write!(f,"{}(",expr)?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            sr::Expr::Error(expr) => write!(f,"{}?",expr),
            sr::Expr::Cast(expr,ty) => write!(f,"{} as {}",expr,ty),
            sr::Expr::Neg(expr) => write!(f,"-{}",expr),
            sr::Expr::Not(expr) => write!(f,"!{}",expr),
            sr::Expr::Mul(expr,expr2) => write!(f,"{} * {}",expr,expr2),
            sr::Expr::Div(expr,expr2) => write!(f,"{} / {}",expr,expr2),
            sr::Expr::Mod(expr,expr2) => write!(f,"{} % {}",expr,expr2),
            sr::Expr::Add(expr,expr2) => write!(f,"{} + {}",expr,expr2),
            sr::Expr::Sub(expr,expr2) => write!(f,"{} - {}",expr,expr2),
            sr::Expr::Shl(expr,expr2) => write!(f,"{} << {}",expr,expr2),
            sr::Expr::Shr(expr,expr2) => write!(f,"{} >> {}",expr,expr2),
            sr::Expr::And(expr,expr2) => write!(f,"{} & {}",expr,expr2),
            sr::Expr::Xor(expr,expr2) => write!(f,"{} ^ {}",expr,expr2),
            sr::Expr::Or(expr,expr2) => write!(f,"{} | {}",expr,expr2),
            sr::Expr::Eq(expr,expr2) => write!(f,"{} == {}",expr,expr2),
            sr::Expr::NotEq(expr,expr2) => write!(f,"{} != {}",expr,expr2),
            sr::Expr::Gt(expr,expr2) => write!(f,"{} > {}",expr,expr2),
            sr::Expr::NotGt(expr,expr2) => write!(f,"{} <= {}",expr,expr2),
            sr::Expr::Lt(expr,expr2) => write!(f,"{} < {}",expr,expr2),
            sr::Expr::NotLt(expr,expr2) => write!(f,"{} >= {}",expr,expr2),
            sr::Expr::LogAnd(expr,expr2) => write!(f,"{} && {}",expr,expr2),
            sr::Expr::LogOr(expr,expr2) => write!(f,"{} || {}",expr,expr2),
            sr::Expr::Assign(expr,expr2) => write!(f,"{} = {}",expr,expr2),
            sr::Expr::AddAssign(expr,expr2) => write!(f,"{} += {}",expr,expr2),
            sr::Expr::SubAssign(expr,expr2) => write!(f,"{} -= {}",expr,expr2),
            sr::Expr::MulAssign(expr,expr2) => write!(f,"{} *= {}",expr,expr2),
            sr::Expr::DivAssign(expr,expr2) => write!(f,"{} /= {}",expr,expr2),
            sr::Expr::ModAssign(expr,expr2) => write!(f,"{} %= {}",expr,expr2),
            sr::Expr::AndAssign(expr,expr2) => write!(f,"{} &= {}",expr,expr2),
            sr::Expr::XorAssign(expr,expr2) => write!(f,"{} ^= {}",expr,expr2),
            sr::Expr::OrAssign(expr,expr2) => write!(f,"{} |= {}",expr,expr2),
            sr::Expr::Block(stats) => {
                write!(f,"{{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            sr::Expr::Continue => write!(f,"continue"),
            sr::Expr::Break(expr) => {
                write!(f,"break")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            sr::Expr::Return(expr) => {
                write!(f,"return")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            sr::Expr::Loop(stats) => {
                write!(f,"loop {{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            sr::Expr::For(pat,expr,stats) => {
                write!(f,"for {} in {} {{ ",pat,expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            sr::Expr::If(expr,stats,else_expr) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            sr::Expr::IfLet(pats,expr,stats,else_expr) => {
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
                write!(f,"}}")?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            sr::Expr::While(expr,stats) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            sr::Expr::WhileLet(pats,expr,stats) => {
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
            sr::Expr::Match(expr,arms) => {
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
