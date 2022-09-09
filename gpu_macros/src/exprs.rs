use crate::*;

impl Parser {

    // Literal, Ident (Local, Param, Const), Struct, Tuple (Call), Variant, Base, AnonTuple, Array, Cloned
    fn primary_expr(&mut self) -> Expr {

        // Literal
        if let Some(value) = self.boolean_literal() {
            Expr::Boolean(value)
        }
        else if let Some(value) = self.integer_literal() {
            Expr::Integer(value as i64)
        }
        else if let Some(value) = self.float_literal() {
            Expr::Float(value)
        }

        // Local, Param, Const, Struct, Tuple, Variant, Base, Call
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_exprs) = self.brace_ident_exprs() {
                Expr::UnknownStruct(ident,ident_exprs)
            }

            // Call, Tuple
            else if let Some(exprs) = self.paren_exprs() {
                Expr::UnknownCall(ident,exprs)
            }

            // Base
            else if self.punct('<') {
                // Vec2<T>, etc. literals
                let element_type = self.ident().expect("identifier expected");
                self.punct('>');
                let ident = format!("{}<{}>",ident,element_type);
                let base_type = sr::BaseType::from_rust(&ident).expect("base type expected");
                let ident_exprs = self.brace_ident_exprs().expect("{ expected");
                Expr::Base(base_type,ident_exprs)
            }

            // Variant, Base
            else if self.punct2(':',':') {
                if self.punct('<') {
                    // Vec2::<T>, etc. literals
                    let element_type = self.ident().expect("identifier expected");
                    self.punct('>');
                    let ident = format!("{}<{}>",ident,element_type);
                    let base_type = sr::BaseType::from_rust(&ident).expect("base type expected");
                    let ident_exprs = self.brace_ident_exprs().expect("{ expected");
                    Expr::Base(base_type,ident_exprs)
                }
                else {
                    let variant = self.ident().expect("identifier expected");
                    if let Some(ident_exprs) = self.brace_ident_exprs() {
                        Expr::UnknownVariant(ident,VariantExpr::Struct(variant,ident_exprs))
                    }
                    else if let Some(exprs) = self.paren_exprs() {
                        Expr::UnknownVariant(ident,VariantExpr::Tuple(variant,exprs))
                    }
                    else {
                        Expr::UnknownVariant(ident,VariantExpr::Naked(variant))
                    }
                }
            }

            // Local, Param, Const
            else {
                Expr::UnknownIdent(ident)
            }
        }

        // Array, Cloned
        else if let Some(mut parser) = self.group('[') {
            let mut exprs: Vec<Expr> = Vec::new();
            while !parser.done() {
                let expr = parser.expr();
                if parser.punct(';') {
                    let expr2 = parser.expr();
                    return Expr::Cloned(Box::new(expr),Box::new(expr2));
                }
                else {
                    exprs.push(expr);
                }
                parser.punct(',');
            }
            Expr::Array(exprs)
        }

        // AnonTuple
        else if let Some(exprs) = self.paren_exprs() {
            Expr::AnonTuple(exprs)
        }

        else {
            panic!("expression expected");
        }
    }

    // Field, TupleIndex, Index, Cast
    fn postfix_expr(&mut self) -> Expr {
        let mut expr = self.primary_expr();
        loop {

            // Field, Tuple
            if self.punct('.') {

                // Field
                if let Some(ident) = self.ident() {
                    expr = Expr::Field(Box::new(expr),ident);
                }

                // Tuple
                else if let Some(value) = self.integer_literal() {
                    expr = Expr::Field(Box::new(expr),format!("_{}",value));
                }

                else {
                    panic!("field or tuple index expected");
                }
            }

            // Index
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.expr()));
            }

            // Cast
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),self.type_());
            }

            else {
                break;
            }
        }
        expr
    }

    // Neg, Not
    fn prefix_expr(&mut self) -> Expr {

        // Neg
        if self.punct('-') {
            Expr::Neg(Box::new(self.prefix_expr()))
        }

        // Not
        else if self.punct('!') {
            Expr::Not(Box::new(self.prefix_expr()))
        }

        else {
            self.postfix_expr()
        }
    }

    // Mul, Div, Mod
    fn mul_expr(&mut self) -> Expr {
        let mut expr = self.prefix_expr();
        loop {

            // Mul
            if self.punct('*') {
                expr = Expr::Mul(Box::new(expr),Box::new(self.prefix_expr()));
            }

            // Div
            else if self.punct('/') {
                expr = Expr::Div(Box::new(expr),Box::new(self.prefix_expr()));
            }

            // Mod
            else if self.punct('%') {
                expr = Expr::Mod(Box::new(expr),Box::new(self.prefix_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Add, Sub
    fn add_expr(&mut self) -> Expr {
        let mut expr = self.mul_expr();
        loop {

            // Add
            if self.punct('+') {
                expr = Expr::Add(Box::new(expr),Box::new(self.mul_expr()));
            }

            // Sub
            else if self.punct('-') {
                expr = Expr::Sub(Box::new(expr),Box::new(self.mul_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Shl, Shr
    fn shift_expr(&mut self) -> Expr {
        let mut expr = self.add_expr();
        loop {

            // Shl
            if self.punct2('<','<') {
                expr = Expr::Shl(Box::new(expr),Box::new(self.add_expr()));
            }

            // Shr
            else if self.punct2('>','>') {
                expr = Expr::Shr(Box::new(expr),Box::new(self.add_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // And
    fn and_expr(&mut self) -> Expr {
        let mut expr = self.shift_expr();
        while self.punct('&') {
            expr = Expr::And(Box::new(expr),Box::new(self.shift_expr()));
        }
        expr
    }

    // Or
    fn or_expr(&mut self) -> Expr {
        let mut expr = self.and_expr();
        while self.punct('|') {
            expr = Expr::Or(Box::new(expr),Box::new(self.and_expr()));
        }
        expr
    }

    // Xor
    fn xor_expr(&mut self) -> Expr {
        let mut expr = self.or_expr();
        while self.punct('^') {
            expr = Expr::Xor(Box::new(expr),Box::new(self.or_expr()));
        }
        expr
    }

    // Eq, NotEq, Less, Greater, LessEq, GreaterEq
    fn comp_expr(&mut self) -> Expr {
        let mut expr = self.xor_expr();
        loop {

            // Eq
            if self.punct2('=','=') {
                expr = Expr::Eq(Box::new(expr),Box::new(self.xor_expr()));
            }

            // NotEq
            if self.punct2('!','=') {
                expr = Expr::NotEq(Box::new(expr),Box::new(self.xor_expr()));
            }

            // Less
            if self.punct('<') {
                expr = Expr::Less(Box::new(expr),Box::new(self.xor_expr()));
            }

            // Greater
            if self.punct('>') {
                expr = Expr::Greater(Box::new(expr),Box::new(self.xor_expr()));
            }

            // LessEq
            if self.punct2('<','=') {
                expr = Expr::LessEq(Box::new(expr),Box::new(self.xor_expr()));
            }

            // GreaterEq
            if self.punct2('>','=') {
                expr = Expr::GreaterEq(Box::new(expr),Box::new(self.xor_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // LogAnd
    fn logand_expr(&mut self) -> Expr {
        let mut expr = self.comp_expr();
        while self.punct2('&','&') {
            expr = Expr::LogAnd(Box::new(expr),Box::new(self.comp_expr()));
        }
        expr
    }

    // LogOr
    fn logor_expr(&mut self) -> Expr {
        let mut expr = self.logand_expr();
        while self.punct2('|','|') {
            expr = Expr::LogOr(Box::new(expr),Box::new(self.logand_expr()));
        }
        expr
    }

    // Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign
    fn assign_expr(&mut self) -> Expr {
        let expr = self.logor_expr();
        if self.punct('=') {
            Expr::Assign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('+','=') {
            Expr::AddAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('-','=') {
            Expr::SubAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('*','=') {
            Expr::MulAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('/','=') {
            Expr::DivAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('%','=') {
            Expr::ModAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('&','=') {
            Expr::AndAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('|','=') {
            Expr::OrAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct2('^','=') {
            Expr::XorAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct3('<','<','=') {
            Expr::ShlAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else if self.punct3('>','>','=') {
            Expr::ShrAssign(Box::new(expr),Box::new(self.logor_expr()))
        }
        else {
            expr
        }
    }

    // Block
    pub fn block(&mut self) -> Option<Block> {
        let mut last_expr: Option<Box<Expr>> = None;
        if let Some(mut parser) = self.group('{') {
            let mut stats: Vec<Stat> = Vec::new();
            while !parser.done() {

                // Let
                if parser.keyword("let") {
                    let pat = parser.pat();
                    let type_ = if parser.punct(':') {
                        parser.type_()
                    }
                    else {
                        Type::Inferred
                    };
                    parser.punct('=');
                    let expr = parser.expr();
                    parser.punct(';');
                    stats.push(Stat::Let(Box::new(pat),Box::new(type_),Box::new(expr)));
                }

                // Expr
                else {
                    let expr = parser.expr();
                    if parser.punct(';') {
                        stats.push(Stat::Expr(Box::new(expr)));
                    }
                    else {
                        // assuming that no ; only happens at the end of a block...
                        last_expr = Some(Box::new(expr));
                    }
                }
            }
            Some(Block { stats,expr: last_expr, })
        }
        else {
            None
        }
    }

    // If, IfLet, Block
    fn else_expr(&mut self) -> Option<Expr> {

        // Block
        if let Some(block) = self.block() {
            Some(Expr::Block(block))
        }

        // If, IfLet
        else if self.keyword("if") {

            // IfLet
            if self.keyword("let") {
                let pats = self.or_pats();
                self.punct('=');
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                if self.keyword("else") {
                    let else_expr = self.else_expr().expect("if, if let, or block expected");
                    Some(Expr::IfLet(pats,Box::new(expr),block,Some(Box::new(else_expr))))
                }
                else {
                    Some(Expr::IfLet(pats,Box::new(expr),block,None))
                }
            }

            // If
            else {
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                if self.keyword("else") {
                    let else_expr = self.else_expr().expect("if, if let, or block expected");
                    Some(Expr::If(Box::new(expr),block,Some(Box::new(else_expr))))
                }
                else {
                    Some(Expr::If(Box::new(expr),block,None))
                }
            }
        }

        else {
            None
        }
    }

    // Continue, Break, Return, Block, If, IfLet, Loop, For, While, WhileLet, Match, *Assign
    pub fn expr(&mut self) -> Expr {

        // Continue
        if self.keyword("continue") {
            Expr::Continue
        }

        // Break
        else if self.keyword("break") {
            if !self.peek_punct(';') {
                let expr = self.expr();
                Expr::Break(Some(Box::new(expr)))
            }
            else {
                Expr::Break(None)
            }
        }

        // Return
        else if self.keyword("return") {
            if !self.peek_punct(';') {
                let expr = self.expr();
                Expr::Return(Some(Box::new(expr)))
            }
            else {
                Expr::Return(None)
            }
        }

        // If, IfLet, Block
        else if let Some(expr) = self.else_expr() {
            expr
        }

        // While, WhileLet
        else if self.keyword("while") {

            // WhileLet
            if self.keyword("let") {
                let pats = self.or_pats();
                self.punct('=');
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                Expr::WhileLet(pats,Box::new(expr),block)
            }

            // While
            else {
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                Expr::While(Box::new(expr),block)
            }
        }

        // Loop
        else if self.keyword("loop") {
            Expr::Loop(self.block().expect("{ expected"))
        }

        // For
        else if self.keyword("for") {
            let pats = self.or_pats();
            self.keyword("in");
            let range = if self.punct2('.','.') {
                if self.peek_group('{') {
                    Range::All
                }
                else {
                    if self.punct('=') {
                        Range::ToIncl(Box::new(self.expr()))
                    }
                    else {
                        Range::To(Box::new(self.expr()))
                    }
                }
            }
            else {
                let expr = self.expr();
                if self.punct2('.','.') {
                    if self.peek_group('{') {
                        Range::From(Box::new(expr))
                    }
                    else {
                        if self.punct('=') {
                            Range::FromToIncl(Box::new(expr),Box::new(self.expr()))
                        }
                        else {
                            Range::FromTo(Box::new(expr),Box::new(self.expr()))
                        }
                    }
                }
                else {
                    Range::Only(Box::new(expr))
                }
            };
            let block = self.block().expect("block expected");
            Expr::For(pats,range,block)
        }

        // Match
        else if self.keyword("match") {
            let expr = self.expr();
            let mut arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
            if let Some(mut parser) = self.group('{') {
                while !parser.done() {
                    let pats = parser.or_pats();
                    let if_expr = if parser.keyword("if") {
                        Some(Box::new(parser.expr()))
                    }
                    else {
                        None
                    };
                    parser.punct2('=','>');
                    let expr = parser.expr();
                    parser.punct(',');
                    arms.push((pats,if_expr,Box::new(expr)));
                }
            }
            else {
                panic!("{}","{ expected");
            }
            Expr::Match(Box::new(expr),arms)
        }

        // *Assign
        else {
            self.assign_expr()
        }
    }
}
