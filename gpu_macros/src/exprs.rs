use {
    crate::*,
};

impl Parser {

    // Literal, Ident (Local, Param, Const), Struct, Tuple (Call), Variant, AnonTuple, Array, Cloned
    fn parse_primary_expr(&mut self) -> Expr {

        // Literal
        if let Some(literal) = self.literal() {
            Expr::Literal(literal)
        }

        // Local, Param, Const, Struct, Tuple, Variant, Call
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_exprs) = self.parse_brace_ident_exprs() {
                Expr::Struct(ident,ident_exprs)
            }

            // Tuple, Call
            else if let Some(exprs) = self.parse_paren_exprs() {
                Expr::Tuple(ident,exprs)
            }

            // Variant
            else if self.punct2(':',':') {
                let variant = self.ident().expect("identifier expected");
                if let Some(ident_exprs) = self.parse_brace_ident_exprs() {
                    Expr::Variant(ident,VariantExpr::Struct(variant,ident_exprs))
                }
                else if let Some(exprs) = self.parse_paren_exprs() {
                    Expr::Variant(ident,VariantExpr::Tuple(variant,exprs))
                }
                else {
                    Expr::Variant(ident,VariantExpr::Naked(variant))
                }
            }

            // Local, Param, Const
            else {
                Expr::Ident(ident)
            }
        }

        // Array, Cloned
        else if let Some(mut parser) = self.group('[') {
            let mut exprs: Vec<Expr> = Vec::new();
            while !parser.done() {
                let expr = parser.parse_expr();
                if parser.punct(';') {
                    let expr2 = parser.parse_expr();
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
        else if let Some(exprs) = self.parse_paren_exprs() {
            Expr::AnonTuple(exprs)
        }

        else {
            panic!("expression expected");
        }
    }

    // Field, TupleIndex, Index, Cast
    fn parse_postfix_expr(&mut self) -> Expr {
        let mut expr = self.parse_primary_expr();
        loop {

            // Field, Tuple
            if self.punct('.') {

                // Field
                if let Some(ident) = self.ident() {
                    expr = Expr::Field(Box::new(expr),ident);
                }

                // Tuple
                else if let Some(literal) = self.literal() {
                    if let Literal::Integer(i) = literal {
                        expr = Expr::TupleIndex(Box::new(expr),i);
                    }
                    else {
                        panic!("tuple index should be integer");
                    }
                }

                else {
                    panic!("field or tuple index expected");
                }
            }

            // Index
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.parse_expr()));
            }

            // Cast
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),self.parse_type());
            }

            else {
                break;
            }
        }
        expr
    }

    // Neg, Not
    fn parse_prefix_expr(&mut self) -> Expr {

        // Neg
        if self.punct('-') {
            Expr::Neg(Box::new(self.parse_prefix_expr()))
        }

        // Not
        else if self.punct('!') {
            Expr::Not(Box::new(self.parse_prefix_expr()))
        }

        else {
            self.parse_postfix_expr()
        }
    }

    // Mul, Div, Mod
    fn parse_mul_expr(&mut self) -> Expr {
        let mut expr = self.parse_prefix_expr();
        loop {

            // Mul
            if self.punct('*') {
                expr = Expr::Mul(Box::new(expr),Box::new(self.parse_prefix_expr()));
            }

            // Div
            else if self.punct('/') {
                expr = Expr::Div(Box::new(expr),Box::new(self.parse_prefix_expr()));
            }

            // Mod
            else if self.punct('%') {
                expr = Expr::Mod(Box::new(expr),Box::new(self.parse_prefix_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Add, Sub
    fn parse_add_expr(&mut self) -> Expr {
        let mut expr = self.parse_mul_expr();
        loop {

            // Add
            if self.punct('+') {
                expr = Expr::Add(Box::new(expr),Box::new(self.parse_mul_expr()));
            }

            // Sub
            else if self.punct('-') {
                expr = Expr::Sub(Box::new(expr),Box::new(self.parse_mul_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Shl, Shr
    fn parse_shift_expr(&mut self) -> Expr {
        let mut expr = self.parse_add_expr();
        loop {

            // Shl
            if self.punct2('<','<') {
                expr = Expr::Add(Box::new(expr),Box::new(self.parse_add_expr()));
            }

            // Shr
            else if self.punct2('>','>') {
                expr = Expr::Sub(Box::new(expr),Box::new(self.parse_add_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // And
    fn parse_and_expr(&mut self) -> Expr {
        let mut expr = self.parse_shift_expr();
        while self.punct('&') {
            expr = Expr::And(Box::new(expr),Box::new(self.parse_shift_expr()));
        }
        expr
    }

    // Or
    fn parse_or_expr(&mut self) -> Expr {
        let mut expr = self.parse_and_expr();
        while self.punct('|') {
            expr = Expr::Or(Box::new(expr),Box::new(self.parse_and_expr()));
        }
        expr
    }

    // Xor
    fn parse_xor_expr(&mut self) -> Expr {
        let mut expr = self.parse_or_expr();
        while self.punct('^') {
            expr = Expr::Xor(Box::new(expr),Box::new(self.parse_or_expr()));
        }
        expr
    }

    // Eq, NotEq, Less, Greater, LessEq, GreaterEq
    fn parse_comp_expr(&mut self) -> Expr {
        let mut expr = self.parse_xor_expr();
        loop {

            // Eq
            if self.punct2('=','=') {
                expr = Expr::Eq(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            // NotEq
            if self.punct2('!','=') {
                expr = Expr::NotEq(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            // Less
            if self.punct('<') {
                expr = Expr::Less(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            // Greater
            if self.punct('>') {
                expr = Expr::Greater(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            // LessEq
            if self.punct2('<','=') {
                expr = Expr::LessEq(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            // GreaterEq
            if self.punct2('>','=') {
                expr = Expr::GreaterEq(Box::new(expr),Box::new(self.parse_xor_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // LogAnd
    fn parse_logand_expr(&mut self) -> Expr {
        let mut expr = self.parse_comp_expr();
        while self.punct2('&','&') {
            expr = Expr::LogAnd(Box::new(expr),Box::new(self.parse_comp_expr()));
        }
        expr
    }

    // LogOr
    fn parse_logor_expr(&mut self) -> Expr {
        let mut expr = self.parse_logand_expr();
        while self.punct2('|','|') {
            expr = Expr::LogOr(Box::new(expr),Box::new(self.parse_logand_expr()));
        }
        expr
    }

    // Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign
    fn parse_assign_expr(&mut self) -> Expr {
        let expr = self.parse_logor_expr();
        if self.punct('=') {
            Expr::Assign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('+','=') {
            Expr::AddAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('-','=') {
            Expr::SubAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('*','=') {
            Expr::MulAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('/','=') {
            Expr::DivAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('%','=') {
            Expr::ModAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('&','=') {
            Expr::AndAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('|','=') {
            Expr::OrAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct2('^','=') {
            Expr::XorAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct3('<','<','=') {
            Expr::ShlAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else if self.punct3('>','>','=') {
            Expr::ShrAssign(Box::new(expr),Box::new(self.parse_logor_expr()))
        }
        else {
            expr
        }
    }

    // Block
    pub fn parse_block(&mut self) -> Option<Block> {
        let mut last_expr: Option<Box<Expr>> = None;
        if let Some(mut parser) = self.group('{') {
            let mut stats: Vec<Stat> = Vec::new();
            while !parser.done() {

                // Let
                if parser.keyword("let") {
                    let pat = parser.parse_pat();
                    let ty = if parser.punct(':') {
                        Some(parser.parse_type())
                    }
                    else {
                        None
                    };
                    parser.punct('=');
                    let expr = parser.parse_expr();
                    parser.punct(';');
                    stats.push(Stat::Let(pat,ty,Box::new(expr)));
                }

                // Expr
                else {
                    let expr = parser.parse_expr();
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
    fn parse_else_expr(&mut self) -> Option<Expr> {

        // Block
        if let Some(block) = self.parse_block() {
            Some(Expr::Block(block))
        }

        // If, IfLet
        else if self.keyword("if") {

            // IfLet
            if self.keyword("let") {
                let pats = self.parse_or_pats();
                self.punct('=');
                let expr = self.parse_expr();
                let block = self.parse_block().expect("{ expected");
                if self.keyword("else") {
                    let else_expr = self.parse_else_expr().expect("if, if let, or block expected");
                    Some(Expr::IfLet(pats,Box::new(expr),block,Some(Box::new(else_expr))))
                }
                else {
                    Some(Expr::IfLet(pats,Box::new(expr),block,None))
                }
            }

            // If
            else {
                let expr = self.parse_expr();
                let block = self.parse_block().expect("{ expected");
                if self.keyword("else") {
                    let else_expr = self.parse_else_expr().expect("if, if let, or block expected");
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
    pub fn parse_expr(&mut self) -> Expr {

        // Continue
        if self.keyword("continue") {
            Expr::Continue
        }

        // Break
        else if self.keyword("break") {
            if !self.peek_punct(';') {
                let expr = self.parse_expr();
                Expr::Break(Some(Box::new(expr)))
            }
            else {
                Expr::Break(None)
            }
        }

        // Return
        else if self.keyword("return") {
            if !self.peek_punct(';') {
                let expr = self.parse_expr();
                Expr::Return(Some(Box::new(expr)))
            }
            else {
                Expr::Return(None)
            }
        }

        // If, IfLet, Block
        else if let Some(expr) = self.parse_else_expr() {
            expr
        }

        // While, WhileLet
        else if self.keyword("while") {

            // WhileLet
            if self.keyword("let") {
                let pats = self.parse_or_pats();
                self.punct('=');
                let expr = self.parse_expr();
                let block = self.parse_block().expect("{ expected");
                Expr::WhileLet(pats,Box::new(expr),block)
            }

            // While
            else {
                let expr = self.parse_expr();
                let block = self.parse_block().expect("{ expected");
                Expr::While(Box::new(expr),block)
            }
        }

        // Loop
        else if self.keyword("loop") {
            Expr::Loop(self.parse_block().expect("{ expected"))
        }

        // For
        else if self.keyword("for") {
            let pats = self.parse_or_pats();
            self.keyword("in");
            let range = if self.punct2('.','.') {
                if self.peek_group('{') {
                    Range::All
                }
                else {
                    if self.punct('=') {
                        Range::ToIncl(Box::new(self.parse_expr()))
                    }
                    else {
                        Range::To(Box::new(self.parse_expr()))
                    }
                }
            }
            else {
                let expr = self.parse_expr();
                if self.punct2('.','.') {
                    if self.peek_group('{') {
                        Range::From(Box::new(expr))
                    }
                    else {
                        if self.punct('=') {
                            Range::FromToIncl(Box::new(expr),Box::new(self.parse_expr()))
                        }
                        else {
                            Range::FromTo(Box::new(expr),Box::new(self.parse_expr()))
                        }
                    }
                }
                else {
                    Range::Only(Box::new(expr))
                }
            };
            let block = self.parse_block().expect("block expected");
            Expr::For(pats,range,block)
        }

        // Match
        else if self.keyword("match") {
            let expr = self.parse_expr();
            let mut arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
            if let Some(mut parser) = self.group('{') {
                while !parser.done() {
                    let pats = parser.parse_or_pats();
                    let if_expr = if parser.keyword("if") {
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
            }
            else {
                panic!("{{ expected");
            }
            Expr::Match(Box::new(expr),arms)
        }

        // *Assign
        else {
            self.parse_assign_expr()
        }
    }
}
