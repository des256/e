use {
    crate::*,
    sr::*,
};

impl Parser {

    // Boolean, Integer, Float, UnknownIdent, UnknownStruct, UnknownTupleOrCall, UnknownVariant, AnonTuple, Array, Cloned
    pub(crate) fn primary_expr(&mut self) -> ast::Expr {

        // Boolean
        if let Some(value) = self.boolean_literal() {
            ast::Expr::Boolean(value)
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            ast::Expr::Integer(value as i64)
        }

        // Float
        else if let Some(value) = self.float_literal() {
            ast::Expr::Float(value)
        }

        // UnknownIdent, UnknownStruct, UnknownTupleOrCall, UnknownVariant
        else if let Some(ident) = self.ident() {

            // UnknownStruct
            if let Some(ident_exprs) = self.brace_ident_exprs() {
                ast::Expr::UnknownStruct(ident,ident_exprs)
            }

            // UnknownTupleOrCall
            else if let Some(exprs) = self.paren_exprs() {
                ast::Expr::UnknownTupleOrCall(ident,exprs)
            }

            // UnknownVariant
            else if self.punct2(':',':') {
                if self.punct('<') {
                    // process double :: style (Vec2::<T>) as UnknownStruct
                    let element_type = self.ident().expect("identifier expected");
                    self.punct('>');
                    let ident = format!("{}<{}>",ident,element_type);
                    if let Some(ident_exprs) = self.brace_ident_exprs() {
                        ast::Expr::UnknownStruct(ident,ident_exprs)
                    }
                    else {
                        self.fatal(&format!("struct literal expected after {}",ident));
                    }
                }
                else {
                    let variant = self.ident().expect("identifier expected");
                    if let Some(ident_exprs) = self.brace_ident_exprs() {
                        ast::Expr::UnknownVariant(ident,ast::UnknownExprVariant::Struct(variant,ident_exprs))
                    }
                    else if let Some(exprs) = self.paren_exprs() {
                        ast::Expr::UnknownVariant(ident,ast::UnknownExprVariant::Tuple(variant,exprs))
                    }
                    else {
                        ast::Expr::UnknownVariant(ident,ast::UnknownExprVariant::Naked(variant))
                    }
                }
            }

            // UnknownIdent
            else {
                ast::Expr::UnknownIdent(ident)
            }
        }

        // AnonTuple
        else if let Some(exprs) = self.paren_exprs() {
            ast::Expr::AnonTuple(exprs)
        }

        // Array, Cloned
        else if let Some(mut parser) = self.group('[') {
            let mut exprs: Vec<ast::Expr> = Vec::new();
            while !parser.done() {
                let expr = parser.expr();
                if parser.punct(';') {
                    let expr2 = parser.expr();
                    return ast::Expr::Cloned(Box::new(expr),Box::new(expr2));
                }
                else {
                    exprs.push(expr);
                }
                parser.punct(',');
            }
            ast::Expr::Array(exprs)
        }

        else {
            self.fatal("expression expected");
        }
    }

    // UnknownField, UnknownMethod, UnknownTupleIndex, Index, Cast
    pub(crate) fn postfix_expr(&mut self) -> ast::Expr {
        let mut expr = self.primary_expr();
        loop {

            // UnknownField, UnknownMethod, UnknownTupleIndex
            if self.punct('.') {

                // UnknownField, UnknownMethod
                if let Some(ident) = self.ident() {
                    if let Some(exprs) = self.paren_exprs() {
                        expr = ast::Expr::UnknownMethod(Box::new(expr),ident,exprs);
                    }
                    else {
                        expr = ast::Expr::UnknownField(Box::new(expr),ident);
                    }
                }

                // TupleIndex
                else if let Some(value) = self.integer_literal() {
                    expr = ast::Expr::UnknownTupleIndex(Box::new(expr),value as usize);
                }

                else {
                    self.fatal("field or tuple index expected");
                }
            }

            // Index
            else if let Some(mut parser) = self.group('[') {
                expr = ast::Expr::Index(Box::new(expr),Box::new(parser.expr()));
            }

            // Cast
            else if self.keyword("as") {
                expr = ast::Expr::Cast(Box::new(expr),Box::new(self.type_()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Neg, Not
    pub(crate) fn prefix_expr(&mut self) -> ast::Expr {

        // Neg
        if self.punct('-') {
            ast::Expr::Unary(ast::UnaryOp::Neg,Box::new(self.prefix_expr()))
        }

        // Not
        else if self.punct('!') {
            ast::Expr::Unary(ast::UnaryOp::Not,Box::new(self.prefix_expr()))
        }

        else {
            self.postfix_expr()
        }
    }

    // Mul, Div, Mod
    pub(crate) fn mul_expr(&mut self) -> ast::Expr {
        let mut expr = self.prefix_expr();
        loop {

            // Mul
            if self.punct('*') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Mul,Box::new(self.prefix_expr()));
            }

            // Div
            else if self.punct('/') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Div,Box::new(self.prefix_expr()));
            }

            // Mod
            else if self.punct('%') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Mod,Box::new(self.prefix_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Add, Sub
    pub(crate) fn add_expr(&mut self) -> ast::Expr {
        let mut expr = self.mul_expr();
        loop {

            // Add
            if self.punct('+') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Add,Box::new(self.mul_expr()));
            }

            // Sub
            else if self.punct('-') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Sub,Box::new(self.mul_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Shl, Shr
    pub(crate) fn shift_expr(&mut self) -> ast::Expr {
        let mut expr = self.add_expr();
        loop {

            // Shl
            if self.punct2('<','<') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Shl,Box::new(self.add_expr()));
            }

            // Shr
            else if self.punct2('>','>') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Shr,Box::new(self.add_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // And
    pub(crate) fn and_expr(&mut self) -> ast::Expr {
        let mut expr = self.shift_expr();
        while self.punct('&') {
            expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::And,Box::new(self.shift_expr()));
        }
        expr
    }

    // Or
    pub(crate) fn or_expr(&mut self) -> ast::Expr {
        let mut expr = self.and_expr();
        while self.punct('|') {
            expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Or,Box::new(self.and_expr()));
        }
        expr
    }

    // Xor
    pub(crate) fn xor_expr(&mut self) -> ast::Expr {
        let mut expr = self.or_expr();
        while self.punct('^') {
            expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Xor,Box::new(self.or_expr()));
        }
        expr
    }

    // Eq, NotEq, Less, Greater, LessEq, GreaterEq
    pub(crate) fn comp_expr(&mut self) -> ast::Expr {
        let mut expr = self.xor_expr();
        loop {

            // Eq
            if self.punct2('=','=') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Eq,Box::new(self.xor_expr()));
            }

            // NotEq
            if self.punct2('!','=') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::NotEq,Box::new(self.xor_expr()));
            }

            // Less
            if self.punct('<') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Less,Box::new(self.xor_expr()));
            }

            // Greater
            if self.punct('>') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Greater,Box::new(self.xor_expr()));
            }

            // LessEq
            if self.punct2('<','=') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::LessEq,Box::new(self.xor_expr()));
            }

            // GreaterEq
            if self.punct2('>','=') {
                expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::GreaterEq,Box::new(self.xor_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // LogAnd
    pub(crate) fn logand_expr(&mut self) -> ast::Expr {
        let mut expr = self.comp_expr();
        while self.punct2('&','&') {
            expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::LogAnd,Box::new(self.comp_expr()));
        }
        expr
    }

    // LogOr
    pub(crate) fn logor_expr(&mut self) -> ast::Expr {
        let mut expr = self.logand_expr();
        while self.punct2('|','|') {
            expr = ast::Expr::Binary(Box::new(expr),ast::BinaryOp::LogOr,Box::new(self.logand_expr()));
        }
        expr
    }

    // Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign
    pub(crate) fn assign_expr(&mut self) -> ast::Expr {
        let expr = self.logor_expr();
        if self.punct('=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::Assign,Box::new(self.logor_expr()))
        }
        else if self.punct2('+','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::AddAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('-','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::SubAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('*','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::MulAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('/','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::DivAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('%','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::ModAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('&','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::AndAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('|','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::OrAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('^','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::XorAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('<','<','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::ShlAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('>','>','=') {
            ast::Expr::Binary(Box::new(expr),ast::BinaryOp::ShrAssign,Box::new(self.logor_expr()))
        }
        else {
            expr
        }
    }

    // Block
    pub(crate) fn block(&mut self) -> Option<ast::Block> {
        let mut last_expr: Option<Box<ast::Expr>> = None;
        if let Some(mut parser) = self.group('{') {
            let mut stats: Vec<ast::Stat> = Vec::new();
            while !parser.done() {

                // Let
                if parser.keyword("let") {
                    let pat = parser.pat();
                    let type_ = if parser.punct(':') {
                        parser.type_()
                    }
                    else {
                        ast::Type::Inferred
                    };
                    parser.punct('=');
                    let expr = parser.expr();
                    parser.punct(';');
                    stats.push(ast::Stat::Let(Box::new(pat),Box::new(type_),Box::new(expr)));
                }

                // ast::Expr
                else {
                    let expr = parser.expr();
                    if parser.punct(';') {
                        stats.push(ast::Stat::Expr(Box::new(expr)));
                    }
                    else {
                        // assuming that no ; only happens at the end of a block...
                        last_expr = Some(Box::new(expr));
                    }
                }
            }
            Some(ast::Block { stats,expr: last_expr, })
        }
        else {
            None
        }
    }

    // If, IfLet, Block
    pub(crate) fn else_expr(&mut self) -> Option<ast::Expr> {

        // Block
        if let Some(block) = self.block() {
            Some(ast::Expr::Block(block))
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
                    Some(ast::Expr::IfLet(pats,Box::new(expr),block,Some(Box::new(else_expr))))
                }
                else {
                    Some(ast::Expr::IfLet(pats,Box::new(expr),block,None))
                }
            }

            // If
            else {
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                if self.keyword("else") {
                    let else_expr = self.else_expr().expect("if, if let, or block expected");
                    Some(ast::Expr::If(Box::new(expr),block,Some(Box::new(else_expr))))
                }
                else {
                    Some(ast::Expr::If(Box::new(expr),block,None))
                }
            }
        }

        else {
            None
        }
    }

    // Continue, Break, Return, Block, If, IfLet, Loop, For, While, WhileLet, Match, *Assign
    pub(crate) fn expr(&mut self) -> ast::Expr {

        // Continue
        if self.keyword("continue") {
            ast::Expr::Continue
        }

        // Break
        else if self.keyword("break") {
            if !self.peek_punct(';') {
                let expr = self.expr();
                ast::Expr::Break(Some(Box::new(expr)))
            }
            else {
                ast::Expr::Break(None)
            }
        }

        // Return
        else if self.keyword("return") {
            if !self.peek_punct(';') {
                let expr = self.expr();
                ast::Expr::Return(Some(Box::new(expr)))
            }
            else {
                ast::Expr::Return(None)
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
                ast::Expr::WhileLet(pats,Box::new(expr),block)
            }

            // While
            else {
                let expr = self.expr();
                let block = self.block().expect("{ expected");
                ast::Expr::While(Box::new(expr),block)
            }
        }

        // Loop
        else if self.keyword("loop") {
            ast::Expr::Loop(self.block().expect("{ expected"))
        }

        // For
        else if self.keyword("for") {
            let pats = self.or_pats();
            self.keyword("in");
            let range = if self.punct2('.','.') {
                if self.peek_group('{') {
                    ast::Range::All
                }
                else {
                    if self.punct('=') {
                        ast::Range::ToIncl(Box::new(self.expr()))
                    }
                    else {
                        ast::Range::To(Box::new(self.expr()))
                    }
                }
            }
            else {
                let expr = self.expr();
                if self.punct2('.','.') {
                    if self.peek_group('{') {
                        ast::Range::From(Box::new(expr))
                    }
                    else {
                        if self.punct('=') {
                            ast::Range::FromToIncl(Box::new(expr),Box::new(self.expr()))
                        }
                        else {
                            ast::Range::FromTo(Box::new(expr),Box::new(self.expr()))
                        }
                    }
                }
                else {
                    ast::Range::Only(Box::new(expr))
                }
            };
            let block = self.block().expect("block expected");
            ast::Expr::For(pats,range,block)
        }

        // Match
        else if self.keyword("match") {
            let expr = self.expr();
            let mut arms: Vec<(Vec<ast::Pat>,Option<Box<ast::Expr>>,Box<ast::Expr>)> = Vec::new();
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
            ast::Expr::Match(Box::new(expr),arms)
        }

        // *Assign
        else {
            self.assign_expr()
        }
    }
}
