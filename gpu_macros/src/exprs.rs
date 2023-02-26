use super::*;

impl Parser {

    // Boolean, Integer, Float, UnknownIdent, UnknownStruct, UnknownTupleOrCall, UnknownVariant, AnonTuple, Array, Cloned
    pub(crate) fn primary_expr(&mut self) -> Expr {

        // Boolean
        if let Some(value) = self.boolean_literal() {
            Expr::Boolean(value)
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            Expr::Integer(value as i64)
        }

        // Float
        else if let Some(value) = self.float_literal() {
            Expr::Float(value)
        }

        // UnknownIdent, UnknownStruct, UnknownTupleOrCall, UnknownVariant
        else if let Some(ident) = self.ident() {

            // UnknownStruct
            if let Some(ident_exprs) = self.brace_ident_exprs() {
                Expr::UnknownStruct(ident,ident_exprs)
            }

            // UnknownTupleOrCall
            else if let Some(exprs) = self.paren_exprs() {
                Expr::UnknownTupleOrCall(ident,exprs)
            }

            // UnknownVariant
            else if self.punct2(':',':') {
                if self.punct('<') {
                    // process double :: style (Vec2::<T>) as UnknownStruct
                    let element_type = self.ident().expect("identifier expected");
                    self.punct('>');
                    let ident = format!("{}<{}>",ident,element_type);
                    if let Some(ident_exprs) = self.brace_ident_exprs() {
                        Expr::UnknownStruct(ident,ident_exprs)
                    }
                    else {
                        self.fatal(&format!("struct literal expected after {}",ident));
                    }
                }
                else {
                    let variant = self.ident().expect("identifier expected");
                    if let Some(ident_exprs) = self.brace_ident_exprs() {
                        Expr::UnknownVariant(ident,UnknownVariantExpr::Struct(variant,ident_exprs))
                    }
                    else if let Some(exprs) = self.paren_exprs() {
                        Expr::UnknownVariant(ident,UnknownVariantExpr::Tuple(variant,exprs))
                    }
                    else {
                        Expr::UnknownVariant(ident,UnknownVariantExpr::Naked(variant))
                    }
                }
            }

            // UnknownIdent
            else {
                Expr::UnknownIdent(ident)
            }
        }

        // AnonTuple
        else if let Some(exprs) = self.paren_exprs() {
            if exprs.len() == 1 {
                exprs[0].clone()
            }
            else {
                Expr::AnonTuple(exprs)
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

        else {
            self.fatal("expression expected");
        }
    }

    // UnknownField, UnknownMethod, UnknownTupleIndex, Index, Cast
    pub(crate) fn postfix_expr(&mut self) -> Expr {
        let mut expr = self.primary_expr();
        loop {

            // UnknownField, UnknownMethod, UnknownTupleIndex
            if self.punct('.') {

                // UnknownField, UnknownMethod
                if let Some(ident) = self.ident() {
                    if let Some(exprs) = self.paren_exprs() {
                        expr = Expr::UnknownMethod(Box::new(expr),ident,exprs);
                    }
                    else {
                        expr = Expr::UnknownField(Box::new(expr),ident);
                    }
                }

                // TupleIndex
                else if let Some(value) = self.integer_literal() {
                    expr = Expr::UnknownTupleIndex(Box::new(expr),value as usize);
                }

                else {
                    self.fatal("field or tuple index expected");
                }
            }

            // Index
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.expr()));
            }

            // Cast
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),Box::new(self.type_()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Neg, Not
    pub(crate) fn prefix_expr(&mut self) -> Expr {

        // Neg
        if self.punct('-') {
            Expr::Unary(UnaryOp::Neg,Box::new(self.prefix_expr()))
        }

        // Not
        else if self.punct('!') {
            Expr::Unary(UnaryOp::Not,Box::new(self.prefix_expr()))
        }

        else {
            self.postfix_expr()
        }
    }

    // Mul, Div, Mod
    pub(crate) fn mul_expr(&mut self) -> Expr {
        let mut expr = self.prefix_expr();
        loop {

            // Mul
            if self.punct('*') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mul,Box::new(self.prefix_expr()));
            }

            // Div
            else if self.punct('/') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Div,Box::new(self.prefix_expr()));
            }

            // Mod
            else if self.punct('%') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mod,Box::new(self.prefix_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Add, Sub
    pub(crate) fn add_expr(&mut self) -> Expr {
        let mut expr = self.mul_expr();
        loop {

            // Add
            if self.punct('+') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Add,Box::new(self.mul_expr()));
            }

            // Sub
            else if self.punct('-') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Sub,Box::new(self.mul_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // Shl, Shr
    pub(crate) fn shift_expr(&mut self) -> Expr {
        let mut expr = self.add_expr();
        loop {

            // Shl
            if self.punct2('<','<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shl,Box::new(self.add_expr()));
            }

            // Shr
            else if self.punct2('>','>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shr,Box::new(self.add_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // And
    pub(crate) fn and_expr(&mut self) -> Expr {
        let mut expr = self.shift_expr();
        while self.punct('&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::And,Box::new(self.shift_expr()));
        }
        expr
    }

    // Or
    pub(crate) fn or_expr(&mut self) -> Expr {
        let mut expr = self.and_expr();
        while self.punct('|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Or,Box::new(self.and_expr()));
        }
        expr
    }

    // Xor
    pub(crate) fn xor_expr(&mut self) -> Expr {
        let mut expr = self.or_expr();
        while self.punct('^') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Xor,Box::new(self.or_expr()));
        }
        expr
    }

    // Eq, NotEq, Less, Greater, LessEq, GreaterEq
    pub(crate) fn comp_expr(&mut self) -> Expr {
        let mut expr = self.xor_expr();
        loop {

            // Eq
            if self.punct2('=','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Eq,Box::new(self.xor_expr()));
            }

            // NotEq
            if self.punct2('!','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::NotEq,Box::new(self.xor_expr()));
            }

            // Less
            if self.punct('<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Less,Box::new(self.xor_expr()));
            }

            // Greater
            if self.punct('>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Greater,Box::new(self.xor_expr()));
            }

            // LessEq
            if self.punct2('<','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::LessEq,Box::new(self.xor_expr()));
            }

            // GreaterEq
            if self.punct2('>','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::GreaterEq,Box::new(self.xor_expr()));
            }

            else {
                break;
            }
        }
        expr
    }

    // LogAnd
    pub(crate) fn logand_expr(&mut self) -> Expr {
        let mut expr = self.comp_expr();
        while self.punct2('&','&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogAnd,Box::new(self.comp_expr()));
        }
        expr
    }

    // LogOr
    pub(crate) fn logor_expr(&mut self) -> Expr {
        let mut expr = self.logand_expr();
        while self.punct2('|','|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogOr,Box::new(self.logand_expr()));
        }
        expr
    }

    // Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign
    pub(crate) fn assign_expr(&mut self) -> Expr {
        let expr = self.logor_expr();
        if self.punct('=') {
            Expr::Binary(Box::new(expr),BinaryOp::Assign,Box::new(self.logor_expr()))
        }
        else if self.punct2('+','=') {
            Expr::Binary(Box::new(expr),BinaryOp::AddAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('-','=') {
            Expr::Binary(Box::new(expr),BinaryOp::SubAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('*','=') {
            Expr::Binary(Box::new(expr),BinaryOp::MulAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('/','=') {
            Expr::Binary(Box::new(expr),BinaryOp::DivAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('%','=') {
            Expr::Binary(Box::new(expr),BinaryOp::ModAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('&','=') {
            Expr::Binary(Box::new(expr),BinaryOp::AndAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('|','=') {
            Expr::Binary(Box::new(expr),BinaryOp::OrAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('^','=') {
            Expr::Binary(Box::new(expr),BinaryOp::XorAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('<','<','=') {
            Expr::Binary(Box::new(expr),BinaryOp::ShlAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('>','>','=') {
            Expr::Binary(Box::new(expr),BinaryOp::ShrAssign,Box::new(self.logor_expr()))
        }
        else {
            expr
        }
    }

    // Block
    pub(crate) fn block(&mut self) -> Option<Block> {
        if let Some(mut parser) = self.group('{') {
            Some(parser.finish_block(None))
        }
        else {
            None
        }
    }

    // If, IfLet, Block
    pub(crate) fn else_expr(&mut self) -> Option<Expr> {

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
                Some(self.if_expr())
            }
        }

        else {
            None
        }
    }

    pub(crate) fn finish_expr_tail(&mut self,mut expr: Expr) -> Expr {
        loop {
            if self.punct('.') {
                if let Some(ident) = self.ident() {
                    if let Some(exprs) = self.paren_exprs() {
                        expr = Expr::UnknownMethod(Box::new(expr),ident,exprs);
                    }
                    else {
                        expr = Expr::UnknownField(Box::new(expr),ident);
                    }
                }
                else if let Some(value) = self.integer_literal() {
                    expr = Expr::UnknownTupleIndex(Box::new(expr),value as usize);
                }
                else {
                    self.fatal("field or tuple index expected");
                }
            }
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.expr()));
            }
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),Box::new(self.type_()));
            }
            else {
                break;
            }
        }
        loop {
            if self.punct('*') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mul,Box::new(self.prefix_expr()));
            }
            else if self.punct('/') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Div,Box::new(self.prefix_expr()));
            }
            else if self.punct('%') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mod,Box::new(self.prefix_expr()));
            }
            else {
                break;
            }
        }        
        loop {
            if self.punct('+') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Add,Box::new(self.mul_expr()));
            }
            else if self.punct('-') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Sub,Box::new(self.mul_expr()));
            }
            else {
                break;
            }
        }
        loop {
            if self.punct2('<','<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shl,Box::new(self.add_expr()));
            }
            else if self.punct2('>','>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shr,Box::new(self.add_expr()));
            }       
            else {
                break;
            }
        }
        while self.punct('&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::And,Box::new(self.shift_expr()));
        }
        while self.punct('|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Or,Box::new(self.and_expr()));
        }
        while self.punct('^') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Xor,Box::new(self.or_expr()));
        }
        loop {
            if self.punct2('=','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Eq,Box::new(self.xor_expr()));
            }
            if self.punct2('!','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::NotEq,Box::new(self.xor_expr()));
            }
            if self.punct('<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Less,Box::new(self.xor_expr()));
            }
            if self.punct('>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Greater,Box::new(self.xor_expr()));
            }
            if self.punct2('<','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::LessEq,Box::new(self.xor_expr()));
            }
            if self.punct2('>','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::GreaterEq,Box::new(self.xor_expr()));
            }      
            else {
                break;
            }
        }
        while self.punct2('&','&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogAnd,Box::new(self.comp_expr()));
        }
        while self.punct2('|','|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogOr,Box::new(self.logand_expr()));
        }
        if self.punct('=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Assign,Box::new(self.logor_expr()))
        }
        else if self.punct2('+','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::AddAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('-','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::SubAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('*','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::MulAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('/','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::DivAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('%','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ModAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('&','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::AndAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('|','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::OrAssign,Box::new(self.logor_expr()))
        }
        else if self.punct2('^','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::XorAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('<','<','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ShlAssign,Box::new(self.logor_expr()))
        }
        else if self.punct3('>','>','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ShrAssign,Box::new(self.logor_expr()))
        }
        expr
    }

    pub(crate) fn finish_ident_as_expr(&mut self,ident: String) -> Expr {
        let expr = if let Some(exprs) = self.paren_exprs() {
            Expr::UnknownTupleOrCall(ident,exprs)
        }
        else if self.punct2(':',':') {
            if self.punct('<') {
                let element_type = self.ident().expect("identifier expected");
                self.punct('>');
                let ident = format!("{}<{}>",ident,element_type);
                if let Some(ident_exprs) = self.brace_ident_exprs() {
                    Expr::UnknownStruct(ident,ident_exprs)
                }
                else {
                    self.fatal(&format!("struct literal expected after {}",ident));
                }
            }
            else {
                let variant = self.ident().expect("identifier expected");
                if let Some(ident_exprs) = self.brace_ident_exprs() {
                    Expr::UnknownVariant(ident,UnknownVariantExpr::Struct(variant,ident_exprs))
                }
                else if let Some(exprs) = self.paren_exprs() {
                    Expr::UnknownVariant(ident,UnknownVariantExpr::Tuple(variant,exprs))
                }
                else {
                    Expr::UnknownVariant(ident,UnknownVariantExpr::Naked(variant))
                }
            }
        }
        else {
            Expr::UnknownIdent(ident)
        };

        self.finish_expr_tail(expr)
    }

    pub(crate) fn finish_variant_expr(&mut self,ident: String) -> Expr {
        let expr = if self.punct('<') {
            let element_type = self.ident().expect("identifier expected");
            self.punct('>');
            let ident = format!("{}<{}>",ident,element_type);
            if let Some(ident_exprs) = self.brace_ident_exprs() {
                Expr::UnknownStruct(ident,ident_exprs)
            }
            else {
                self.fatal(&format!("struct literal expected after {}",ident));
            }
        }
        else {
            let variant = self.ident().expect("identifier expected");
            if let Some(ident_exprs) = self.brace_ident_exprs() {
                Expr::UnknownVariant(ident,UnknownVariantExpr::Struct(variant,ident_exprs))
            }
            else if let Some(exprs) = self.paren_exprs() {
                Expr::UnknownVariant(ident,UnknownVariantExpr::Tuple(variant,exprs))
            }
            else {
                Expr::UnknownVariant(ident,UnknownVariantExpr::Naked(variant))
            }
        };
        self.finish_expr_tail(expr)
    }

    pub(crate) fn finish_variant_pat(&mut self,ident: String) -> Pat {
        let variant_ident = self.ident().expect("identifier expected");
        let pat = if let Some(ident_pats) = self.brace_ident_pats() {
            Pat::UnknownVariant(ident,UnknownVariantPat::Struct(variant_ident,ident_pats))
        }
        else if let Some(pats) = self.paren_pats() {
            Pat::UnknownVariant(ident,UnknownVariantPat::Tuple(variant_ident,pats))
        }
        else {
            Pat::UnknownVariant(ident,UnknownVariantPat::Naked(variant_ident))
        };
        pat
    }

    pub(crate) fn finish_pat(&mut self,ident: String) -> Pat {
        let mut pat = if let Some(ident_pats) = self.brace_ident_pats() {
            Pat::UnknownStruct(ident,ident_pats)
        }
        else if let Some(pats) = self.paren_pats() {
            Pat::UnknownTuple(ident,pats)
        }
        else if self.punct2(':',':') {
            self.finish_variant_pat(ident)
        }
        else {
            Pat::UnknownIdent(ident)
        };
        if self.punct3('.','.','=') {
            pat = Pat::Range(Box::new(pat),Box::new(self.pat()))
        }
        pat
    }

    pub(crate) fn finish_unknown_struct(&mut self,ident: String,sub_ident: String) -> Expr {
        let mut ident_exprs: Vec<(String,Expr)> = Vec::new();
        let expr = self.expr();
        ident_exprs.push((sub_ident,expr));
        self.punct(',');
        while !self.done() {
            let ident = self.ident().expect("identifier expected");
            if !self.punct(':') {
                panic!(": expected");
            }
            let expr = self.expr();
            ident_exprs.push((ident,expr));
        }
        Expr::UnknownStruct(ident,ident_exprs)
    }

    pub(crate) fn finish_block(&mut self,expr: Option<Expr>) -> Block {
        let mut stats: Vec<Stat> = Vec::new();
        let mut last_expr: Option<Box<Expr>> = None;
        if let Some(expr) = expr {
            if self.punct(';') {
                stats.push(Stat::Expr(Box::new(expr)));
            }
            else {
                if let Some(last_expr) = last_expr {
                    stats.push(Stat::Expr(last_expr));
                }
                last_expr = Some(Box::new(expr));
            }
        }
        while !self.done() {

            // Let
            if self.keyword("let") {
                let pat = self.pat();
                let type_ = if self.punct(':') {
                    self.type_()
                }
                else {
                    Type::Inferred
                };
                self.punct('=');
                let expr = self.expr();
                self.punct(';');
                stats.push(Stat::Let(Box::new(pat),Box::new(type_),Box::new(expr)));
            }

            // Expr
            else {
                let expr = self.expr();
                if self.punct(';') {
                    stats.push(Stat::Expr(Box::new(expr)));
                }
                else {
                    if let Some(last_expr) = last_expr {
                        stats.push(Stat::Expr(last_expr));
                    }
                    last_expr = Some(Box::new(expr));
                }
            }
        }
        Block { stats,expr: last_expr, }
    }

    pub(crate) fn expr_brace_pat(&mut self) -> (Expr,Pat,Parser) {

        // ident
        if let Some(ident) = self.ident() {

            // ident {
            if let Some(mut parser) = self.group('{') {
                
                // ident { sub_ident
                if let Some(sub_ident) = parser.ident() {

                    // ident { sub_ident :
                    if parser.punct(':') {

                        // ident { sub_ident::...  -> variant pattern
                        if parser.punct(':') {
                            let expr = Expr::UnknownIdent(ident);
                            let mut pat = parser.finish_variant_pat(sub_ident);
                            if self.punct3('.','.','=') {
                                pat = Pat::Range(Box::new(pat),Box::new(self.pat()))
                            }
                            (expr,pat,parser)
                        }

                        // ident { sub_ident: ...  -> UnknownStruct followed by { pat
                        else {
                            let expr = parser.finish_unknown_struct(ident,sub_ident);
                            if let Some(mut sub_parser) = self.group('{') {
                                let pat = sub_parser.pat();
                                (expr,pat,sub_parser)
                            }
                            else {
                                panic!("{}","{ expected");
                            }
                        }
                    }

                    // ident { sub_ident ...  -> expr { pat
                    else {
                        let expr = Expr::UnknownIdent(ident);
                        let pat = parser.finish_pat(sub_ident);
                        (expr,pat,parser)
                    }
                }

                // ident { ...  -> expr {
                else {
                    let expr = Expr::UnknownIdent(ident);
                    let pat = parser.pat();
                    (expr,pat,parser)
                }
            }

            else {
                panic!("{}","{ expected");
            }
        }

        // ...
        else {
            let expr = self.expr();
            if let Some(mut parser) = self.group('{') {
                let pat = parser.pat();
                (expr,pat,parser)
            }
            else {
                panic!("{}","{ expected");
            }
        }
    }

    pub(crate) fn expr_brace_block(&mut self) -> (Expr,Block) {

        // ident
        if let Some(ident) = self.ident() {

            // ident {
            if let Some(mut parser) = self.group('{') {
                
                // ident { sub_ident
                if let Some(sub_ident) = parser.ident() {

                    // ident { sub_ident :
                    if parser.punct(':') {

                        // ident { sub_ident::...  -> variant expr
                        if parser.punct(':') {
                            let main_expr = Expr::UnknownIdent(ident);
                            let expr = parser.finish_variant_expr(sub_ident);
                            let block = parser.finish_block(Some(expr));
                            (main_expr,block)
                        }

                        // ident { sub_ident: ...  -> UnknownStruct followed by { ... }
                        else {
                            let main_expr = parser.finish_unknown_struct(ident,sub_ident);
                            let block = self.block().expect("{ expected");
                            (main_expr,block)
                        }
                    }

                    // ident { sub_ident ...  -> expr { ... }
                    else {
                        let main_expr = Expr::UnknownIdent(ident);
                        let expr = parser.finish_ident_as_expr(sub_ident);
                        let block = parser.finish_block(Some(expr));
                        (main_expr,block)
                    }
                }

                // ident { ...  -> expr {
                else {
                    let main_expr = Expr::UnknownIdent(ident);
                    let expr = parser.expr();
                    let block = parser.finish_block(Some(expr));
                    (main_expr,block)
                }
            }

            else {
                let main_expr = self.finish_ident_as_expr(ident);
                let block = self.block().expect("{ expected");
                (main_expr,block)
            }
        }

        // ...
        else {
            let main_expr = self.expr();
            let block = self.block().expect("{ expected");
            (main_expr,block)
        }
    }

    pub(crate) fn if_expr(&mut self) -> Expr {
        let (expr,block) = self.expr_brace_block();
        if self.keyword("else") {
            let else_expr = self.else_expr().expect("if, if let, or block expected");
            Expr::If(Box::new(expr),block,Some(Box::new(else_expr)))
        }
        else {
            Expr::If(Box::new(expr),block,None)
        }
    }

    pub(crate) fn while_expr(&mut self) -> Expr {
        let (expr,block) = self.expr_brace_block();
        Expr::While(Box::new(expr),block)
    }

    pub(crate) fn match_expr(&mut self) -> Expr {

        let (match_expr,pat,mut parser) = self.expr_brace_pat();
        let mut pats: Vec<Pat> = Vec::new();
        pats.push(pat);
        while parser.punct('|') {
            pats.push(parser.ranged_pat());
        }
        let mut arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
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
        Expr::Match(Box::new(match_expr),arms)
    }

    // Continue, Break, Return, Block, If, IfLet, Loop, For, While, WhileLet, Match, *Assign
    pub(crate) fn expr(&mut self) -> Expr {

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
                self.while_expr()
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

            self.match_expr()
        }

        // *Assign
        else {
            self.assign_expr()
        }
    }
}
