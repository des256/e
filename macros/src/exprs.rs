use super::*;

impl Parser {

    // Boolean, Integer, Float, UnknownIdent, UnknownStruct, UnknownTupleFunctionCall, UnknownVariant, AnonTuple, Array, Cloned
    pub(crate) fn primary_expr(&mut self) -> Result<Expr,String> {

        // Boolean
        if let Some(value) = self.boolean_literal() {
            Ok(Expr::Boolean(value))
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            Ok(Expr::Integer(value as i64))
        }

        // Float
        else if let Some(value) = self.float_literal() {
            Ok(Expr::Float(value))
        }

        // Ident, Struct, TupleOrCall, Variant
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_exprs) = self.brace_ident_exprs()? {
                Ok(Expr::Struct(ident,ident_exprs))
            }

            // TupleOrCall
            else if let Some(exprs) = self.paren_exprs()? {
                Ok(Expr::TupleOrCall(ident,exprs))
            }

            // Variant
            else if self.punct2(':',':') {
                if self.punct('<') {
                    // process double :: style (Vec2::<T>) as UnknownStruct
                    let element_type = self.ident().expect("identifier expected");
                    self.punct('>');
                    let ident = format!("{}<{}>",ident,element_type);
                    if let Some(ident_exprs) = self.brace_ident_exprs()? {
                        Ok(Expr::Struct(ident,ident_exprs))
                    }
                    else {
                        self.err(&format!("struct literal expected after {}",ident))
                    }
                }
                else {
                    let variant_ident = self.ident().expect("identifier expected");
                    if let Some(ident_exprs) = self.brace_ident_exprs()? {
                        Ok(Expr::Variant(ident,variant_ident,VariantExpr::Struct(ident_exprs)))
                    }
                    else if let Some(exprs) = self.paren_exprs()? {
                        Ok(Expr::Variant(ident,variant_ident,VariantExpr::Tuple(exprs)))
                    }
                    else {
                        Ok(Expr::Variant(ident,variant_ident,VariantExpr::Naked))
                    }
                }
            }

            // UnknownLocalConst
            else {
                Ok(Expr::Ident(ident))
            }
        }

        // AnonTuple
        else if let Some(exprs) = self.paren_exprs()? {
            if exprs.len() == 1 {
                Ok(exprs[0].clone())
            }
            else {
                Ok(Expr::AnonTuple(exprs))
            }
        }

        // Array, Cloned
        else if let Some(mut parser) = self.group('[') {
            let mut exprs: Vec<Expr> = Vec::new();
            while !parser.done() {
                let expr = parser.expr()?;
                if parser.punct(';') {
                    let expr2 = parser.expr()?;
                    return Ok(Expr::Cloned(Box::new(expr),Box::new(expr2)));
                }
                else {
                    exprs.push(expr);
                }
                parser.punct(',');
            }
            Ok(Expr::Array(exprs))
        }

        else {
            self.err("expression expected")
        }
    }

    // Field, Method, TupleIndex, Index, Cast
    pub(crate) fn postfix_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.primary_expr()?;
        loop {

            // Field, Method, TupleIndex
            if self.punct('.') {

                // Field, Method
                if let Some(ident) = self.ident() {
                    if let Some(exprs) = self.paren_exprs()? {
                        expr = Expr::Method(Box::new(expr),ident,exprs);
                    }
                    else {
                        expr = Expr::Field(Box::new(expr),ident);
                    }
                }

                // tuple index -> Field
                else if let Some(value) = self.integer_literal() {
                    expr = Expr::Field(Box::new(expr),format!("f{}",value));
                }

                else {
                    return self.err("field or tuple index expected");
                }
            }

            // Index
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.expr()?));
            }

            // Cast
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),Box::new(self.type_()?));
            }

            else {
                break;
            }
        }
        Ok(expr)
    }

    // Neg, Not
    pub(crate) fn prefix_expr(&mut self) -> Result<Expr,String> {

        // Neg
        if self.punct('-') {
            Ok(Expr::Unary(UnaryOp::Neg,Box::new(self.prefix_expr()?)))
        }

        // Not
        else if self.punct('!') {
            Ok(Expr::Unary(UnaryOp::Not,Box::new(self.prefix_expr()?)))
        }

        else {
            self.postfix_expr()
        }
    }

    // Mul, Div, Mod
    pub(crate) fn mul_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.prefix_expr()?;
        loop {

            // Mul
            if self.punct('*') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mul,Box::new(self.prefix_expr()?));
            }

            // Div
            else if self.punct('/') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Div,Box::new(self.prefix_expr()?));
            }

            // Mod
            else if self.punct('%') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mod,Box::new(self.prefix_expr()?));
            }

            else {
                break;
            }
        }
        Ok(expr)
    }

    // Add, Sub
    pub(crate) fn add_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.mul_expr()?;
        loop {

            // Add
            if self.punct('+') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Add,Box::new(self.mul_expr()?));
            }

            // Sub
            else if self.punct('-') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Sub,Box::new(self.mul_expr()?));
            }

            else {
                break;
            }
        }
        Ok(expr)
    }

    // Shl, Shr
    pub(crate) fn shift_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.add_expr()?;
        loop {

            // Shl
            if self.punct2('<','<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shl,Box::new(self.add_expr()?));
            }

            // Shr
            else if self.punct2('>','>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shr,Box::new(self.add_expr()?));
            }

            else {
                break;
            }
        }
        Ok(expr)
    }

    // And
    pub(crate) fn and_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.shift_expr()?;
        while self.punct('&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::And,Box::new(self.shift_expr()?));
        }
        Ok(expr)
    }

    // Or
    pub(crate) fn or_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.and_expr()?;
        while self.punct('|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Or,Box::new(self.and_expr()?));
        }
        Ok(expr)
    }

    // Xor
    pub(crate) fn xor_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.or_expr()?;
        while self.punct('^') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Xor,Box::new(self.or_expr()?));
        }
        Ok(expr)
    }

    // Eq, NotEq, Less, Greater, LessEq, GreaterEq
    pub(crate) fn comp_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.xor_expr()?;
        loop {

            // Eq
            if self.punct2('=','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Eq,Box::new(self.xor_expr()?));
            }

            // NotEq
            if self.punct2('!','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::NotEq,Box::new(self.xor_expr()?));
            }

            // Less
            if self.punct('<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Less,Box::new(self.xor_expr()?));
            }

            // Greater
            if self.punct('>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Greater,Box::new(self.xor_expr()?));
            }

            // LessEq
            if self.punct2('<','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::LessEq,Box::new(self.xor_expr()?));
            }

            // GreaterEq
            if self.punct2('>','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::GreaterEq,Box::new(self.xor_expr()?));
            }

            else {
                break;
            }
        }
        Ok(expr)
    }

    // LogAnd
    pub(crate) fn logand_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.comp_expr()?;
        while self.punct2('&','&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogAnd,Box::new(self.comp_expr()?));
        }
        Ok(expr)
    }

    // LogOr
    pub(crate) fn logor_expr(&mut self) -> Result<Expr,String> {
        let mut expr = self.logand_expr()?;
        while self.punct2('|','|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogOr,Box::new(self.logand_expr()?));
        }
        Ok(expr)
    }

    // Assign, AddAssign, SubAssign, MulAssign, DivAssign, ModAssign, AndAssign, OrAssign, XorAssign, ShlAssign, ShrAssign
    pub(crate) fn assign_expr(&mut self) -> Result<Expr,String> {
        let expr = self.logor_expr()?;
        if self.punct('=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::Assign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('+','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::AddAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('-','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::SubAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('*','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::MulAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('/','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::DivAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('%','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::ModAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('&','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::AndAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('|','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::OrAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct2('^','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::XorAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct3('<','<','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::ShlAssign,Box::new(self.logor_expr()?)))
        }
        else if self.punct3('>','>','=') {
            Ok(Expr::Binary(Box::new(expr),BinaryOp::ShrAssign,Box::new(self.logor_expr()?)))
        }
        else {
            Ok(expr)
        }
    }

    // Block
    pub(crate) fn block(&mut self) -> Result<Option<Block>,String> {
        if let Some(mut parser) = self.group('{') {
            Ok(Some(parser.finish_block(None)?))
        }
        else {
            Ok(None)
        }
    }

    // If, IfLet, Block
    pub(crate) fn else_expr(&mut self) -> Result<Option<Expr>,String> {

        // Block
        if let Some(block) = self.block()? {
            Ok(Some(Expr::Block(block)))
        }

        // If, IfLet
        else if self.keyword("if") {

            // IfLet
            if self.keyword("let") {
                let pats = self.or_pats()?;
                self.punct('=');
                let expr = self.expr()?;
                let block = self.block()?;
                if block.is_none() {
                    return self.err("{{ expected");
                }
                let block = block.unwrap();
                if self.keyword("else") {
                    let else_expr = self.else_expr()?;
                    if else_expr.is_none() {
                        return self.err("else-expression expected");
                    }
                    let else_expr = else_expr.unwrap();
                    Ok(Some(Expr::IfLet(pats,Box::new(expr),block,Some(Box::new(else_expr)))))
                }
                else {
                    Ok(Some(Expr::IfLet(pats,Box::new(expr),block,None)))
                }
            }

            // If
            else {
                Ok(Some(self.if_expr()?))
            }
        }

        else {
            Ok(None)
        }
    }

    pub(crate) fn finish_expr_tail(&mut self,mut expr: Expr) -> Result<Expr,String> {
        loop {
            if self.punct('.') {
                if let Some(ident) = self.ident() {
                    if let Some(exprs) = self.paren_exprs()? {
                        expr = Expr::Method(Box::new(expr),ident,exprs);
                    }
                    else {
                        expr = Expr::Field(Box::new(expr),ident);
                    }
                }
                else if let Some(value) = self.integer_literal() {
                    expr = Expr::Field(Box::new(expr),format!("f{}",value));
                }
                else {
                    return self.err("field or tuple index expected");
                }
            }
            else if let Some(mut parser) = self.group('[') {
                expr = Expr::Index(Box::new(expr),Box::new(parser.expr()?));
            }
            else if self.keyword("as") {
                expr = Expr::Cast(Box::new(expr),Box::new(self.type_()?));
            }
            else {
                break;
            }
        }
        loop {
            if self.punct('*') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mul,Box::new(self.prefix_expr()?));
            }
            else if self.punct('/') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Div,Box::new(self.prefix_expr()?));
            }
            else if self.punct('%') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Mod,Box::new(self.prefix_expr()?));
            }
            else {
                break;
            }
        }        
        loop {
            if self.punct('+') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Add,Box::new(self.mul_expr()?));
            }
            else if self.punct('-') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Sub,Box::new(self.mul_expr()?));
            }
            else {
                break;
            }
        }
        loop {
            if self.punct2('<','<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shl,Box::new(self.add_expr()?));
            }
            else if self.punct2('>','>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Shr,Box::new(self.add_expr()?));
            }       
            else {
                break;
            }
        }
        while self.punct('&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::And,Box::new(self.shift_expr()?));
        }
        while self.punct('|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Or,Box::new(self.and_expr()?));
        }
        while self.punct('^') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Xor,Box::new(self.or_expr()?));
        }
        loop {
            if self.punct2('=','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Eq,Box::new(self.xor_expr()?));
            }
            if self.punct2('!','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::NotEq,Box::new(self.xor_expr()?));
            }
            if self.punct('<') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Less,Box::new(self.xor_expr()?));
            }
            if self.punct('>') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::Greater,Box::new(self.xor_expr()?));
            }
            if self.punct2('<','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::LessEq,Box::new(self.xor_expr()?));
            }
            if self.punct2('>','=') {
                expr = Expr::Binary(Box::new(expr),BinaryOp::GreaterEq,Box::new(self.xor_expr()?));
            }      
            else {
                break;
            }
        }
        while self.punct2('&','&') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogAnd,Box::new(self.comp_expr()?));
        }
        while self.punct2('|','|') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::LogOr,Box::new(self.logand_expr()?));
        }
        if self.punct('=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::Assign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('+','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::AddAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('-','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::SubAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('*','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::MulAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('/','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::DivAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('%','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ModAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('&','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::AndAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('|','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::OrAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct2('^','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::XorAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct3('<','<','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ShlAssign,Box::new(self.logor_expr()?))
        }
        else if self.punct3('>','>','=') {
            expr = Expr::Binary(Box::new(expr),BinaryOp::ShrAssign,Box::new(self.logor_expr()?))
        }
        Ok(expr)
    }

    pub(crate) fn finish_ident_as_expr(&mut self,ident: String) -> Result<Expr,String> {
        let expr = if let Some(exprs) = self.paren_exprs()? {
            Expr::TupleOrCall(ident,exprs)
        }
        else if self.punct2(':',':') {
            if self.punct('<') {
                let element_type = self.ident();
                if element_type.is_none() {
                    return self.err("identifier expected");
                }
                let element_type = element_type.unwrap();
                self.punct('>');
                let ident = format!("{}<{}>",ident,element_type);
                if let Some(ident_exprs) = self.brace_ident_exprs()? {
                    Expr::Struct(ident,ident_exprs)
                }
                else {
                    return self.err(&format!("struct literal expected after {}",ident));
                }
            }
            else {
                let variant_ident = self.ident();
                if variant_ident.is_none() {
                    return self.err("identifier expected");
                }
                let variant_ident = variant_ident.unwrap();
                if let Some(ident_exprs) = self.brace_ident_exprs()? {
                    Expr::Variant(ident,variant_ident,VariantExpr::Struct(ident_exprs))
                }
                else if let Some(exprs) = self.paren_exprs()? {
                    Expr::Variant(ident,variant_ident,VariantExpr::Tuple(exprs))
                }
                else {
                    Expr::Variant(ident,variant_ident,VariantExpr::Naked)
                }
            }
        }
        else {
            Expr::Ident(ident)
        };

        self.finish_expr_tail(expr)
    }

    pub(crate) fn finish_variant_expr(&mut self,ident: String) -> Result<Expr,String> {
        let expr = if self.punct('<') {
            let element_type = self.ident();
            if element_type.is_none() {
                return self.err("identifier expected");
            }
            let element_type = element_type.unwrap();
            self.punct('>');
            let ident = format!("{}<{}>",ident,element_type);
            if let Some(ident_exprs) = self.brace_ident_exprs()? {
                Expr::Struct(ident,ident_exprs)
            }
            else {
                return self.err(&format!("struct literal expected after {}",ident));
            }
        }
        else {
            let variant_ident = self.ident();
            if variant_ident.is_none() {
                return self.err("identifier expected");
            }
            let variant_ident = variant_ident.unwrap();
            if let Some(ident_exprs) = self.brace_ident_exprs()? {
                Expr::Variant(ident,variant_ident,VariantExpr::Struct(ident_exprs))
            }
            else if let Some(exprs) = self.paren_exprs()? {
                Expr::Variant(ident,variant_ident,VariantExpr::Tuple(exprs))
            }
            else {
                Expr::Variant(ident,variant_ident,VariantExpr::Naked)
            }
        };
        self.finish_expr_tail(expr)
    }

    pub(crate) fn finish_variant_pat(&mut self,ident: String) -> Result<Pat,String> {
        let variant_ident = self.ident();
        if variant_ident.is_none() {
            return self.err("identifier expected");
        }
        let variant_ident = variant_ident.unwrap();
        let pat = if let Some(ident_pats) = self.brace_ident_pats()? {
            Pat::Variant(ident,variant_ident,VariantPat::Struct(ident_pats))
        }
        else if let Some(pats) = self.paren_pats()? {
            Pat::Variant(ident,variant_ident,VariantPat::Tuple(pats))
        }
        else {
            Pat::Variant(ident,variant_ident,VariantPat::Naked)
        };
        Ok(pat)
    }

    pub(crate) fn finish_pat(&mut self,ident: String) -> Result<Pat,String> {
        let mut pat = if let Some(ident_pats) = self.brace_ident_pats()? {
            Pat::Struct(ident,ident_pats)
        }
        else if let Some(pats) = self.paren_pats()? {
            Pat::Tuple(ident,pats)
        }
        else if self.punct2(':',':') {
            self.finish_variant_pat(ident)?
        }
        else {
            Pat::Ident(ident)
        };
        if self.punct3('.','.','=') {
            pat = Pat::Range(Box::new(pat),Box::new(self.pat()?))
        }
        Ok(pat)
    }

    pub(crate) fn finish_unknown_struct(&mut self,ident: String,sub_ident: String) -> Result<Expr,String> {
        let mut ident_exprs: Vec<(String,Expr)> = Vec::new();
        let expr = self.expr()?;
        ident_exprs.push((sub_ident,expr));
        self.punct(',');
        while !self.done() {
            let ident = self.ident();
            if ident.is_none() {
                return self.err("identifier expected");
            }
            let ident = ident.unwrap();
            if !self.punct(':') {
                return self.err(": expected");
            }
            let expr = self.expr()?;
            ident_exprs.push((ident,expr));
        }
        Ok(Expr::Struct(ident,ident_exprs))
    }

    pub(crate) fn finish_block(&mut self,expr: Option<Expr>) -> Result<Block,String> {
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
                let pat = self.pat()?;
                let type_ = if self.punct(':') {
                    self.type_()?
                }
                else {
                    Type::Inferred
                };
                self.punct('=');
                let expr = self.expr()?;
                self.punct(';');
                stats.push(Stat::Let(Box::new(pat),Box::new(type_),Box::new(expr)));
            }

            // Expr
            else {
                let expr = self.expr()?;
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
        Ok(Block { stats,expr: last_expr, })
    }

    pub(crate) fn expr_brace_pat(&mut self) -> Result<(Expr,Pat,Parser),String> {

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
                            let expr = Expr::Ident(ident);
                            let mut pat = parser.finish_variant_pat(sub_ident)?;
                            if self.punct3('.','.','=') {
                                pat = Pat::Range(Box::new(pat),Box::new(self.pat()?))
                            }
                            Ok((expr,pat,parser))
                        }

                        // ident { sub_ident: ...  -> UnknownStruct followed by { pat
                        else {
                            let expr = parser.finish_unknown_struct(ident,sub_ident)?;
                            if let Some(mut sub_parser) = self.group('{') {
                                let pat = sub_parser.pat()?;
                                Ok((expr,pat,sub_parser))
                            }
                            else {
                                self.err("{{ expected (expr_brace_pat)")
                            }
                        }
                    }

                    // ident { sub_ident ...  -> expr { pat
                    else {
                        let expr = Expr::Ident(ident);
                        let pat = parser.finish_pat(sub_ident)?;
                        Ok((expr,pat,parser))
                    }
                }

                // ident { ...  -> expr {
                else {
                    let expr = Expr::Ident(ident);
                    let pat = parser.pat()?;
                    Ok((expr,pat,parser))
                }
            }

            else {
                self.err("{{ expected (expr_brace_pat)")
            }
        }

        // ...
        else {
            let expr = self.expr()?;
            if let Some(mut parser) = self.group('{') {
                let pat = parser.pat()?;
                Ok((expr,pat,parser))
            }
            else {
                self.err("{{ expected (expr_brace_pat)")
            }
        }
    }

    pub(crate) fn expr_brace_block(&mut self) -> Result<(Expr,Block),String> {

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
                            let main_expr = Expr::Ident(ident);
                            let expr = parser.finish_variant_expr(sub_ident)?;
                            let block = parser.finish_block(Some(expr))?;
                            Ok((main_expr,block))
                        }

                        // ident { sub_ident: ...  -> UnknownStruct followed by { ... }
                        else {
                            let main_expr = parser.finish_unknown_struct(ident,sub_ident)?;
                            let block = self.block()?;
                            if block.is_none() {
                                return self.err("{{ expected (expr_brace_block)");
                            }
                            let block = block.unwrap();
                            Ok((main_expr,block))
                        }
                    }

                    // ident { sub_ident ...  -> expr { ... }
                    else {
                        let main_expr = Expr::Ident(ident);
                        let expr = parser.finish_ident_as_expr(sub_ident)?;
                        let block = parser.finish_block(Some(expr))?;
                        Ok((main_expr,block))
                    }
                }

                // ident { ...  -> expr {
                else {
                    let main_expr = Expr::Ident(ident);
                    let expr = parser.expr()?;
                    let block = parser.finish_block(Some(expr))?;
                    Ok((main_expr,block))
                }
            }

            else {
                let main_expr = self.finish_ident_as_expr(ident)?;
                let block = self.block()?;
                if block.is_none() {
                    return self.err("{{ expected");
                }
                let block = block.unwrap();
                Ok((main_expr,block))
            }
        }

        // ...
        else {
            let main_expr = self.expr()?;
            let block = self.block()?;
            if block.is_none() {
                return self.err("{{ expected (expr_brace_block)");
            }
            let block = block.unwrap();
            Ok((main_expr,block))
        }
    }

    pub(crate) fn if_expr(&mut self) -> Result<Expr,String> {
        let (expr,block) = self.expr_brace_block()?;
        if self.keyword("else") {
            let else_expr = self.else_expr()?;
            if else_expr.is_none() {
                return self.err("if, if let, or block expected");
            }
            let else_expr = else_expr.unwrap();
            Ok(Expr::If(Box::new(expr),block,Some(Box::new(else_expr))))
        }
        else {
            Ok(Expr::If(Box::new(expr),block,None))
        }
    }

    pub(crate) fn while_expr(&mut self) -> Result<Expr,String> {
        let (expr,block) = self.expr_brace_block()?;
        Ok(Expr::While(Box::new(expr),block))
    }

    pub(crate) fn match_expr(&mut self) -> Result<Expr,String> {

        let (match_expr,pat,mut parser) = self.expr_brace_pat()?;
        let mut pats: Vec<Pat> = Vec::new();
        pats.push(pat);
        while parser.punct('|') {
            pats.push(parser.ranged_pat()?);
        }
        let mut arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
        let if_expr = if parser.keyword("if") {
            Some(Box::new(parser.expr()?))
        }
        else {
            None
        };
        parser.punct2('=','>');
        let expr = parser.expr()?;
        parser.punct(',');
        arms.push((pats,if_expr,Box::new(expr)));
        while !parser.done() {
            let pats = parser.or_pats()?;
            let if_expr = if parser.keyword("if") {
                Some(Box::new(parser.expr()?))
            }
            else {
                None
            };
            parser.punct2('=','>');
            let expr = parser.expr()?;
            parser.punct(',');
            arms.push((pats,if_expr,Box::new(expr)));
        }
        Ok(Expr::Match(Box::new(match_expr),arms))
    }

    // Continue, Break, Return, Block, If, IfLet, Loop, For, While, WhileLet, Match, *Assign
    pub(crate) fn expr(&mut self) -> Result<Expr,String> {

        // Continue
        if self.keyword("continue") {
            Ok(Expr::Continue)
        }

        // Break
        else if self.keyword("break") {
            if !self.peek_punct(';') {
                let expr = self.expr()?;
                Ok(Expr::Break(Some(Box::new(expr))))
            }
            else {
                Ok(Expr::Break(None))
            }
        }

        // Return
        else if self.keyword("return") {
            if !self.peek_punct(';') {
                let expr = self.expr()?;
                Ok(Expr::Return(Some(Box::new(expr))))
            }
            else {
                Ok(Expr::Return(None))
            }
        }

        // If, IfLet, Block
        else if let Some(expr) = self.else_expr()? {
            Ok(expr)
        }

        // While, WhileLet
        else if self.keyword("while") {

            // WhileLet
            if self.keyword("let") {
                let pats = self.or_pats()?;
                self.punct('=');
                let expr = self.expr()?;
                let block = self.block()?;
                if block.is_none() {
                    return self.err("{{ expected (expr, while)");
                }
                let block = block.unwrap();
                Ok(Expr::WhileLet(pats,Box::new(expr),block))
            }

            // While
            else {
                self.while_expr()
            }
        }

        // Loop
        else if self.keyword("loop") {
            let block = self.block()?;
            if block.is_none() {
                return self.err("{{ expected (expr, loop)");
            }
            let block = block.unwrap();
            Ok(Expr::Loop(block))
        }

        // For
        else if self.keyword("for") {
            let pats = self.or_pats()?;
            self.keyword("in");
            let range = if self.punct2('.','.') {
                if self.peek_group('{') {
                    Range::All
                }
                else {
                    if self.punct('=') {
                        Range::ToIncl(Box::new(self.expr()?))
                    }
                    else {
                        Range::To(Box::new(self.expr()?))
                    }
                }
            }
            else {
                let expr = self.expr()?;
                if self.punct2('.','.') {
                    if self.peek_group('{') {
                        Range::From(Box::new(expr))
                    }
                    else {
                        if self.punct('=') {
                            Range::FromToIncl(Box::new(expr),Box::new(self.expr()?))
                        }
                        else {
                            Range::FromTo(Box::new(expr),Box::new(self.expr()?))
                        }
                    }
                }
                else {
                    Range::Only(Box::new(expr))
                }
            };
            let block = self.block()?;
            if block.is_none() {
                return self.err("{{ expected");
            }
            let block = block.unwrap();
            Ok(Expr::For(pats,range,block))
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
