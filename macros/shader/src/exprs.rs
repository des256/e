use {
    crate::*,
};

pub(crate) fn parse_primary_expr(lexer: &Lexer) -> Expr {
    if let Some(literal) = lexer.literal() {
        Expr::Literal(literal)
    }
    else if lexer.ident("true") {
        Expr::Literal("true".to_string())  // TODO: real bool
    }
    else if lexer.ident("false") {
        Expr::Literal("false".to_string())  // TODO: real bool
    }
    else {
        Expr::Segs(parse_segs(lexer))
    }
}

pub(crate) fn parse_direct_expr(lexer: &Lexer) -> Expr {
    if let Some(sublexer) = lexer.group('[') {
        let mut exprs: Vec<Box<Expr>> = Vec::new();
        if sublexer.done() {
            Expr::Array(Vec::new())
        }
        else {
            exprs.push(Box::new(parse_expr(&sublexer)));
            if sublexer.punct(';') {
                let expr = parse_expr(&sublexer);
                Expr::CloneArray(exprs[0],Box::new(expr))
            }
            else {
                sublexer.punct(',');
                while !sublexer.done() {
                    exprs.push(Box::new(parse_expr(&sublexer)));
                    sublexer.punct(',');
                }
                Expr::Array(exprs)
            }
        }
    }
    else if let Some(sublexer) = lexer.group('(') {
        let mut exprs: Vec<Box<Expr>> = Vec::new();
        while !sublexer.done() {
            exprs.push(Box::new(parse_expr(&sublexer)));
            sublexer.punct(',');
        }
        Expr::Tuple(exprs)
    }
    else {
        let expr = parse_primary_expr(lexer);
        if let Some(sublexer) = lexer.group('{') {
            let mut fields: Vec<ExprField> = Vec::new();
            let mut last_expr: Option<Box<Expr>> = None;
            if sublexer.punct2('.','.') {
                let last_expr = Some(Box::new(parse_expr(&sublexer)));
                Expr::StructEnumStruct(Box::new(expr),Vec::new(),last_expr)
            }
            else {
                while !sublexer.done() {
                    fields.push(if let Some(ident) = sublexer.any_ident() {
                        if sublexer.punct(':') {
                            let expr = parse_expr(&sublexer);
                            ExprField::IdentExpr(ident,Box::new(expr))
                        }
                        else {
                            ExprField::Ident(ident)
                        }
                    }
                    else if let Some(literal) = sublexer.literal() {
                        sublexer.punct(':');
                        let expr = parse_expr(&sublexer);
                        ExprField::LiteralExpr(literal,Box::new(expr))
                    }
                    else {
                        panic!("identifier or literal expected");
                    });
                    sublexer.punct(',');
                    if sublexer.punct2('.','.') {
                        last_expr = Some(Box::new(parse_expr(&sublexer)));
                        sublexer.punct(',');
                    }
                }
                Expr::StructEnumStruct(Box::new(expr),fields,last_expr)
            }
        }
        else if let Some(sublexer) = lexer.group('(') {
            let mut exprs: Vec<Box<Expr>> = Vec::new();
            while !sublexer.done() {
                exprs.push(Box::new(parse_expr(&sublexer)));
                sublexer.punct(',');
            }
            Expr::StructEnumTuple(Box::new(expr),exprs)
        }
        else {
            Expr::StructEnum(Box::new(expr))
        }
    }
}

pub(crate) fn parse_field_index_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_direct_expr(lexer);
    while lexer.punct('.') {
        if let Some(literal) = lexer.literal() {
            expr = Expr::TupleIndex(Box::new(expr),literal);
        }
        else if let Some(ident) = lexer.any_ident() {
            expr = Expr::Field(Box::new(expr),ident)
        }
    }
    expr
}

pub(crate) fn parse_index_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_field_index_expr(lexer);
    while let Some(sublexer) = lexer.group('[') {
        expr = Expr::Index(Box::new(expr),Box::new(parse_expr(&sublexer)));
    }
    expr
}

pub(crate) fn parse_call_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_index_expr(lexer);
    while let Some(sublexer) = lexer.group('(') {
        let exprs: Vec<Box<Expr>> = Vec::new();
        while !sublexer.done() {
            exprs.push(Box::new(parse_expr(&sublexer)));
            sublexer.punct(',');
        }
        expr = Expr::Call(Box::new(expr),exprs);
    }
    expr
}

pub(crate) fn parse_error_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_call_expr(lexer);
    while lexer.punct('?') {
        expr = Expr::Error(Box::new(expr));
    }
    expr
}

pub(crate) fn parse_borrow_expr(lexer: &Lexer) -> Expr {
    if lexer.punct('&') {
        let is_double = lexer.punct('&');
        let is_mut = lexer.ident("mut");
        let mut expr = parse_error_expr(lexer);
        Expr::Borrow(is_double,is_mut,Box::new(expr))
    }
    else {
        parse_error_expr(lexer)
    }
}

pub(crate) fn parse_deref_expr(lexer: &Lexer) -> Expr {
    if lexer.punct('*') {
        Expr::Deref(Box::new(parse_deref_expr(lexer)))
    }
    else {
        parse_borrow_expr(lexer)
    }
}

pub(crate) fn parse_neg_expr(lexer: &Lexer) -> Expr {
    if lexer.punct('-') {
        Expr::Negate(Box::new(parse_neg_expr(lexer)))
    }
    else if lexer.punct('!') {
        Expr::LogNot(Box::new(parse_neg_expr(lexer)))
    }
    else {
        parse_deref_expr(lexer)
    }
}

pub(crate) fn parse_cast_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_neg_expr(lexer);
    while lexer.ident("as") {
        expr = Expr::Cast(Box::new(expr),Box::new(parse_type(lexer)));
    }
    expr
}

pub(crate) fn parse_mul_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_cast_expr(lexer);
    loop {
        if lexer.punct('*') {
            expr = Expr::Mul(Box::new(expr),Box::new(parse_cast_expr(lexer)));
        }
        else if lexer.punct('/') {
            expr = Expr::Div(Box::new(expr),Box::new(parse_cast_expr(lexer)));
        }
        else if lexer.punct('%') {
            expr = Expr::Mod(Box::new(expr),Box::new(parse_cast_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_add_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_mul_expr(lexer);
    loop {
        if lexer.punct('+') {
            expr = Expr::Add(Box::new(expr),Box::new(parse_mul_expr(lexer)));
        }
        else if lexer.punct('-') {
            expr = Expr::Sub(Box::new(expr),Box::new(parse_mul_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_shift_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_add_expr(lexer);
    loop {
        if lexer.punct2('<','<') {
            expr = Expr::Shl(Box::new(expr),Box::new(parse_add_expr(lexer)));
        }
        else if lexer.punct2('>','>') {
            expr = Expr::Shr(Box::new(expr),Box::new(parse_add_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_and_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_shift_expr(lexer);
    loop {
        if lexer.punct('&') {
            expr = Expr::And(Box::new(expr),Box::new(parse_shift_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_xor_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_and_expr(lexer);
    loop {
        if lexer.punct('^') {
            expr = Expr::Xor(Box::new(expr),Box::new(parse_and_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_or_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_xor_expr(lexer);
    loop {
        if lexer.punct('|') {
            expr = Expr::Or(Box::new(expr),Box::new(parse_xor_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_comp_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_or_expr(lexer);
    loop {
        if lexer.punct2('=','=') {
            expr = Expr::Eq(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else if lexer.punct2('!','=') {
            expr = Expr::NotEq(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else if lexer.punct('>') {
            expr = Expr::Gt(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else if lexer.punct2('<','=') {
            expr = Expr::NotGt(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else if lexer.punct('<') {
            expr = Expr::Lt(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else if lexer.punct2('>','=') {
            expr = Expr::NotLt(Box::new(expr),Box::new(parse_or_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_logand_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_comp_expr(lexer);
    loop {
        if lexer.punct2('&','&') {
            expr = Expr::LogAnd(Box::new(expr),Box::new(parse_comp_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_logor_expr(lexer: &Lexer) -> Expr {
    let mut expr = parse_logand_expr(lexer);
    loop {
        if lexer.punct2('|','|') {
            expr = Expr::LogOr(Box::new(expr),Box::new(parse_logand_expr(lexer)));
        }
        else {
            break;
        }
    }
    expr
}

pub(crate) fn parse_range_expr(lexer: &Lexer) -> Expr {
    if lexer.punct2('.','.') {
        let with_eq = lexer.punct('=');
        let expr = parse_logor_expr(lexer);
        if with_eq {
            Expr::RangeToIncl(Box::new(expr))
        }
        else {
            Expr::RangeTo(Box::new(expr))
        }
        // TODO: RangeFull
    }
    else {
        let expr = parse_logor_expr(lexer);
        if lexer.punct2('.','.') {
            let with_eq = lexer.punct('=');
            let expr2 = parse_logor_expr(lexer);
            if with_eq {
                Expr::RangeIncl(Box::new(expr),Box::new(expr2))
            }
            else {
                Expr::Range(Box::new(expr),Box::new(expr2))
            }
            // TODO: RangeFrom
        }
        else {
            expr
        }
    }
}

pub(crate) fn parse_assign_expr(lexer: &Lexer) -> Expr {
    let expr = parse_range_expr(lexer);
    if lexer.punct('=') {
        expr = Expr::Assign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('+','=') {
        expr = Expr::AddAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('-','=') {
        expr = Expr::SubAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('*','=') {
        expr = Expr::MulAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('/','=') {
        expr = Expr::DivAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('%','=') {
        expr = Expr::ModAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('&','=') {
        expr = Expr::AndAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('^','=') {
        expr = Expr::XorAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    else if lexer.punct2('|','=') {
        expr = Expr::OrAssign(Box::new(expr),Box::new(parse_range_expr(lexer)))
    }
    // TODO: ShlAssign
    // TODO: ShrAssign
    expr
}

pub(crate) fn parse_block_expr(lexer: &Lexer) -> Expr {
    let sublexer = lexer.group('{').expect("{ expected");
    let mut stats: Vec<Box<Stat>> = Vec::new();
    while !sublexer.done() {
        lexer.punct(';');
        if lexer.ident("let") {
            let pat = Box::new(parse_pat(&sublexer));
            let ty = if sublexer.punct(':') {
                Some(Box::new(parse_type(&sublexer)))
            }
            else {
                None
            };
            let expr = if sublexer.punct('=') {
                Some(Box::new(parse_expr(&sublexer)))
            }
            else {
                None
            };
            lexer.punct(';');
            stats.push(Box::new(Stat::Let(pat,ty,expr)));
        }
        else if let Some(item) = parse_item(&sublexer) {
            stats.push(Box::new(Stat::Item(item)));
        }
        else {
            stats.push(Box::new(Stat::Expr(Box::new(parse_expr(&sublexer)))));
        }
    }
    Expr::Block(stats)
}

pub(crate) fn parse_expr(lexer: &Lexer) -> Expr {
    if lexer.ident("continue") {
        Expr::Continue
    }
    else if lexer.ident("loop") {
        Expr::Loop(Box::new(parse_block_expr(lexer)))
    }
    else if lexer.ident("while") {
        // PredicateLoopExpr = `while` Expr BlockExpr .
        // MatchArmPats = [ `|` ] Pat { `|` Pat } .
        // PredicatePatLoopExpr = `while` `let` MatchArmsPats `=` Expr BlockExpr .
    }
    else if lexer.ident("for") {
        // IteratorLoopExpr = `for` Pat `in` Expr BlockExpr .
    }
    else if lexer.ident("if") {
        // IfExpr = `if` Expr BlockExpr [ `else` BlockExpr | IfExpr | IfLetExpr ] .
        // IfLetExpr = `if` `let` MatchArmPats `=` Expr BlockExpr [ `else` BlockExpr | IfExpr | IfLetExpr ] .
    }
    else if lexer.ident("match") {
        // MatchArm = { OuterAttribute } MatchArmPats [ `if` Expr ] `=>` Expr .
        // MatchExpr = `match` Expr `{` [ { MatchArm `,` } MatchArm [ `,` ] ] `}` .
    }
    else if lexer.ident("break") {
        Expr::Break
    }
    else if lexer.ident("return") {
        Expr::Return
    }
    else {
        parse_assign_expr(lexer)
    }
}
