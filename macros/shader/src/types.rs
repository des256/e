use {
    crate::*,
};

pub(crate) fn parse_type(lexer: &Lexer) -> Type {
    if let Some(sublexer) = lexer.group('(') {
        let mut types: Vec<Box<Type>> = Vec::new();
        while !sublexer.done() {
            types.push(Box::new(parse_type(&sublexer)));
            sublexer.punct(',');
        }
        Type::Tuple(types)
    }
    else if let Some(sublexer) = lexer.group('[') {
        let ty = Box::new(parse_type(&sublexer));
        if sublexer.punct(';') {
            let expr = Box::new(parse_expr(&sublexer));
            Type::Array(ty,expr)
        }
        else {
            Type::Slice(ty)
        }
    }
    else if lexer.punct('_') {
        Type::Inferred
    }
    else if lexer.punct('!') {
        Type::Never
    }
    else if lexer.punct('&') {
        let is_mut = lexer.ident("mut");
        let ty = Box::new(parse_type(lexer));
        Type::Ref(is_mut,ty)
    }
    else if lexer.punct('*') {
        let is_mut = lexer.ident("mut");
        if !is_mut {
            lexer.ident("const");
        }
        let ty = Box::new(parse_type(lexer));
        Type::Pointer(is_mut,ty)
    }
    else if lexer.ident("fn") {
        let sublexer = lexer.group('(').expect("( expected");
        let mut params: Vec<AnonParam> = Vec::new();
        let mut is_var = false;
        while !sublexer.done() {
            params.push(if let Some(ident) = sublexer.any_ident() {
                lexer.punct(':');
                let ty = Box::new(parse_type(&sublexer));
                AnonParam::Param(ident,ty)
            }
            else {
                if sublexer.punct('_') {
                    sublexer.punct(':');
                }
                let ty = Box::new(parse_type(&sublexer));
                AnonParam::Anon(ty)
            });
            sublexer.punct(',');
            if sublexer.punct('.') {
                sublexer.punct('.');
                sublexer.punct('.');
                is_var = true;
            }
        }
        let return_ty = if lexer.punct('-') {
            lexer.punct('>');
            Some(Box::new(parse_type(lexer)))
        }
        else {
            None
        };
        Type::Function(params,is_var,return_ty)
    }
    else {
        Type::Segs(parse_segs(lexer))
    }
}
