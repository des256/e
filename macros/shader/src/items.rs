use {
    crate::*,
};

// Generics = `<` [ IDENT { `,` IDENT } [ `,` ] ] `>` .
pub(crate) fn parse_generics(lexer: &mut Lexer) -> Vec<String> {
    // after '<'
    let mut idents: Vec<String> = Vec::new();
    while !lexer.punct('>') {
        idents.push(lexer.any_ident().expect("identifier expected"));
        lexer.punct(',');
    }
    idents
}

// Module = `mod` IDENT `;` | ( `{` { Item } `}` ) .
pub(crate) fn parse_mod(lexer: &mut Lexer) -> Item {
    // after "mod"
    let ident = lexer.any_ident().expect("identifier expected");
    let mut items: Vec<Item> = Vec::new();
    if let Some(mut sublexer) = lexer.group('{') {
        while !sublexer.done() {
            items.push(parse_item(&mut sublexer).unwrap());
        }
    }
    else {
        lexer.punct(';');
    }
    Item::Module(ident,items)
}

// FunctionParam = Pat `:` Type .
// FunctionParameters = FunctionParam { `,` FunctionParam } [ `,` ] .
// Function = `fn` IDENT [ Generics ] `(` [ FunctionParameters ] `)` [ `->` Type ] BlockExpr .
pub(crate) fn parse_fn(lexer: &mut Lexer) -> Item {
    // after "fn"
    let ident = lexer.any_ident().expect("identifier expected");
    let generics = if lexer.punct('<') {
        parse_generics(lexer)
    }
    else {
        Vec::new()
    };
    let mut sublexer = lexer.group('(').expect("( expected");
    let mut params: Vec<Param> = Vec::new();
    while !sublexer.done() {
        let pat = parse_pat(&mut sublexer);
        sublexer.punct(':');
        let ty = Box::new(parse_type(&mut sublexer));
        params.push(Param { pat,ty, });
        sublexer.punct(',');
    }
    let return_ty = if lexer.punct2('-','>') {
        Some(Box::new(parse_type(lexer)))
    }
    else {
        None
    };
    let stats = parse_block_stats(lexer);
    Item::Function(ident,generics,params,return_ty,stats)
}

// Alias = `type` IDENT [ Generics ] `=` Type `;` .
pub(crate) fn parse_alias(lexer: &mut Lexer) -> Item {
    // after "type"
    let ident = lexer.any_ident().expect("identifier expected");
    let generics = if lexer.punct('<') {
        parse_generics(lexer)
    }
    else {
        Vec::new()
    };
    lexer.punct('=');
    let ty = Box::new(parse_type(lexer));
    lexer.punct(';');
    Item::Alias(ident,generics,ty)
}

// StructField = IDENT `:` Type .
// StructFields = StructField { `,` StructField } [ `,` ] .
// StructStruct = `struct` IDENT [ Generics ] ( `{` [ StructFields ] `}` ) | `;` .
// TupleFields = Type { `,` Type } [ `,` ] .
// TupleStruct = `struct` IDENT [ Generics ] `(` [ TupleFields ] `)` `;` .
pub(crate) fn parse_struct_or_tuple(lexer: &mut Lexer) -> Item {
    // after "struct"
    let ident = lexer.any_ident().expect("identifier expected");
    let generics = if lexer.punct('<') {
        parse_generics(lexer)
    }
    else {
        Vec::new()
    };
    if let Some(mut sublexer) = lexer.group('{') {
        let mut fields: Vec<Field> = Vec::new();
        while !sublexer.done() {
            let ident = sublexer.any_ident().expect("identifier expected");
            sublexer.punct(':');
            let ty = Box::new(parse_type(&mut sublexer));
            fields.push(Field { ident,ty, });
            sublexer.punct(',');
        }
        Item::Struct(ident,generics,fields)
    }
    else if let Some(mut sublexer) = lexer.group('(') {
        let mut types: Vec<Box<Type>> = Vec::new();
        while !sublexer.done() {
            types.push(Box::new(parse_type(&mut sublexer)));
            sublexer.punct(',');
        }
        Item::Tuple(ident,generics,types)
    }
    else {
        lexer.punct(';');
        Item::Struct(ident,generics,Vec::new())
    }
}

// TupleFields = Type { `,` Type } [ `,` ] .
// TupleStruct = `struct` IDENT [ Generics ] `(` [ TupleFields ] `)` `;` .
// EnumItemTuple = `(` [ TupleFields ] `)` .
// EnumItemStruct = `{` [ StructFields ] `}` .
// EnumItemDiscriminant = `=` Expr .
// EnumItem = IDENT [ EnumItemTuple | EnumItemStruct | EnumItemDiscriminant ] .
// EnumItems = EnumItem { `,` EnumItem } [ `,` ] .
// Enum = `enum` IDENT [ Generics ] `{` [ EnumItems ] `}` .
pub(crate) fn parse_enum(lexer: &mut Lexer) -> Item {
    // after "enum"
    let ident = lexer.any_ident().expect("identifier expected");
    let generics = if lexer.punct('<') {
        parse_generics(lexer)
    }
    else {
        Vec::new()
    };
    let mut sublexer = lexer.group('{').expect("{ expected");
    let mut variants: Vec<Variant> = Vec::new();
    while !sublexer.done() {
        let ident = sublexer.any_ident().expect("identifier expected");
        if let Some(mut subsublexer) = sublexer.group('{') {
            let mut fields: Vec<Field> = Vec::new();
            while !subsublexer.done() {
                let ident = subsublexer.any_ident().expect("identifier expected");
                subsublexer.punct(':');
                let ty = Box::new(parse_type(&mut subsublexer));
                fields.push(Field { ident,ty, });
                subsublexer.punct(',');
            }
            variants.push(Variant::Struct(ident,fields));
        }
        else if let Some(mut subsublexer) = sublexer.group('(') {
            let mut types: Vec<Box<Type>> = Vec::new();
            while !subsublexer.done() {
                types.push(Box::new(parse_type(&mut subsublexer)));
                subsublexer.punct(',');
            }
            variants.push(Variant::Tuple(ident,types));
        }
        else if sublexer.punct('=') {
            let expr = Box::new(parse_expr(&mut sublexer));
            variants.push(Variant::Discr(ident,expr));
        }
        else {
            variants.push(Variant::Naked(ident));
        }
        sublexer.punct(',');
    }
    Item::Enum(ident,generics,variants)
}

// Union = `union` IDENT [ Generics ] `{` StructFields `}` .
pub(crate) fn parse_union(lexer: &mut Lexer) -> Item {
    // after "union"
    let ident = lexer.any_ident().expect("identifier expected");
    let generics = if lexer.punct('<') {
        parse_generics(lexer)
    }
    else {
        Vec::new()
    };
    let mut sublexer = lexer.group('{').expect("{ expected");
    let mut fields: Vec<Field> = Vec::new();
    while !sublexer.done() {
        let ident = sublexer.any_ident().expect("identifier expected");
        sublexer.punct(':');
        let ty = Box::new(parse_type(&mut sublexer));
        fields.push(Field { ident,ty, });
        sublexer.punct(',');
    }
    Item::Union(ident,generics,fields)
}

// Constant = `const` IDENT | `_` `:` Type `=` Expr `;` .
pub(crate) fn parse_const(lexer: &mut Lexer) -> Item {
    // after "const"
    let ident = if let Some(ident) = lexer.any_ident() {
        Some(ident)
    }
    else {
        lexer.punct('_');
        None
    };
    lexer.punct(':');
    let ty = Box::new(parse_type(lexer));
    lexer.punct('=');
    let expr = Box::new(parse_expr(lexer));
    lexer.punct(';');
    Item::Const(ident,ty,expr)
}

// Static = `static` [ `mut` ] IDENT `:` Type `=` Expr `;` .
pub(crate) fn parse_static(lexer: &mut Lexer) -> Item {
    // after "static"
    let is_mut = lexer.ident("mut");
    let ident = lexer.any_ident().expect("identifier expected");
    lexer.punct(':');
    let ty = Box::new(parse_type(lexer));
    lexer.punct('=');
    let expr = Box::new(parse_expr(lexer));
    Item::Static(is_mut,ident,ty,expr)
}

// Item = Module | Function | Alias | StructStruct | TupleStruct | Enum | Union | Constant | Static .
pub(crate) fn parse_item(lexer: &mut Lexer) -> Option<Item> {
    if lexer.ident("mod") {
        Some(parse_mod(lexer))
    }
    else if lexer.ident("fn") {
        Some(parse_fn(lexer))
    }
    else if lexer.ident("type") {
        Some(parse_alias(lexer))
    }
    else if lexer.ident("struct") {
        Some(parse_struct_or_tuple(lexer))
    }
    else if lexer.ident("enum") {
        Some(parse_enum(lexer))
    }
    else if lexer.ident("union") {
        Some(parse_union(lexer))
    }
    else if lexer.ident("const") {
        Some(parse_const(lexer))
    }
    else if lexer.ident("static") {
        Some(parse_static(lexer))
    }
    else {
        None
    }
}
