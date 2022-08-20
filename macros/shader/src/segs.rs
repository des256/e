use {
    crate::*,
};

// Segment = IDENT | GenericArgs | TypeFn .
// GenericArgs = `<` { Type [ `=` Type ] [ `as` Segments ] [ `,` ] } `>` .
// TypeFn = `(` { Type [ `,` ] } `)` [ `->` Type ] .
// Segments = [ `::` ] Segment { [ `::` ] Segment } .
pub(crate) fn parse_seg(lexer: &mut Lexer) -> Seg {
    if let Some(ident) = lexer.any_ident() {
        Seg::Ident(ident)
    }
    else if lexer.punct('<') {
        let mut genargs: Vec<GenArg> = Vec::new();
        while !lexer.punct('>') {
            let ty = Box::new(parse_type(lexer));
            let binding = if lexer.punct('=') {
                Some(Box::new(parse_type(lexer)))
            }
            else {
                None
            };
            let as_segs = if lexer.ident("as") {
                parse_segs(lexer)
            }
            else {
                Vec::new()
            };
            lexer.punct(',');
            genargs.push(GenArg { ty,binding,as_segs, });
        }
        Seg::Generic(genargs)
    }
    else if let Some(mut sublexer) = lexer.group('(') {
        let mut types: Vec<Box<Type>> = Vec::new();
        while !sublexer.done() {
            types.push(Box::new(parse_type(&mut sublexer)));
            lexer.punct(',');
        }
        let return_ty = if lexer.punct2('-','>') {
            Some(Box::new(parse_type(&mut sublexer)))
        }
        else {
            None
        };
        Seg::Function(types,return_ty)
    }
    else {
        panic!("invalid path");
    }
}

// Segments = [ `::` ] Segment { [ `::` ] Segment } .
pub(crate) fn parse_segs(lexer: &mut Lexer) -> Vec<Seg> {
    lexer.punct2(':',':');
    let mut segs: Vec<Seg> = Vec::new();
    segs.push(parse_seg(lexer));
    while lexer.punct2(':',':') {
        segs.push(parse_seg(lexer));
    }
    segs
}
