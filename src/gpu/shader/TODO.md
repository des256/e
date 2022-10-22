# TODO

- we had to use RefCell constructions everywhere to make sure the Rc<>s still point to the right objects, maybe do this differently? - only use strings as indices in hashmaps
- add Const to Pat
- new stage: remove aliases

# WHAT NEEDS TO HAPPEN?

## Destructure Patterns

this generates Stat::Local declarations and removes all patterns, if let, while let and match expressions

there are two essential operations: building a boolean expression and destructuring a pattern

make_pat_bool(pat,scrut) -> Option<Expr>
    Pat::Wildcard, Pat::Rest, Pat::UnknownIdent not const => None
    Pat::Boolean(value) => if *value { scrut } else { !scrut }
    Pat::Integer(value) => scrut == value
    Pat::Float(value) => scrut == value
    Pat::Const(ident) => scrut == ident
    Pat::Struct(ident,fields) => for each field containing a pat, && together make_pat_bool(the field's pat,scrut.field name)
    Pat::Tuple(ident,pats) => for each pat, && together make_pat_bool(pat,scrut.index)
    Pat::Array(pats) => for each pat, && together make_pat_bool(pat,scrut[index])
    Pat::AnonTuple(pats) => for each pat, && together make_pat_bool(pat,scrut.index)
    Pat::Variant(enum_ident,Naked(ident)) => discr == ident
    Pat::Variant(enum_ident,Tuple(ident,pats)) => discr == ident && for each pat, && together make_pat_bool(pat,scrut.index)
    Pat::Variant(enum_ident,Struct(ident,pats)) => discr == ident && for each field, && together fields with pats as make_pat_bool(pat,scrut.index)
    Pat::Range(lo,hi) => (scrut >= lo) && (scrut < hi)

make_pats_bool(pats,scrut) => each pat individually || together

destructure_pat(pat,scrut) => Vec<Stat>
    Pat::Wildcard, Pat::Rest, Pat::Boolean, Pat::Integer, Pat::Float, Pat::Const, Pat::Range => no contribution
    Pat::Ident(ident) => Stat::Local(ident,scrut)
    Pat::Struct(_,fields) => for each field:
        Ident(ident) => Stat::Local(ident,scrut.ident)
        IdentPat(ident,pat) => destructure_pat(pat,scrut.ident)
    Pat::Tuple(_,pats) => for each pat: destructure_pat(pat,scrut.index)
    Pat::Array(pats) => for each pat: destructure_pat(pat,scrut[index])
    Pat::AnonTuple(pats) => for each pat: destructure_pat(pat,scrut.index)
    Pat::Variant(_,Naked(ident)) => no contribution
    Pat::Variant(_,Tuple(ident,pats)) => for each pat: destructure_pat(pat,scrut.index)
    Pat::Variant(_,Struct(ident,fields)) => for each field with pat: destructure_pat(pat,scrut.ident)

destructure_pats(pats,scrut) => Vec<Stat>
    destructure each pattern separately

Expr::IfLet(pats,scrut,block,else_expr):
    prepend block with destructuring locals for all pats
    output Expr::Block:
        Stat::Local to capture original scrut
        Expr::If(boolean expression from pats and scrut,block,else_expr)

Expr::WhileLet(pats,scrut,block):
    prepend block with destructuring locals for all pats
    output Expr::Block:
        Stat::Local to capture original scrut
        Expr::While(boolean expression from pats and scrut,block)

Expr::Match(scrut,arms):
    create new block with Stat::local to capture original scrut
    for each arm, build Expr::If(boolean expression from arm.pats and scrut,block prepended with destructuring locals for all arm.pats), one arm has _ with the final else_expr
    concatenate all arms into if-else chain

## Resolve Symbols

resolve all symbols, so they can be referenced later:

Type::UnknownIdent(ident):
    if ident in tuple maps: Type::Tuple(ident)
    else if ident in struct maps: Type::Struct(ident)
    else if ident in enum maps: Type::Enum(ident)
    else if ident in alias maps: Type::Alias(ident)
    else panic: unknown identifier

Expr::UnknownIdent(ident):
    if ident in param map: Expr::Param(ident)
    else if ident in local map: Expr::Local(ident)
    else if ident in const maps: Expr::Const(ident)
    else panic: unknown identifier

Expr::TupleOrCall(ident,exprs):
    if ident in tuple maps: Expr::Tuple(ident,exprs)
    else if ident in function maps: Expr::Call(ident,exprs)
    else panic: unknown identifier

## Convert Named Tuples

named tuples should become structs:

add new structs for each tuple in the context

Type::Tuple(ident): Type::Struct(ident)

Expr::Tuple(ident,exprs): Expr::Struct(ident,fields)

Expr::TupleIndex(expr,index): Expr::Field(expr,field)

## Convert Anonymous Tuples

anonymous tuples should become structs:

in order to properly determine the types, expr.find_type needs to work, and/or use a form of passing an expected type through the tree

Type::AnonTuple(types): add struct if not already exists, Type::Struct(ident)

Expr::AnonTuple(exprs): add struct if not already exists, Type::Struct(ident,exprs)

## Eliminate Named Aliases

named aliases should map to their types:

Type::Alias(ident): follow chain and replace with ultimate type

## Convert Enums

enums should become structs:

for each enum, calculate how many fields are needed and how each variant maps its components

Type::Enum(ident) => Type::Struct

Expr::Variant(enum_ident,variant) => map to struct with indices

Expr::Discriminant(enum_ident) => Expr::Field

Expr::Destructure(enum_ident,variant_index,index) => Expr::Field

# COMBINING PASSES

Destructure Patterns needs to go first and separately

Resolve Symbols, Convert Named and Anonymous Tuples, Eliminate Named Aliases and Convert Enums can go together in a pass where requested types and context are passed down into the tree
