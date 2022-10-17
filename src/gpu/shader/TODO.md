# TODO

- we had to use RefCell constructions everywhere to make sure the Rc<>s still point to the right objects, maybe do this differently? - only use strings as indices in hashmaps
- add Const to Pat
- new stage: remove aliases

## DESTRUCTURE

Transform pattern matching expressions into normal ones, only leaving creation of local variables.

- destructuring: patterns are recursively destructured towards a scrutinee expression for each of the expressions and statements where the patterns are used
- testing: patterns are also translated to boolean expressions to test the match
- Expr::IfLet becomes Expr::If
- Expr::WhileLet becomes Expr::While
- Expr::For becomes Expr::While
- Expr::Match becomes a chain of Expr::Ifs with else expressions
- Stat::Let become local variable declaration

## ANONTUPLELESS DETUPLIFY

Convert all tuples to named structs.

- all tuples in the module are converted to structs
- Type::Tuple becomes Type::Struct with a directly converted tuple
- Pat::Tuple becomes Pat::Struct with a directly converted tuple
- Expr::Tuple becomes Expr::Struct
- Expr::TupleIndex becomes Expr::Field

## ANONTUPLEFULL DETUPLIFY

Convert all anonymous tuples to named structs.

Needs symbols to be resolved in order to match Expr::AnonTuple with expected struct.

- Type::AnonTuple becomes Type::Struct with a matching struct from an anonymous tuple struct list
- Pat::AnonTuple does not need to be transformed
- Expr::AnonTuple becomes Expr::Struct with a matching struct from an anonymous tuple struct list

## DISENUMIFY

Convert all enums to named structs.

This might be doable before symbol resolution?

- all enums in the module are converted to structs, and an index is created to map each enum to a struct; the first field of the struct is the discriminant ID
- TODO

## RESOLVE SYMBOLS

Convert all unknown symbols to reference params, locals, structs, tuples, enums, functions, consts and aliases.

- Type::UnknownIdent becomes Type::Tuple, Type::Struct, Type::Enum or Type::Alias
- Pat::UnknownTuple becomes Pat::Tuple
- Pat::UnknownStruct becomes Pat::Struct
- Pat::UnknownVariant becomes Pat::Variant
- Expr::UnknownIdent becomes Expr::Const, Expr::Local or Expr::Param
- Expr::UnknownTupleOrCall becomes Expr::Tuple or Expr::Call
- Expr::UnknownStruct becomes Expr::Struct
- Expr::UnknownVariant becomes Expr::Variant
- Expr::UnknownMethod becomes Expr::Method
- Expr::UnknownField becomes Expr::Field
- Expr::UnknownTupleIndex becomes Expr::TupleIndex

## RESOLVE ALIASES

Convert all aliases to their actual types.

- Type::Alias becomes whatever type is at the end of the alias chain

# SO?

1. resolve aliases, disenumify, anontupleless detuplify, destructure
2. resolve symbols
3. anontuplefull detuplify
4. translate to local language AST
5. optimize
6. render

# STEP1

Convert all aliases to their actual types.
Convert all enums to named structs.
Convert all tuples to named structs.
Transform pattern matching expressions into normal ones, only leaving creation of local variables.

Transformations that don't need anything else:

- TODO disenumify
- Pat::Tuple becomes Pat::Struct with a directly converted tuple
- all enums in the module are converted to structs, and an index is created to map each enum to a struct; the first field of the struct is the discriminant ID
- all tuples in the module are converted to structs
- Type::Alias becomes whatever type is at the end of the alias chain
- Type::Tuple becomes Type::Struct with a directly converted tuple
- Expr::Tuple becomes Expr::Struct
- Expr::TupleIndex becomes Expr::Field

Transformations that need to access more elaborate type info:

- destructuring: patterns are recursively destructured towards a scrutinee expression for each of the expressions and statements where the patterns are used
- testing: patterns are also translated to boolean expressions to test the match
- Expr::IfLet becomes Expr::If
- Expr::WhileLet becomes Expr::While
- Expr::For becomes Expr::While
- Expr::Match becomes a chain of Expr::Ifs with else expressions
- Stat::Let become local variable declaration

# STEP2

Convert all unknown symbols to reference params, locals, structs, tuples, enums, functions, consts and aliases.

- Type::UnknownIdent becomes Type::Tuple, Type::Struct, Type::Enum or Type::Alias
- Pat::UnknownTuple becomes Pat::Tuple
- Pat::UnknownStruct becomes Pat::Struct
- Pat::UnknownVariant becomes Pat::Variant
- Expr::UnknownIdent becomes Expr::Const, Expr::Local or Expr::Param
- Expr::UnknownTupleOrCall becomes Expr::Tuple or Expr::Call
- Expr::UnknownStruct becomes Expr::Struct
- Expr::UnknownVariant becomes Expr::Variant
- Expr::UnknownMethod becomes Expr::Method
- Expr::UnknownField becomes Expr::Field
- Expr::UnknownTupleIndex becomes Expr::TupleIndex

# STEP3

Convert all anonymous tuples to named structs.

- Type::AnonTuple becomes Type::Struct with a matching struct from an anonymous tuple struct list
- Pat::AnonTuple does not need to be transformed
- Expr::AnonTuple becomes Expr::Struct with a matching struct from an anonymous tuple struct list

=====

# AGAIN: ALL TRANSFORMATIONS

## Decode UnknownTupleOrCall

- Expr::UnknownTupleOrCall => Expr::Tuple or Expr::Call, depending on whether or not the function ident exists

## Build Matcher

Build boolean expression from pat and scrut.

- Pat::Boolean(value) => if value { scrut } else { !scrut }
- Pat::Integer(value) => scrut == value
- Pat::Float(value) => scrut == value
- Pat::AnonTuple([pat]) => assuming scrut is of type Type::AnonTuple, && matchers for each pat
- Pat::Array([pat]) => assuming scrut is of type Type::Array, && matches for each pat
- Pat::Range(lo,hi) => (scrut >= lo) && (scrut < hi), where lo and hi are Pat::Integer or Pat::Float
- Pat::UnknownTuple(_,[pat]) => && matches for each pat
- Pat::UnknownStruct(_,[identpat]) => && matches for each pat that isn't _ or ..
- Pat::UnknownVariant(_,variant) => (scrut.discriminant() == id) && matches for each pat that isn't _ or ..
- Pat::Tuple(_,[pat]) => assuming scrut is of type Type::Tuple, && matches for each pat that isn't _ or ..
- Pat::Struct(_,[indexpat]) => assuming scrut is of type Type::Struct, && matches for each pat that isn't _ or ..
- Pat::Variant(_,variant) => (scrut.discriminant() == id) && matches for each pat that isn't _ or ..

## Destructure

Add to Vec<Stat> from pat and scrut.

- Pat::UnknownIdent => let ident: found type = scrut;
- Pat::Array([pat]) => destructure recursively from pat and scrut[i]
- Pat::AnonTuple([pat]) => destrcture recursively from pat and scrut.i
- Pat::Struct(_,[field]) => destructure recursively from field.pat and scrut.ident or let ident: field type = scrut.ident;
- Pat::Tuple(_,[pat]) => destructure recursively from pat and scrut.index or let ident: type = scrut.index;
- Pat::Variant(_,variant) => destructure recursively
- Pat::UnknownStruct(_,[field]) => destructure recursively from field.pat and scrut.ident or let ident: field type = scrut.ident;
- Pat::UnknownTuple(_,[pat]) => destructure recursively from pat and scrut.index or let ident: type = scrut.index;
- Pat::UnknownVariant(_,variant) => destructure recursively
- Expr::IfLet => { let scrut = ...; if matcher { destructured pats; block.stats; block.expr } else { ... } }
- Expr::WhileLet => { let scrut = ...; while matcher { destructured pats; block.stats; block.expr } }
- Expr::For => ...
- Expr::Match => { let scrut = ...; if matcher { destructured pats; block.stats; block.expr } else ... }
- Stat::Let => destructure recursively, this will generate a Stat::Local

## Detuplify

- Type::UnknownTuple => Type::UnknownStruct
- Type::Tuple => Type::Struct
- Pat::UnknownTuple => Pat::UnknownStruct
- Pat::Tuple => Pat::Struct
- Expr::UnknownTuple => Expr::UnknownStruct
- Expr::UnknownTupleIndex => Expr::UnknownField

## Detuplify AnonTuple

- Type::AnonTuple => Type::Struct
- Pat::AnonTuple => Pat::Struct
- Expr::AnonTuple => Expr::Struct

## Disenumify

- Type::Enum => Type::Struct
- Pat::UnknownVariant => Pat::UnknownStruct
- Pat::Variant => Pat::Struct
- Expr::UnknownVariant => Expr::UnknownStruct
- Expr::Variant => Expr::Struct

## Resolve Symbols

- Type::UnknownIdent => Type::Tuple, Type::Struct, Type::Enum or Type::Alias
- Pat::UnknownTuple => Pat::Tuple
- Pat::UnknownStruct => Pat::Struct
- Pat::UnknownVariant => Pat::Variant
- Expr::UnknownIdent => Expr::Param, Expr::Local or Expr::Const
- Expr::UnknownTuple => Expr::Tuple
- Expr::UnknownCall => Expr::Call
- Expr::UnknownStruct => Expr::Struct
- Expr::UnknownVariant => Expr::Variant
- Expr::UnknownMethod => Expr::Method
- Expr::UnknownField => Expr::Field
- Expr::UnknownTupleIndex => Expr::TupleIndex

# QUESTION

Can we do everything at the same time, except maybe remove anonymous tuples?

So that means:

- Type::UnknownIdent => Type::Tuple, Type::Struct, Type::Enum or Type::Alias
- Type::Tuple => Type::Struct
- Type::Enum => Type::Struct

- Pat::UnknownTuple => Pat::Struct
- Pat::UnknownStruct => Pat::Struct
- Pat::UnknownVariant => Pat::Struct
- Pat::Tuple => Pat::Struct
- Pat::Variant => Pat::Struct

- Expr::IfLet => { let scrut = ...; if matcher { destructured pats; block.stats; block.expr } else { ... } }
- Expr::WhileLet => { let scrut = ...; while matcher { destructured pats; block.stats; block.expr } }
- Expr::For => ...
- Expr::Match => { let scrut = ...; if matcher { destructured pats; block.stats; block.expr } else ... }
- Expr::UnknownIdent => Expr::Param, Expr::Local, Expr::Const
- Expr::UnknownTupleOrCall => Expr::Tuple, Expr::Call
- Expr::UnknownStruct => Expr::Struct
- Expr::UnknownVariant => Expr::Field
- Expr::UnknownMethod => Expr::Method
- Expr::UnknownField => Expr::Field
- Expr::UnknownTupleIndex => Expr::Field
- Expr::Variant => Expr::Field

- Stat::Let => destructure recursively, this will generate a Stat::Local
