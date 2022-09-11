# FROM REDUCED RUST TO TARGET SHADER LANGUAGE

Reduced Rust has three main concepts that have no counterpart in any of the target shader languages.

- Pattern Matching
- Tuples
- Enums

## PATTERN MATCHING

Pattern matching happens in Stat::Let, as well as Expr::For, Expr::IfLet, Expr::WhileLet and of course Expr::Match. We're looking for an exhaustive way to convert the pattern matching into individual C-like instructions.

### Stat::Let

Rust:

let a = 4;
let a = (b,4);
let a = Foobar { a: 4,b: 5, };
let Foobar { a,b: _, } = func();
let (a,b,c) = func();
let (a,..) = (c,d);

C-like:

let a = 4;
let a = Tuple0 { _0: b,_1: 4, };
let a = Foobar { a: 4,b: 5, };
let t = func();
let a = t.a;
let t = func();
let a = t._0;
let b = t._1;
let c = t._2;
let t = AnonTuple0 { _0: c,_1: d, };
let a = t._0;

## TUPLES

Named tuples are just structs without field identifiers, so they are already taken care of in the compile-time part. The only thing left are anonymous tuple expressions and anonymous tuple patterns.