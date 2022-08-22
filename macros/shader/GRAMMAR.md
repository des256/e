# Reduced Rust Grammar

Ultimately we want to describe shaders in Rust, so this grammar is without:

- macros
- module hierarchy
- use declarations
- lifetimes
- traits
- closures
- where clauses
- methods and implementations
- async/await
- labels
- type aliasing, only fixed type names
- generics
- references
- pointers
- function types

And ignore:

- InnerAttribute = `#` `!` `[` Attr `]` .
- OuterAttribute = `#` `[` Attr `]` .
- Visibility = `pub` [ `(` `crate` | `self` | `super` | ( `in` SimplePath ) `)` ] .
- unsafe keyword

Also, the Rust compiler itself is compiling the shader too, so this parser will not have to fix errors, only generate a parse tree.

Are we going to support structures from elsewhere to be accessible here? No, due to limitations of rustc.

How can we specify input and output formats from the shaders then? By using fixed types in tuples.

The fixed types are (and should be the same outside the shader):

    bool
    u8
    u16
    u32
    u64
    i8
    i16
    i32
    i64
    f16
    f32
    f64
    Vec2<bool>
    Vec2<u8>
    Vec2<u16>
    Vec2<u32>
    Vec2<u64>
    Vec2<i8>
    Vec2<i16>
    Vec2<i32>
    Vec2<i64>
    Vec2<f16>
    Vec2<f32>
    Vec2<f64>
    Vec3<bool>
    Vec3<u8>
    Vec3<u16>
    Vec3<u32>
    Vec3<u64>
    Vec3<i8>
    Vec3<i16>
    Vec3<i32>
    Vec3<i64>
    Vec3<f16>
    Vec3<f32>
    Vec3<f64>
    Vec4<bool>
    Vec4<u8>
    Vec4<u16>
    Vec4<u32>
    Vec4<u64>
    Vec4<i8>
    Vec4<i16>
    Vec4<i32>
    Vec4<i64>
    Vec4<f16>
    Vec4<f32>
    Vec4<f64>
    Color<u8>
    Color<u16>
    Color<u32>
    Color<u64>
    Color<f16>
    Color<f32>
    Color<f64>
    Mat2<f32>
    Mat2<f64>
    Mat2x3<f32>
    Mat2x3<f64>
    Mat2x4<f32>
    Mat2x4<f64>
    Mat3x2<f32>
    Mat3x2<f64>
    Mat3<f32>
    Mat3<f64>
    Mat3x4<f32>
    Mat3x4<f64>
    Mat4x2<f32>
    Mat4x2<f64>
    Mat4x3<f32>
    Mat4x3<f64>
    Mat4<f32>
    Mat4<f64>

How to deal with Uniform structures?

How to deal with Uniform arrays?

Perhaps it is possible to supply types in the macro attribute that implement Vertex, Uniform, Varying or Fragment traits. And then bind these as layouts in the resulting shader code (GLSL, MSL, etc.), and pass through down to the graphics pipeline as well.

Specify vertex_shader and fragment_shader separately, and have fixed parameters:

vertex_shader: [Uniform,...],Vertex,Varying
fragment_shader: [Uniform,...],Varying,Fragment

possible to implement Uniform on arrays as well as structs?

enum Literal {
    Bool(bool),
    Char(char),
    String(String),
    Byte(u8),
    ByteString(Vec<u8>),
    Integer(u64),
    Float(f64),
}

We implement stuff in stages. Structs later, enums later or maybe not, ranges later.

## Pats

enum PatField {
    Ident(String),
    Index(usize,Pat),
    Field(String,Pat),  // later
}

enum Pat {
    Wildcard,
    Rest,
    Literal(Literal),
    Tuple(Vec<Pat>),
    Slice(Vec<Pat>),
    Struct(String,Vec<PatField>),  // later
    TupleStruct(String,Vec<Pat>),
    Ident(String),
    Range(Pat,Pat),  // later
}

### Grammar

StructPatField =
    ( TUPLE_INDEX `:` Pat ) |
    ( IDENT `:` Pat ) |
    ( [ `ref` ] [ `mut` ] IDENT ) .

PrimaryPat =
    `_` |
    `..` |
    `true` |
    `false` |
    `'C'` |
    `b'C'` |
    `"STRING"` |
    `b"STRING"` |
    `r#"STRING"#` |
    `br#"STRING"#` |
    ( [ `-` ] DEC_LITERAL | OCT_LITERAL | HEX_LITERAL | BIN_LITERAL | FLOAT_LITERAL ) .
    ( `(` [
        `..` |
        ( Pat { `,` Pat } [ `,` ] )
    ] `)` ) |
    ( `[` [ Pat { `,` Pat } [ `,` ] ] `]` ) |
    ( IDENT [
        ( `@` Pat ) |
        ( `{` [ StructPatField { `,` StructPatField } ] [ `,` `..` ] `}` ) |  // later
        ( `(` [ Pat { `,` Pat } [ `,` ] ] `)` )
    ] ) |

Pat = PrimaryPat [ `..=`
    `'C'` |
    `b'C'` |
    IDENT |
    ( [ `-` ] DEC_LITERAL | OCT_LITERAL | HEX_LITERAL | BIN_LITERAL | FLOAT_LITERAL )
] .

## Types

enum Type {
    Tuple(Vec<Type>),
    Array(Type,Expr),
    Ident(String),
    Never,
    Inferred,
}

### Grammar

Type = 
    ( `(` [ Type { `,` Type } [ `,` ] ] `)` ) |
    ( `[` Type [ `;` Expr ] `]` ) |
    IDENT |
    `!` |
    `_` .

## Exprs

enum ExprField {  // later
    LiteralExpr(Literal,Expr),
    IdentExpr(String,Expr),
    Ident(String),
}

enum Expr {
    Literal(Literal),
    Ident(String),
    Array(Vec<Expr>),
    Cloned(Expr,Expr),
    Tuple(Vec<Expr>),
    EnumStruct(Expr,Vec<ExprField>,Option<Expr>),  // later
    EnumTuple(Expr,Vec<Expr>),
    Enum(Expr),
    TupleIndex(Expr,usize),
    Field(Expr,String),
    Index(Expr,Expr),
    Call(Expr,Vec<Expr>),
    Error(Expr),
    Neg(Expr),
    Not(Expr),
    Cast(Expr,Type),
    Mul(Expr,Expr),
    Div(Expr,Expr),
    Mod(Expr,Expr),
    Add(Expr,Expr),
    Sub(Expr,Expr),
    Shl(Expr,Expr),
    Shr(Expr,Expr),
    And(Expr,Expr),
    Xor(Expr,Expr),
    Or(Expr,Expr),
    Eq(Expr,Expr),
    NotEq(Expr,Expr),
    Gt(Expr,Expr),
    NotGt(Expr,Expr),
    Lt(Expr,Expr),
    NotLt(Expr,Expr),
    LogAnd(Expr,Expr),
    LogOr(Expr,Expr),
    RangeFromTo(Expr,Expr),  // later
    RangeFromToIncl(Expr,Expr),  // later
    RangeFrom(Expr),  // later
    RangeTo(Expr),  // later
    RangeToIncl(Expr),  // later
    Range,  // later
    Assign(Expr,Expr),
    AddAssign(Expr,Expr),
    SubAssign(Expr,Expr),
    MulAssign(Expr,Expr),
    DivAssign(Expr,Expr),
    ModAssign(Expr,Expr),
    AndAssign(Expr,Expr),
    XorAssign(Expr,Expr),
    OrAssign(Expr,Expr),
    Block(Vec<Stat>),
    Continue,
    Break(Option<Expr>),
    Return(Option<Expr>),
    Loop(Vec<Stat>),
    For(Pat,Expr,Vec<Stat>),
    IfLet(Vec<Pat>,Expr,Vec<Stat>,Option<Expr>),
    If(Expr,Vec<Stat>,Option<Expr>),
    WhileLet(Vec<Pat>,Expr,Vec<Stat>),
    While(Expr,Vec<Stat>),
    Match(Expr,Vec<Arm>),
}

### Grammar

LiteralExpr =
    `true` |
    `false` |
    `'C'` |
    `b'C'` |
    `"STRING"` |
    `b"STRING"` |
    `r#"STRING"#` |
    `br#"STRING"#` |
    ( DEC_LITERAL | OCT_LITERAL | HEX_LITERAL | BIN_LITERAL | FLOAT_LITERAL ) .

PrimaryExpr =
    LiteralExpr |
    IDENT .

ExprField =  // later
    IDENT |
    ( IDENT | TUPLE_INDEX `:` Expr ) .

StructExprStruct = PrimaryExpr `{` [ ( ExprField { `,` ExprField } [ `,` `..` Expr [ `,` ] ] ) | `..` Expr ] `}` .  // later

EnumExprStruct = PrimaryExpr `{` [ ExprField { `,` ExprField } [ `,` ] ] `}` .  // later

ExprTuple = PrimaryExpr `(` [ Expr { `,` Expr } [ `,` ] ] `)` .

DirectExpr =
    ( `[` [ Expr { `,` Expr } [ `,` ] [ `;` Expr ] ] `]` ) |
    ( `(` [ Expr { `,` Expr } [ `,` ] ] `)` ) |
    StructExprStruct |  // later
    EnumExprStruct |  // later
    ExprTuple |
    PrimaryExpr .

TupleIndexingExpr = DirectExpr { `.` TUPLE_INDEX } .

FieldExpr = TupleIndexingExpr { `.` IDENT } .  // only for .x, .y, .r, etc.

IndexExpr = FieldExpr { `[` Expr `]` } .

CallParams = Expr { `,` Expr } [ `,` ] .
CallExpr = IndexExpr { `(` [ CallParams ] `)` } .

ErrorPropagationExpr = CallExpr { `?` } .

NegationExpr = { `-` | `!` } ErrorPropagationExpr .

TypeCastExpr = NegationExpr { `as` Type } .

MulExpr = TypeCastExpr { `*` | `/` | `%` TypeCastExpr } .

AddExpr = MulExpr { `+` | `-` MulExpr } .

ShiftExpr = AddExpr { `<<` | `>>` AddExpr } .

AndExpr = ShiftExpr { `&` ShiftExpr } .

XorExpr = AndExpr { `^` AndExpr } .

OrExpr = XorExpr { `|` XorExpr } .

ComparisonExpr = OrExpr { `==` | `!=` | `>` | `<` | `>=` | `<=` OrExpr } .

LogAndExpr = ComparisonExpr { `&&` ComparisonExpr } .

LogOrExpr = LogAndExpr { `||` LogAndExpr } .

RangeFromToExpr = LogOrExpr [ `..` LogOrExpr ] .  // later
RangeFromExpr = LogOrExpr [ `..` ] .  // later
RangeToExpr = `..` LogOrExpr .  // later
RangeFullExpr = `..` .  // later
RangeInclusiveExpr = LogOrExpr [ `..=` LogOrExpr ] .  // later
RangeToInclusiveExpr = `..=` LogOrExpr .  // later
RangeExpr = RangeFromToExpr | RangeFromExpr | RangeToExpr | RangeFullExpr | RangeInclusiveExpr | RangeToInclusiveExpr .  // later

//AssignmentExpr = RangeExpr [ `=` | `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` RangeExpr ] .  // later
AssignmentExpr = LogOrExpr [ `=` | `+=` | `-=` | `*=` | `/=` | `%=` | `&=` | `|=` | `^=` | `<<=` | `>>=` LogOrExpr ] .

IfExpr = `if` Expr `{` { Stat } `}` [ `else` ( `{` { Stat } `}` ) | IfExpr | IfLetExpr ] .

IfLetExpr = `if` `let` [ `|` ] Pat { `|` Pat } `=` Expr `{` { Stat } `}` [ `else` ( `{` { Stat } `}` ) | IfExpr | IfLetExpr ] .

MatchArm = [ `|` ] Pat { `|` Pat } [ `if` Expr ] `=>` Expr .

Expr =
    AssignmentExpr |
    `continue` |
    ( `break` [ Expr ] ) |
    ( `return` [ Expr ] ) |
    ( `{` { Stat } `}` ) |
    ( `loop` `{` { Stat } `}` ) |
    ( `while` Expr `{` { Stat } `}` ) |
    ( `while` `let` [ `|` ] Pat { `|` Pat } `=` Expr `{` { Stat } `}` ) |
    ( `for` Pat `in` Expr `{` { Stat } `}` ) |
    IfExpr |
    IfLetExpr |
    ( `match` Expr `{` [ { MatchArm `,` } MatchArm [ `,` ] ] `}` ) .

## Stats

ExprStat = Expr [ `;` ] .

LetStat = `let` Pat [ `:` Type ] [ `=` Expr ] `;` .

Stat = `;` | Item | LetStat | ExprStat .

## Items

enum Variant {  // later
    Ident(String),
    IdentExpr(String,Expr),
    Struct(String,Vec<(String,Type)>),
    Tuple(String,Vec<Type>),
}

enum Item {
    Module(String,Vec<Item>),
    Function(String,Vec<(Pat,Type)>,Option<Type>,Vec<Stat>),
    Struct(String,Vec<(String,Type)>),  // later
    Tuple(String,Vec<Type>),
    Enum(String,Vec<Variant>),  // later
    Union(String,Vec<(String,Type)>),  // later
    Constant(Option<String>,Type,Expr),  // later
    Static(String,Type,Expr),  // later
}

### Grammar

Module = `mod` IDENT `;` | ( `{` { Item } `}` ) .

Function = `fn` IDENT `(` [ Pat `:` Type { `,` Pat `:` Type } [ `,` ] ] `)` [ `->` Type ] `{` { Stat } `}` .

StructStruct = `struct` IDENT ( `{` [ IDENT `:` Type { `,` IDENT `:` Type } [ `,` ] ] `}` ) | `;` .

TupleStruct = `struct` IDENT `(` [ Type { `,` Type } [ `,` ] ] `)` `;` .

EnumItem = IDENT [
    ( `(` [ Type { `,` Type } [ `,` ] ] `)` ) |
    ( `{` [ IDENT `:` Type { `,` IDENT `:` Type } [ `,` ] ] `}` ) |
    ( `=` Expr )
] .
Enum = `enum` IDENT `{` [ EnumItem { `,` EnumItem } [ `,` ] ] `}` .

Union = `union` IDENT `{` IDENT `:` Type { `,` IDENT `:` Type } [ `,` ] `}` .

Constant = `const` IDENT | `_` `:` Type `=` Expr `;` .

Static = `static` [ `mut` ] IDENT `:` Type `=` Expr `;` .

Item = Module | Function | StructStruct | TupleStruct | Enum | Union | Constant | Static .
