# Rust to SPIR-V, GLSL, HLSL and MSL

## Types

The base types have direct mapping to all shading languages.

Tuples can be refactored into structs with systematic names. Structs also
have a direct mapping to all shading languages.

Arrays seem to be available in all shading languages as well.

Enums will be harder, might be refactored into structs with systematic names.

## Literals

All literals are supported everywhere.

## Patterns, For, If Let, While Let and Match

This is the hardest problem to solve.


pub enum IdentPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
}

pub enum VariantPat {
    Naked(String),
    Struct(String,Vec<IdentPat>),
    Tuple(String,Vec<Pat>),
}

pub enum Pat {
    Wildcard,
    Rest,
    Literal(Literal),
    Const(String),
    Ident(String),  // unknown pattern, either resolves to Const, or remains a matchable identifier
    Struct(String,Vec<IdentPat>),
    Tuple(String,Vec<Pat>),
    Array(Vec<Pat>),
    AnonTuple(Vec<Pat>),
    Variant(String,VariantPat),
    Range(Box<Pat>,Box<Pat>),
}

pub enum VariantExpr {
    Naked(String),
    Struct(String,Vec<(String,Expr)>),
    Tuple(String,Vec<Expr>),
}

pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
}

pub enum Range {
    Only(Box<Expr>),
    FromTo(Box<Expr>,Box<Expr>),
    FromToIncl(Box<Expr>,Box<Expr>),
    From(Box<Expr>),
    To(Box<Expr>),
    ToIncl(Box<Expr>),
    All,
}

pub enum Expr {
    Literal(Literal),
    Ident(String),  // unknown identifier, resolves to Local, Param or Const
    Local(String),
    Param(String),
    Const(String),
    Array(Vec<Expr>),
    Cloned(Box<Expr>,Box<Expr>),
    Struct(String,Vec<(String,Expr)>),
    Tuple(String,Vec<Expr>),
    AnonTuple(Vec<Expr>),
    Variant(String,VariantExpr),
    Call(String,Vec<Expr>),
    Field(Box<Expr>,String),
    TupleIndex(Box<Expr>,u64),
    Index(Box<Expr>,Box<Expr>),
    Cast(Box<Expr>,Type),
    Neg(Box<Expr>),
    Not(Box<Expr>),
    Mul(Box<Expr>,Box<Expr>),
    Div(Box<Expr>,Box<Expr>),
    Mod(Box<Expr>,Box<Expr>),
    Add(Box<Expr>,Box<Expr>),
    Sub(Box<Expr>,Box<Expr>),
    Shl(Box<Expr>,Box<Expr>),
    Shr(Box<Expr>,Box<Expr>),
    And(Box<Expr>,Box<Expr>),
    Or(Box<Expr>,Box<Expr>),
    Xor(Box<Expr>,Box<Expr>),
    Eq(Box<Expr>,Box<Expr>),
    NotEq(Box<Expr>,Box<Expr>),
    Greater(Box<Expr>,Box<Expr>),
    Less(Box<Expr>,Box<Expr>),
    GreaterEq(Box<Expr>,Box<Expr>),
    LessEq(Box<Expr>,Box<Expr>),
    LogAnd(Box<Expr>,Box<Expr>),
    LogOr(Box<Expr>,Box<Expr>),
    Assign(Box<Expr>,Box<Expr>),
    AddAssign(Box<Expr>,Box<Expr>),
    SubAssign(Box<Expr>,Box<Expr>),
    MulAssign(Box<Expr>,Box<Expr>),
    DivAssign(Box<Expr>,Box<Expr>),
    ModAssign(Box<Expr>,Box<Expr>),
    AndAssign(Box<Expr>,Box<Expr>),
    OrAssign(Box<Expr>,Box<Expr>),
    XorAssign(Box<Expr>,Box<Expr>),
    ShlAssign(Box<Expr>,Box<Expr>),
    ShrAssign(Box<Expr>,Box<Expr>),
    Continue,
    Break(Option<Box<Expr>>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(Box<Expr>,Block,Option<Box<Expr>>),
    IfLet(Vec<Pat>,Box<Expr>,Block,Option<Box<Expr>>),
    Loop(Block),
    For(Vec<Pat>,Range,Block),
    While(Box<Expr>,Block),
    WhileLet(Vec<Pat>,Box<Expr>,Block),
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),
}

pub enum Stat {
    Let(Pat,Option<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

pub enum Variant {
    Naked(String),
    Struct(String,Vec<(String,Type)>),
    Tuple(String,Vec<Type>),
}

pub struct Module {
    pub ident: String,
    pub functions: HashMap<String,(Vec<(String,Type)>,Option<Type>,Block)>,
    pub structs: HashMap<String,Vec<(String,Type)>>,
    pub tuples: HashMap<String,Vec<Type>>,
    pub enums: HashMap<String,Vec<Variant>>,
    pub consts: HashMap<String,(Type,Expr)>,
}

// replace ident with according local, param, const
// detuplication
// render out pattern matching for if let, while let and patch