use std::collections::HashMap;

pub enum BaseType {
    Bool,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Vec2Bool,
    Vec2U8,
    Vec2U16,
    Vec2U32,
    Vec2U64,
    Vec2I8,
    Vec2I16,
    Vec2I32,
    Vec2I64,
    Vec2F16,
    Vec2F32,
    Vec2F64,
    Vec3Bool,
    Vec3U8,
    Vec3U16,
    Vec3U32,
    Vec3U64,
    Vec3I8,
    Vec3I16,
    Vec3I32,
    Vec3I64,
    Vec3F16,
    Vec3F32,
    Vec3F64,
    Vec4Bool,
    Vec4U8,
    Vec4U16,
    Vec4U32,
    Vec4U64,
    Vec4I8,
    Vec4I16,
    Vec4I32,
    Vec4I64,
    Vec4F16,
    Vec4F32,
    Vec4F64,
    ColorU8,
    ColorU16,
    ColorF16,
    ColorF32,
    ColorF64,
}

pub enum Literal {
    Boolean(bool),
    Integer(u64),
    Double(f64),
}

pub enum Type {
    Inferred,
    Base(BaseType),
    Ident(String),  // unknown type, should be resolved to Struct, Tuple or Enum
    Struct(String),
    Tuple(String),
    Enum(String),
    Array(Box<Type>,Box<Expr>),
    AnonTuple(Vec<Type>),
}

pub enum IdentPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
}

pub enum VariantPat {
    Naked(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<IdentPat>),
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
    Tuple(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
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
    Ident(String),  // unknown identifier, resolves to Global, Local, Param or Const
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
    Tuple(String,Vec<Type>),
    Struct(String,Vec<(String,Type)>),
}

pub struct Module {
    pub ident: String,
    pub functions: HashMap<String,(Vec<(String,Type)>,Option<Type>,Block)>,
    pub structs: HashMap<String,Vec<(String,Type)>>,
    pub tuples: HashMap<String,Vec<Type>>,
    pub enums: HashMap<String,Vec<Variant>>,
    pub consts: HashMap<String,(Type,Expr)>,
}
