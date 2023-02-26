#[derive(Clone)]
pub enum Type {
    Inferred,
    Void,
    Integer,Float,
    Bool,
    U8,I8,U16,I16,U32,I32,U64,I64,USize,ISize,
    F16,F32,F64,
    AnonTuple(Vec<Type>),
    Array(Box<Type>,Box<Expr>),
    UnknownIdent(String),
    Generic(String,Vec<Type>),
}

#[derive(Clone)]
pub enum UnknownFieldPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
}

#[derive(Clone)]
pub enum UnknownVariantPat {
    Naked(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<UnknownFieldPat>),
}

#[derive(Clone)]
pub enum Pat {
    Wildcard,
    Rest,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    AnonTuple(Vec<Pat>),
    Array(Vec<Pat>),
    Range(Box<Pat>,Box<Pat>),
    UnknownIdent(String),
    UnknownTuple(String,Vec<Pat>),
    UnknownStruct(String,Vec<UnknownFieldPat>),
    UnknownVariant(String,UnknownVariantPat),
}

#[derive(Clone)]
pub enum UnknownVariantExpr {
    Naked(String),
    Tuple(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
}

#[derive(Clone)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
}

#[derive(Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Clone)]
pub enum BinaryOp {
    Mul,Div,Mod,Add,Sub,
    Shl,Shr,And,Or,Xor,
    Eq,NotEq,Greater,Less,GreaterEq,LessEq,
    LogAnd,LogOr,
    Assign,AddAssign,SubAssign,MulAssign,DivAssign,ModAssign,
    AndAssign,OrAssign,XorAssign,ShlAssign,ShrAssign,
}

#[derive(Clone)]
pub enum Range {
    Only(Box<Expr>),
    FromTo(Box<Expr>,Box<Expr>),
    FromToIncl(Box<Expr>,Box<Expr>),
    From(Box<Expr>),
    To(Box<Expr>),
    ToIncl(Box<Expr>),
    All,
}

#[derive(Clone)]
pub enum Expr {
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Array(Vec<Expr>),
    Cloned(Box<Expr>,Box<Expr>),
    Index(Box<Expr>,Box<Expr>),
    Cast(Box<Expr>,Box<Type>),
    AnonTuple(Vec<Expr>),
    Unary(UnaryOp,Box<Expr>),
    Binary(Box<Expr>,BinaryOp,Box<Expr>),
    Continue,
    Break(Option<Box<Expr>>),
    Return(Option<Box<Expr>>),
    Block(Block),
    If(Box<Expr>,Block,Option<Box<Expr>>),
    While(Box<Expr>,Block),
    Loop(Block),
    IfLet(Vec<Pat>,Box<Expr>,Block,Option<Box<Expr>>),
    For(Vec<Pat>,Range,Block),
    WhileLet(Vec<Pat>,Box<Expr>,Block),
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),
    UnknownIdent(String),
    UnknownTupleOrCall(String,Vec<Expr>),
    UnknownStruct(String,Vec<(String,Expr)>),
    UnknownVariant(String,UnknownVariantExpr),
    UnknownMethod(Box<Expr>,String,Vec<Expr>),
    UnknownField(Box<Expr>,String),
    UnknownTupleIndex(Box<Expr>,usize),
}

#[derive(Clone)]
pub enum Stat {
    Let(Box<Pat>,Box<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

pub struct Function {
    pub ident: String,
    pub params: Vec<(String,Type)>,
    pub type_: Type,
    pub block: Block,
}

pub struct Tuple {
    pub ident: String,
    pub types: Vec<Type>,
}

pub struct Struct {
    pub ident: String,
    pub fields: Vec<(String,Type)>,
}

pub enum Variant {
    Naked(String),
    Tuple(String,Vec<Type>),
    Struct(String,Vec<(String,Type)>),
}

pub struct Enum {
    pub ident: String,
    pub variants: Vec<Variant>,
}

pub struct Const {
    pub ident: String,
    pub type_: Type,
    pub expr: Expr,
}

pub struct Alias {
    pub ident: String,
    pub type_: Type,
}

pub struct Module {
    pub ident: String,
    pub tuples: Vec<Tuple>,
    pub structs: Vec<Struct>,
    pub enums: Vec<Enum>,
    pub aliases: Vec<Alias>,
    pub consts: Vec<Const>,
    pub functions: Vec<Function>,
}
