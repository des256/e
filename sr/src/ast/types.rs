use {
    std::{
        collections::HashMap,
        rc::Rc,
    },
};

// type
#[derive(Clone)]
pub enum Type {
    Inferred,  // type needs to be inferred from context
    Void,  // ()
    Integer,  // integer
    Float,  // float
    Bool,  // boolean
    U8,I8,U16,I16,U32,I32,U64,I64,USize,ISize,  // strict integer
    F16,F32,F64,  // strict float
    AnonTuple(Vec<Type>),  // ( type, ..., type, )
    Array(Box<Type>,Box<Expr>),  // [ type; expr ]
    UnknownIdent(String),  // unknown Tuple, Struct, Enum or Alias
    Tuple(Rc<Tuple>),
    Struct(Rc<Struct>),
    Enum(Rc<Enum>),
    Alias(Rc<Alias>),
}

#[derive(Clone)]
pub enum IndexPat {
    Wildcard,  // match anything
    Rest,  // skip the rest
    Index(usize),  // ident
    IndexPat(usize,Pat),  // ident: pat
}

// unknown struct field pattern
#[derive(Clone)]
pub enum IdentPat {
    Wildcard,  // match anything
    Rest,  // skip the rest
    Ident(String),  // ident
    IdentPat(String,Pat),  // ident: pat
}

#[derive(Clone)]
pub enum PatVariant {
    Naked(usize),  // ident
    Tuple(usize,Vec<Pat>),  // ident ( pat, ..., pat, )
    Struct(usize,Vec<IndexPat>),  // ident { patfield, ..., patfield, }
}

// unknown enum variant pattern
#[derive(Clone)]
pub enum UnknownPatVariant {
    Naked(String),  // ident
    Tuple(String,Vec<Pat>),  // ident ( pat, ..., pat, )
    Struct(String,Vec<IdentPat>),  // ident { patfield, ..., patfield, }
}

// pattern
#[derive(Clone)]
pub enum Pat {
    Wildcard,  // match anything
    Rest,  // skip the rest
    Boolean(bool),  // boolean
    Integer(i64),  // integer
    Float(f64),  // float
    AnonTuple(Vec<Pat>),  // ( pat, ..., pat, )
    Array(Vec<Pat>),  // [ pat, ..., pat, ]
    Range(Box<Pat>,Box<Pat>),  // pat ..= pat
    UnknownIdent(String),  // unknown pattern local or Const
    UnknownTuple(String,Vec<Pat>),  // ident ( pat, ..., pat, )
    UnknownStruct(String,Vec<IdentPat>),  // ident { patfield, ..., patfield, }
    UnknownVariant(String,UnknownPatVariant),  // ident :: patvariant
    Tuple(Rc<Tuple>,Vec<Pat>),
    Struct(Rc<Struct>,Vec<IndexPat>),
    Variant(Rc<Enum>,PatVariant),
}

#[derive(Clone)]
pub enum ExprVariant {
    Naked(usize),  // ident
    Tuple(usize,Vec<Expr>),  // ident ( expr, ..., expr, )
    Struct(usize,Vec<(usize,Expr)>),  // ident { ident: expr, ..., ident: expr, }
}

#[derive(Clone)]
pub enum UnknownExprVariant {
    Naked(String),  // ident
    Tuple(String,Vec<Expr>),  // ident ( expr, ..., expr, )
    Struct(String,Vec<(String,Expr)>),  // ident { ident: expr, ..., ident: expr, }
}

// { stat; ... stat; expr }
#[derive(Clone)]
pub struct Block {
    pub stats: Vec<Stat>,  // stat ... stat
    pub expr: Option<Box<Expr>>,  // expr
}

// unary operator
#[derive(Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

// binary operator
#[derive(Clone)]
pub enum BinaryOp {
    Mul,Div,Mod,Add,Sub,
    Shl,Shr,And,Or,Xor,
    Eq,NotEq,Greater,Less,GreaterEq,LessEq,
    LogAnd,LogOr,
    Assign,AddAssign,SubAssign,MulAssign,DivAssign,ModAssign,
    AndAssign,OrAssign,XorAssign,ShlAssign,ShrAssign,
}

// for range
#[derive(Clone)]
pub enum Range {
    Only(Box<Expr>),  // expr
    FromTo(Box<Expr>,Box<Expr>),  // expr .. expr
    FromToIncl(Box<Expr>,Box<Expr>),  // expr ..= expr
    From(Box<Expr>),  // expr ..
    To(Box<Expr>),  // .. expr
    ToIncl(Box<Expr>),  // ..= expr
    All,  // ..
}

// expression
#[derive(Clone)]
pub enum Expr {
    Boolean(bool),  // boolean
    Integer(i64),  // integer
    Float(f64),  // float
    Array(Vec<Expr>),  // [ expr, ..., expr, ]
    Cloned(Box<Expr>,Box<Expr>),  // [ expr; expr ]
    Index(Box<Expr>,Box<Expr>),  // expr [ expr ]
    Cast(Box<Expr>,Box<Type>),  // expr as type
    AnonTuple(Vec<Expr>),  // ( expr, ..., expr, )
    Unary(UnaryOp,Box<Expr>),  // unaryop expr
    Binary(Box<Expr>,BinaryOp,Box<Expr>),  // expr binaryop expr
    Continue,  // continue
    Break(Option<Box<Expr>>),  // break expr
    Return(Option<Box<Expr>>),  // return expr
    Block(Block),  // block
    If(Box<Expr>,Block,Option<Box<Expr>>),  // if expr block else expr
    Loop(Block),  // loop block
    While(Box<Expr>,Block),  // while expr block
    IfLet(Vec<Pat>,Box<Expr>,Block,Option<Box<Expr>>),  // if let pat = expr block else expr
    For(Vec<Pat>,Range,Block),  // for pat in range block
    WhileLet(Vec<Pat>,Box<Expr>,Block),  // while let pat = expr block
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),  // match expr { pat | ... | pat if expr => block, ..., pat | ... | pat if expr => block, }
    UnknownIdent(String),  // unknown Param, Local or Const
    UnknownTupleOrCall(String,Vec<Expr>),  // unknown Tuple or Call
    UnknownStruct(String,Vec<(String,Expr)>),  // unknown Struct
    UnknownVariant(String,UnknownExprVariant),  // unknown Variant
    UnknownMethod(Box<Expr>,String,Vec<Expr>),  // unknown Method
    UnknownField(Box<Expr>,String),  // unknown Field
    UnknownTupleIndex(Box<Expr>,usize),  // unknown TupleIndex
    Param(Rc<Symbol>),
    Local(Rc<Symbol>),
    Const(Rc<Const>),
    Tuple(Rc<Tuple>,Vec<Expr>),
    Call(Rc<Function>,Vec<Expr>),
    Struct(Rc<Struct>,Vec<Expr>),
    Variant(Rc<Enum>,ExprVariant),
    Method(Rc<Method>,Box<Expr>,Vec<Expr>),
    Field(Rc<Struct>,Box<Expr>,usize),
    TupleIndex(Rc<Tuple>,Box<Expr>,usize),
}

// block statement
#[derive(Clone)]
pub enum Stat {
    Let(Box<Pat>,Box<Type>,Box<Expr>),  // let pat: type = expr;
    Expr(Box<Expr>),  // expr;
}

// ident: type
pub struct Symbol {
    pub ident: String,
    pub type_: Type,
}

pub struct Method {
    pub from_type: Type,
    pub ident: String,
    pub params: Vec<Symbol>,
    pub type_: Type,
}

// fn ident ( ident: type, ..., ident: type, ) -> type { stat; ... stat; expr }
pub struct Function {
    pub ident: String,
    pub params: Vec<Symbol>,
    pub type_: Type,
    pub block: Block,
}

// struct ident ( type, ..., type, )
pub struct Tuple {
    pub ident: String,
    pub types: Vec<Type>,
}

// struct ident { ident: type, ..., ident: type, }
pub struct Struct {
    pub ident: String,
    pub fields: Vec<Symbol>,
}

pub enum Variant {
    Naked(String),  // ident
    Tuple(String,Vec<Type>),  // ident ( type, ..., type, )
    Struct(String,Vec<Symbol>),  // ident { ident: type, ..., ident: type, }
}

// enum { variant, ..., variant, }
pub struct Enum {
    pub ident: String,
    pub variants: Vec<Variant>,
}

// const ident: type = expr;
pub struct Const {
    pub ident: String,
    pub type_: Type,
    pub expr: Expr,
}

// type ident = type;
pub struct Alias {
    pub ident: String,
    pub type_: Type,
}

// mod ident { ... }
pub struct Module {
    pub ident: String,
    pub tuples: HashMap<String,Rc<Tuple>>,
    pub structs: HashMap<String,Rc<Struct>>,
    pub enums: HashMap<String,Rc<Enum>>,
    pub aliases: HashMap<String,Rc<Alias>>,
    pub consts: HashMap<String,Rc<Const>>,
    pub functions: HashMap<String,Rc<Function>>,
}
