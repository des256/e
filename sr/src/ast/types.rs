use {
    std::{
        collections::HashMap,
        rc::Rc,
    },
};

// type
#[derive(Clone)]
pub enum Type {
    // during macro
    Inferred,
    Void,
    Integer,
    Float,
    Bool,
    U8,I8,U16,I16,U32,I32,U64,I64,USize,ISize,
    F16,F32,F64,
    AnonTuple(Vec<Type>),
    Array(Box<Type>,Box<Expr>),
    UnknownIdent(String),  // resolves to Tuple, Struct, Enum or Alias

    // during run-time
    Tuple(Rc<Tuple>),
    Struct(Rc<Struct>),
    Enum(Rc<Enum>),
    Alias(Rc<Alias>),
}

// field-pattern during macro
#[derive(Clone)]
pub enum UnknownFieldPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
}

// variant-pattern during macro
#[derive(Clone)]
pub enum UnknownVariantPat {
    Naked(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<UnknownFieldPat>),
}

// field-pattern during run-time
#[derive(Clone)]
pub enum FieldPat {
    Wildcard,
    Rest,
    Index(usize),
    IndexPat(usize,Pat),
}

// variant-pattern during run-time
#[derive(Clone)]
pub enum VariantPat {
    Naked(usize),
    Tuple(usize,Vec<Pat>),
    Struct(usize,Vec<FieldPat>),
}

// pattern
#[derive(Clone)]
pub enum Pat {
    // macro
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

    // run-time
    Tuple(Rc<Tuple>,Vec<Pat>),
    Struct(Rc<Struct>,Vec<FieldPat>),
    Variant(Rc<Enum>,VariantPat),
}

// variant-expression during run-time
#[derive(Clone)]
pub enum VariantExpr {
    Naked(usize),
    Tuple(usize,Vec<Expr>),
    Struct(usize,Vec<Expr>),
}

// variant-expression during macro
#[derive(Clone)]
pub enum UnknownVariantExpr {
    Naked(String),
    Tuple(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
}

// block
#[derive(Clone)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
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

// for-range
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

// expression
#[derive(Clone)]
pub enum Expr {
    // macro
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

    // run-time
    Param(Rc<Symbol>),
    Local(Rc<Symbol>),
    Const(Rc<Const>),
    Tuple(Rc<Tuple>,Vec<Expr>),
    Call(Rc<Function>,Vec<Expr>),
    Struct(Rc<Struct>,Vec<Expr>),
    Variant(Rc<Enum>,VariantExpr),
    Method(Box<Expr>,Rc<Method>,Vec<Expr>),
    Field(Rc<Struct>,Box<Expr>,usize),
    TupleIndex(Rc<Tuple>,Box<Expr>,usize),
}

// statement
#[derive(Clone)]
pub enum Stat {
    // macro
    Let(Box<Pat>,Box<Type>,Box<Expr>),
    Expr(Box<Expr>),

    // run-time
    Local(Rc<Symbol>,Box<Expr>),
}

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

impl Struct {
    pub fn find_field(&self,ident: &str) -> Option<usize> {
        for i in 0..self.fields.len() {
            if self.fields[i].ident == ident {
                return Some(i);
            }
        }
        None
    }
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
