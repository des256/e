use std::{
    cell::RefCell,
    collections::HashMap,
};

#[derive(Clone,Debug,PartialEq)]
pub enum Type {
    Inferred,
    Integer,
    Float,
    Void,
    Base(sr::BaseType),
    Ident(String),
    Struct(String),
    Tuple(String),
    Enum(String),
    Array(Box<Type>,Box<Expr>),
    AnonTuple(Vec<Type>),
}

/*impl PartialEq for Type {
    fn eq(&self,other: &Type) -> bool {
        match self {
            Type::Inferred => if let Type::Inferred = other { true } else { false },
            Type::Integer => if let Type::Integer = other { true } else { false },
            Type::Float => if let Type::Float = other { true } else { false },
            Type::Void => if let Type::Void = other { true } else { false },
            Type::Base(base_type) => if let Type::Base(base_type) = other { true } else { false },
            Type::Ident(ident) => if let Type::Ident(ident) = other { true } else { false },
            Type::Struct(ident) => if let Type::Struct(ident) = other { true } else { false },
            Type::Tuple(ident) => if let Type::Tuple(ident) = other { true } else { false },
            Type::Enum(ident) => if let Type::Enum(ident) = other { true } else { false },
            Type::Array(ty,expr) => if let Type::Array(ty,expr) = other { true } else { false },
            Type::AnonTuple(types) => if let Type::AnonTuple(types) = other { true } else { false },
        }
    }
}*/

#[derive(Clone,Debug,PartialEq)]
pub enum IdentPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
}

#[derive(Clone,Debug,PartialEq)]
pub enum VariantPat {
    Naked(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<IdentPat>),
}

#[derive(Clone,Debug,PartialEq)]
pub enum Pat {
    Wildcard,
    Rest,
    Boolean(bool),
    Integer(u64),
    Float(f64),
    Ident(String),
    Const(String,Type),
    Struct(String,Vec<IdentPat>),
    Tuple(String,Vec<Pat>),
    Array(Vec<Pat>),
    AnonTuple(Vec<Pat>),
    Variant(String,VariantPat),
    Range(Box<Pat>,Box<Pat>),
}

#[derive(Clone,Debug,PartialEq)]
pub enum VariantExpr {
    Naked(String),
    Tuple(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
}

#[derive(Clone,Debug,PartialEq)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
}

#[derive(Clone,Debug,PartialEq)]
pub enum Range {
    Only(Box<Expr>),
    FromTo(Box<Expr>,Box<Expr>),
    FromToIncl(Box<Expr>,Box<Expr>),
    From(Box<Expr>),
    To(Box<Expr>),
    ToIncl(Box<Expr>),
    All,
}

#[derive(Clone,Debug,PartialEq)]
pub enum Expr {
    Boolean(bool),
    Integer(u64),
    Float(f64),
    Ident(String),  // unknown identifier, resolves to Global, Local, Param or Const
    Local(String,Type),
    Param(String,Type),
    Const(String,Type),
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

#[derive(Clone,Debug,PartialEq)]
pub enum Stat {
    Let(Pat,Option<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone,Debug)]
pub enum Variant {
    Naked(String),
    Tuple(String,Vec<Type>),
    Struct(String,Vec<(String,Type)>),
}

#[derive(Clone,Debug)]
pub struct Module {
    pub ident: String,
    pub functions: RefCell<HashMap<String,(Vec<(String,Type)>,RefCell<Type>,RefCell<Block>)>>,
    pub structs: RefCell<HashMap<String,Vec<(String,RefCell<Type>)>>>,
    pub tuples: RefCell<HashMap<String,Vec<RefCell<Type>>>>,
    pub enums: RefCell<HashMap<String,Vec<Variant>>>,
    pub consts: RefCell<HashMap<String,(RefCell<Type>,RefCell<Expr>)>>,
    pub anon_tuple_count: RefCell<usize>,
}
