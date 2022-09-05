use {
    crate::*,
    std::collections::HashMap,
    std::rc::Rc,
};

#[derive(Debug,PartialEq)]
pub struct Variable {
    ident: String,
    ty: Type,
    value: Option<Expr>,
}

#[derive(Debug,PartialEq)]
pub struct Function {
    ident: String,
    params: Vec<Rc<Variable>>,
    return_type: Type,
    block: Block,
}

#[derive(Debug,PartialEq)]
pub struct Struct {
    fields: Vec<(String,Type)>,
}

#[derive(Clone,Debug,PartialEq)]
pub enum Variant {
    Naked(String),
    Tuple(String,Vec<Type>),
    Struct(String,Vec<(String,Type)>),
}

#[derive(Debug,PartialEq)]
pub struct Enum {
    variants: Vec<Variant>,
}

#[derive(Clone,Debug,PartialEq)]
pub enum Type {
    Inferred,
    Boolean,
    Integer,
    Float,
    Void,
    Base(BaseType),
    Ident(String),
    Struct(Rc<Struct>),
    Enum(Rc<Enum>),
    Array(Box<Type>,Box<Expr>),
}

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
    Integer(i64),
    Float(f64),
    Ident(String),
    Const(String),
    Struct(String,Vec<IdentPat>),
    Array(Vec<Pat>),
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
    Integer(i64),
    Float(f64),
    Base(BaseType,Vec<(String,Expr)>),
    Ident(String),  // unknown identifier, resolves to Global, Local, Param or Const
    Local(Rc<Variable>),
    Param(Rc<Variable>),
    Const(Rc<Variable>),
    Array(Vec<Expr>),
    Cloned(Box<Expr>,Box<Expr>),
    Struct(Rc<Struct>,Vec<(String,Expr)>),
    Variant(Rc<Enum>,VariantExpr),
    Call(Rc<Function>,Vec<Expr>),
    Field(Box<Expr>,String),
    Index(Box<Expr>,Box<Expr>),
    Cast(Box<Expr>,Box<Type>),
    AnonTuple(Vec<Expr>),
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
    //Let(Pat,Option<Type>,Box<Expr>),
    Let(Rc<Variable>),
    Expr(Box<Expr>),
}

#[derive(Clone,Debug)]
pub struct Module {
    pub ident: String,
    pub functions: HashMap<String,Rc<Function>>,
    pub structs: HashMap<String,Rc<Struct>>,
    pub enums: HashMap<String,Rc<Enum>>,
    pub consts: HashMap<String,Rc<Variable>>,
}
