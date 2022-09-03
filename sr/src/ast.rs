use {
    crate::*,
    std::collections::HashMap,
};

#[derive(Clone,Debug)]
pub enum Type {
    Inferred,
    Integer,
    Float,
    Void,
    Base(BaseType),
    Ident(&'static str),
    Struct(&'static str),
    Array(Box<Type>,Box<Expr>),
}

#[derive(Clone,Debug)]
pub enum IdentPat {
    Wildcard,
    Rest,
    Ident(&'static str),
    IdentPat(&'static str,Pat),
}

#[derive(Clone,Debug)]
pub enum Pat {
    Wildcard,
    Rest,
    Boolean(bool),
    Integer(u64),
    Float(f64),
    Ident(&'static str),
    Const(&'static str,Type),
    Struct(&'static str,Vec<IdentPat>),
    Array(Vec<Pat>),
    Range(Box<Pat>,Box<Pat>),
}

#[derive(Clone,Debug)]
pub struct Block {
    pub stats: Vec<Stat>,
    pub expr: Option<Box<Expr>>,
}

#[derive(Clone,Debug)]
pub enum Range {
    Only(Box<Expr>),
    FromTo(Box<Expr>,Box<Expr>),
    FromToIncl(Box<Expr>,Box<Expr>),
    From(Box<Expr>),
    To(Box<Expr>),
    ToIncl(Box<Expr>),
    All,
}

#[derive(Clone,Debug)]
pub enum Expr {
    Boolean(bool),
    Integer(u64),
    Float(f64),
    Local(&'static str,Type),
    Param(&'static str,Type),
    Const(&'static str,Type),
    Array(Vec<Expr>),
    Cloned(Box<Expr>,Box<Expr>),
    Struct(&'static str,Vec<(&'static str,Expr)>),
    Call(&'static str,Vec<Expr>),
    Field(Box<Expr>,&'static str),
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

#[derive(Clone,Debug)]
pub enum Stat {
    Let(Pat,Option<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone,Debug)]
pub struct Module {
    pub ident: &'static str,
    pub functions: HashMap<&'static str,(Vec<(&'static str,Type)>,Type,Block)>,
    pub structs: HashMap<&'static str,Vec<(&'static str,Type)>>,
    pub consts: HashMap<&'static str,(Type,Expr)>,
}
