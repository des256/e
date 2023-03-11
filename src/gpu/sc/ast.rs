use std::collections::HashMap;

#[derive(Clone)]
pub enum Type {
    Inferred,
    Void,
    Bool,
    U8,I8,U16,I16,U32,I32,U64,I64,
    F16,F32,F64,
    Vec2Bool,
    Vec2U8,Vec2I8,Vec2U16,Vec2I16,Vec2U32,Vec2I32,Vec2U64,Vec2I64,
    Vec2F16,Vec2F32,Vec2F64,
    Vec3Bool,
    Vec3U8,Vec3I8,Vec3U16,Vec3I16,Vec3U32,Vec3I32,Vec3U64,Vec3I64,
    Vec3F16,Vec3F32,Vec3F64,
    Vec4Bool,
    Vec4U8,Vec4I8,Vec4U16,Vec4I16,Vec4U32,Vec4I32,Vec4U64,Vec4I64,
    Vec4F16,Vec4F32,Vec4F64,
    Mat2x2F32,Mat2x2F64,
    Mat2x3F32,Mat2x3F64,
    Mat2x4F32,Mat2x4F64,
    Mat3x2F32,Mat3x2F64,
    Mat3x3F32,Mat3x3F64,
    Mat3x4F32,Mat3x4F64,
    Mat4x2F32,Mat4x2F64,
    Mat4x3F32,Mat4x3F64,
    Mat4x4F32,Mat4x4F64,
    AnonTuple(Vec<Type>),
    Array(Box<Type>,Box<Expr>),
    Ident(&'static str),

    AnonTupleRef(usize),
}

#[derive(Clone)]
pub enum FieldPat {
    Wildcard,
    Rest,
    Ident(&'static str),
    IdentPat(&'static str,Pat),
}

#[derive(Clone)]
pub enum VariantPat {
    Naked,
    Tuple(Vec<Pat>),
    Struct(Vec<FieldPat>),
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
    Ident(&'static str),
    Tuple(&'static str,Vec<Pat>),
    Struct(&'static str,Vec<FieldPat>),
    Variant(&'static str,&'static str,VariantPat),
}

#[derive(Clone)]
pub enum VariantExpr {
    Naked,
    Tuple(Vec<Expr>),
    Struct(Vec<(&'static str,Expr)>),
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
    Ident(&'static str),
    TupleOrCall(&'static str,Vec<Expr>),
    Struct(&'static str,Vec<(&'static str,Expr)>),
    Variant(&'static str,&'static str,VariantExpr),
    Method(Box<Expr>,&'static str,Vec<Expr>),
    Field(Box<Expr>,&'static str),
    TupleIndex(Box<Expr>,usize),
}

#[derive(Clone)]
pub enum Stat {
    Let(Box<Pat>,Box<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone)]
pub struct Method {
    pub from_type: Type,
    pub ident: &'static str,
    pub params: Vec<(&'static str,Type)>,
    pub return_type: Type,
}

#[derive(Clone)]
pub struct Function {
    pub ident: &'static str,
    pub params: Vec<(&'static str,Type)>,
    pub return_type: Type,
    pub block: Block,
}

#[derive(Clone)]
pub struct Struct {
    pub ident: &'static str,
    pub fields: Vec<(&'static str,Type)>,
}

#[derive(Clone)]
pub enum Variant {
    Naked,
    Tuple(Vec<Type>),
    Struct(Vec<(&'static str,Type)>),
}

#[derive(Clone)]
pub struct Enum {
    pub ident: &'static str,
    pub variants: Vec<(&'static str,Variant)>,
}

#[derive(Clone)]
pub struct Const {
    pub ident: &'static str,
    pub type_: Type,
    pub expr: Expr,
}

#[derive(Clone)]
pub struct Alias {
    pub ident: &'static str,
    pub type_: Type,
}

#[derive(Clone)]
pub struct Module {
    pub ident: &'static str,
    pub structs: HashMap<&'static str,Struct>,
    pub extern_structs: HashMap<&'static str,Struct>,
    pub enums: HashMap<&'static str,Enum>,
    pub aliases: HashMap<&'static str,Alias>,
    pub consts: HashMap<&'static str,Const>,
    pub functions: HashMap<&'static str,Function>,
}
