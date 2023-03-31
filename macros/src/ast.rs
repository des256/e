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
    Array(Box<Type>,usize),
    Ident(String),
}

#[derive(Clone)]
pub enum FieldPat {
    Wildcard,
    Rest,
    Ident(String),
    IdentPat(String,Pat),
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
    Ident(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<FieldPat>),
    Variant(String,String,VariantPat),
}

#[derive(Clone)]
pub enum VariantExpr {
    Naked,
    Tuple(Vec<Expr>),
    Struct(Vec<(String,Expr)>),
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
    Cloned(Box<Expr>,usize),
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
    Ident(String),
    TupleLitOrFunctionCall(String,Vec<Expr>),
    StructLit(String,Vec<(String,Expr)>),
    Variant(String,String,VariantExpr),
    MethodCall(Box<Expr>,String,Vec<Expr>),
    Field(Box<Expr>,String),
}

#[derive(Clone)]
pub enum Stat {
    Let(Box<Pat>,Box<Type>,Box<Expr>),
    Expr(Box<Expr>),
}

#[derive(Clone)]
pub struct Local {
    pub ident: String,
    pub type_: Type,
}

#[derive(Clone)]
pub struct Method {
    pub from_type: Type,
    pub ident: String,
    pub params: Vec<(String,Type)>,
    pub return_type: Type,
}

#[derive(Clone)]
pub struct Function {
    pub ident: String,
    pub params: Vec<(String,Type)>,
    pub return_type: Type,
    pub block: Block,
}

#[derive(Clone)]
pub struct Tuple {
    pub ident: String,
    pub types: Vec<Type>,
}

#[derive(Clone)]
pub struct Struct {
    pub ident: String,
    pub fields: Vec<(String,Type)>,
}

#[derive(Clone)]
pub enum Variant {
    Naked,
    Tuple(Vec<Type>),
    Struct(Vec<(String,Type)>),
}

#[derive(Clone)]
pub struct Enum {
    pub ident: String,
    pub variants: Vec<(String,Variant)>,
}

#[derive(Clone)]
pub struct Const {
    pub ident: String,
    pub type_: Type,
    pub expr: Expr,
}

#[derive(Clone)]
pub struct Alias {
    pub ident: String,
    pub type_: Type,
}

#[derive(Clone)]
pub struct Module {
    pub ident: String,
    pub tuples: Vec<Tuple>,
    pub structs: Vec<Struct>,
    pub extern_structs: Vec<Struct>,
    pub enums: Vec<Enum>,
    pub aliases: Vec<Alias>,
    pub consts: Vec<Const>,
    pub functions: Vec<Function>,
}
