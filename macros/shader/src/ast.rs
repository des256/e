pub(crate) struct GenArg {
    pub ty: Box<Type>,
    pub binding: Option<Box<Type>>,
    pub as_segs: Vec<Seg>,
}

pub(crate) enum Seg {
    Ident(String),
    Generic(Vec<GenArg>),
    Function(Vec<Box<Type>>,Option<Box<Type>>),
}

pub(crate) enum AnonParam {
    Anon(Box<Type>),
    Param(String,Box<Type>),
}

pub(crate) enum Type {
    Tuple(Vec<Box<Type>>),
    Array(Box<Type>,Box<Expr>),
    Slice(Box<Type>),
    Inferred,
    Never,
    Ref(bool,Box<Type>),
    Pointer(bool,Box<Type>),
    Function(Vec<AnonParam>,bool,Option<Box<Type>>),
    Segs(Vec<Seg>),
}

pub(crate) enum Pat {

}

pub(crate) enum Stat {
    Let(Box<Pat>,Option<Box<Type>>,Option<Box<Expr>>),
    Item(Item),
    Expr(Box<Expr>),
}

pub(crate) enum ExprField {
    LiteralExpr(String,Box<Expr>),
    IdentExpr(String,Box<Expr>),
    Ident(String),
}

pub(crate) struct Arm {
    pats: Vec<Box<Pat>>,
    if_expr: Option<Box<Expr>>,
    expr: Box<Expr>,
}

pub(crate) enum Expr {
    Literal(String),
    Segs(Vec<Seg>),
    Array(Vec<Box<Expr>>),
    CloneArray(Box<Expr>,Box<Expr>),
    Tuple(Vec<Box<Expr>>),
    StructEnumStruct(Box<Expr>,Vec<ExprField>,Option<Box<Expr>>),
    StructEnumTuple(Box<Expr>,Vec<Box<Expr>>),
    StructEnum(Box<Expr>),
    TupleIndex(Box<Expr>,String),
    Field(Box<Expr>,String),
    Index(Box<Expr>,Box<Expr>),
    Call(Box<Expr>,Vec<Box<Expr>>),
    Error(Box<Expr>),
    Borrow(bool,bool,Box<Expr>),
    Deref(Box<Expr>),
    Negate(Box<Expr>),
    LogNot(Box<Expr>),
    Cast(Box<Expr>,Box<Type>),
    Mul(Box<Expr>,Box<Expr>),
    Div(Box<Expr>,Box<Expr>),
    Mod(Box<Expr>,Box<Expr>),
    Add(Box<Expr>,Box<Expr>),
    Sub(Box<Expr>,Box<Expr>),
    Shl(Box<Expr>,Box<Expr>),
    Shr(Box<Expr>,Box<Expr>),
    And(Box<Expr>,Box<Expr>),
    Xor(Box<Expr>,Box<Expr>),
    Or(Box<Expr>,Box<Expr>),
    Eq(Box<Expr>,Box<Expr>),
    NotEq(Box<Expr>,Box<Expr>),
    Gt(Box<Expr>,Box<Expr>),
    NotGt(Box<Expr>,Box<Expr>),
    Lt(Box<Expr>,Box<Expr>),
    NotLt(Box<Expr>,Box<Expr>),
    LogAnd(Box<Expr>,Box<Expr>),
    LogOr(Box<Expr>,Box<Expr>),
    RangeIncl(Box<Expr>,Box<Expr>),
    Range(Box<Expr>,Box<Expr>),
    RangeToIncl(Box<Expr>),
    RangeTo(Box<Expr>),
    RangeFull,
    Assign(Box<Expr>,Box<Expr>),
    AddAssign(Box<Expr>,Box<Expr>),
    SubAssign(Box<Expr>,Box<Expr>),
    MulAssign(Box<Expr>,Box<Expr>),
    DivAssign(Box<Expr>,Box<Expr>),
    ModAssign(Box<Expr>,Box<Expr>),
    AndAssign(Box<Expr>,Box<Expr>),
    XorAssign(Box<Expr>,Box<Expr>),
    OrAssign(Box<Expr>,Box<Expr>),
    Block(Vec<Box<Stat>>),
    Continue,
    Break(Option<Box<Expr>>),
    Return(Option<Box<Expr>>),
    Loop(Vec<Box<Stat>>),
    For(Box<Pat>,Box<Expr>,Vec<Box<Stat>>),
    IfLet(Vec<Box<Pat>>,Box<Expr>,Vec<Box<Stat>>,Option<Box<Expr>>),
    If(Box<Expr>,Vec<Box<Stat>>,Option<Box<Expr>>),
    WhileLet(Vec<Box<Pat>>,Box<Expr>,Vec<Box<Stat>>),
    While(Box<Expr>,Vec<Box<Stat>>),
    Match(Box<Expr>,Vec<Box<Arm>>),
}

pub(crate) struct Param {
    pub pat: Pat,
    pub ty: Box<Type>,
}

pub(crate) struct Field {
    pub ident: String,
    pub ty: Box<Type>,
}

pub(crate) enum Variant {
    Struct(String,Vec<Field>),
    Tuple(String,Vec<Box<Type>>),
    Discr(String,Box<Expr>),
    Naked(String),
}

pub(crate) enum Item {
    Module(String,Vec<Item>),
    Function(String,Vec<String>,Vec<Param>,Option<Box<Type>>,Vec<Box<Stat>>),
    Alias(String,Vec<String>,Box<Type>),
    Struct(String,Vec<String>,Vec<Field>),
    Tuple(String,Vec<String>,Vec<Box<Type>>),
    Enum(String,Vec<String>,Vec<Variant>),
    Union(String,Vec<String>,Vec<Field>),
    Const(Option<String>,Box<Type>,Box<Expr>),
    Static(bool,String,Box<Type>,Box<Expr>),
}
