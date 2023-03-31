// ast::Type describes type nodes 
#[derive(Clone,Eq,Hash)]
pub enum Type {

    // the type is inferred by the context around it
    Inferred,

    // ()
    Void,

    // the expected types relevant to shaders
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

    // anonymous tuple, these are converted to AnonTupleRef in the prepare pass
    AnonTuple(Vec<Type>),

    // fixed-size array
    Array(Box<Type>,usize),

    // named struct, tuple, alias or enum reference, converted to other types in prepare and deenumify passes
    Ident(&'static str),

    AnonTupleRef(usize),
    StructRef(&'static str),
    TupleRef(&'static str),
    EnumRef(&'static str),
}

// pattern matching: ast::FieldPat describe a (partial) field in a struct or struct enum variant, resolved in the destructure pass
#[derive(Clone)]
pub enum FieldPat {

    // matches anything for this field
    Wildcard,

    // matches anything for this and further fields
    Rest,

    // matches an identifier for this field
    Ident(&'static str),

    // matches an identifier and a pattern for this field
    IdentPat(&'static str,Pat),
}

// pattern matching: ast::VariantPat describe matching for the different kinds of enum variants, resolved in the destructure pass
#[derive(Clone)]
pub enum VariantPat {

    // no additional specification needed
    Naked,

    // the variant has extra patterns matching the tuple components
    Tuple(Vec<Pat>),

    // the variant has extra patterns matching the struct fields
    Struct(Vec<FieldPat>),
}

// pattern matching: ast::Pat describes pattern nodes, resolved in the destructure pass
#[derive(Clone)]
pub enum Pat {

    // this pattern matches anything
    Wildcard,

    // this and following patterns match anything
    Rest,

    // match boolean constant
    Boolean(bool),

    // match integer constant
    Integer(i64),

    // match float constant
    Float(f64),

    // match anonymous tuple literal of patterns
    AnonTuple(Vec<Pat>),

    // match array of patterns
    Array(Vec<Pat>),

    // match range
    Range(Box<Pat>,Box<Pat>),

    // match local variable (to be destructured) or constant
    Ident(&'static str),

    // match tuple literal of patterns
    Tuple(&'static str,Vec<Pat>),

    // match struct literal of patterns
    Struct(&'static str,Vec<FieldPat>),

    // match enum variant literal
    Variant(&'static str,&'static str,VariantPat),
}

// literals of the different kinds of enum variants
#[derive(Clone)]
pub enum VariantExpr {

    // no additional specification needed
    Naked,

    // tuple component literals
    Tuple(Vec<Expr>),

    // struct field literals
    Struct(Vec<(&'static str,Expr)>),
}

// ast::Block describes block nodes
#[derive(Clone)]
pub struct Block {

    // statements (with no return type)
    pub stats: Vec<Stat>,

    // optional return expression
    pub expr: Option<Box<Expr>>,
}

// unary operators
#[derive(Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

// binary operators
#[derive(Clone)]
pub enum BinaryOp {
    Mul,Div,Mod,Add,Sub,
    Shl,Shr,And,Or,Xor,
    Eq,NotEq,Greater,Less,GreaterEq,LessEq,
    LogAnd,LogOr,
    Assign,AddAssign,SubAssign,MulAssign,DivAssign,ModAssign,
    AndAssign,OrAssign,XorAssign,ShlAssign,ShrAssign,
}

// ast::Range describes ranges for for-statements
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

// ast::Expr describes statement/expression nodes
#[derive(Clone)]
pub enum Expr {

    // boolean literal
    Boolean(bool),

    // integer literal
    Integer(i64),

    // float literal
    Float(f64),

    // array literal
    Array(Vec<Expr>),

    // array cloning literal
    Cloned(Box<Expr>,usize),

    // array index
    Index(Box<Expr>,Box<Expr>),

    // type cast
    Cast(Box<Expr>,Box<Type>),

    // anonymous tuple, these are converted to AnonTupleLit
    AnonTuple(Vec<Expr>),

    // unary operation
    Unary(UnaryOp,Box<Expr>),

    // binary operation
    Binary(Box<Expr>,BinaryOp,Box<Expr>),

    // continue-statement
    Continue,

    // break-statement
    Break(Option<Box<Expr>>),

    // return-statement
    Return(Option<Box<Expr>>),

    // nested block
    Block(Block),

    // if-statement
    If(Box<Expr>,Block,Option<Box<Expr>>),

    // while-statement
    While(Box<Expr>,Block),

    // loop-statement
    Loop(Block),

    // if-let-statement, these are converted to if-statements in the destructure pass
    IfLet(Vec<Pat>,Box<Expr>,Block,Option<Box<Expr>>),

    // for-statement
    For(Vec<Pat>,Range,Block),

    // while-let-statement, these are converted to while-statements in the destructure pass
    WhileLet(Vec<Pat>,Box<Expr>,Block),

    // match-statement, these are converted to if-statements in the destructure pass
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),

    // const, local or parameter references, converted to ConstRef, LocalRef or ParamRef in the prepare pass
    Ident(&'static str),

    // named tuple literal or function call, converted to TupleRef or FunctionRef in the prepare pass
    TupleLitOrFunctionCall(&'static str,Vec<Expr>),

    // named struct literal, converted to StructRef in the prepare pass
    StructLit(&'static str,Vec<(&'static str,Expr)>),

    // named enum variant literal, converted to Struct in the deenumify pass
    VariantLit(&'static str,&'static str,VariantExpr),

    // method call (only for stdlib objects)
    MethodCall(Box<Expr>,&'static str,Vec<Expr>),

    // structure field selector
    Field(Box<Expr>,&'static str),

    // tuple index selector, converted to Field in the prepare pass
    TupleIndex(Box<Expr>,usize),

    AnonTupleLit(usize,Vec<Expr>),
    LocalRefOrParamRef(&'static str),
    ConstRef(&'static str),
    FunctionCall(&'static str,Vec<Expr>),
    TupleLit(&'static str,Vec<Expr>),
    EnumDiscr(Box<Expr>,usize),
    EnumArg(Box<Expr>,usize,usize),
}

// ast::Stat describes statement expressions that appear in blocks
#[derive(Clone)]
pub enum Stat {

    // let-statement, converted to Local nodes in the destructure pass
    Let(Box<Pat>,Box<Type>,Box<Expr>),

    // expression-statement where the result is ignored
    Expr(Box<Expr>),

    Local(&'static str,Box<Type>,Box<Expr>),
}

// ast::Method describes a method (only in stdlib)
#[derive(Clone)]
pub struct Method {

    // type this method can be applied to
    pub from_type: Type,

    // identifier of the method
    pub ident: &'static str,

    // method parameter descriptions
    pub params: Vec<(&'static str,Type)>,

    // method return type
    pub return_type: Type,
}

// ast::Function describes a function
#[derive(Clone)]
pub struct Function {

    // identifier for the function
    pub ident: &'static str,

    // function parameter descriptions
    pub params: Vec<(&'static str,Type)>,

    // function return type
    pub return_type: Type,

    // contents of the function
    pub block: Block,
}

// ast::Struct describes a struct
#[derive(Clone)]
pub struct Struct {

    // identifier for the struct
    pub ident: &'static str,

    // struct field descriptions
    pub fields: Vec<(&'static str,Type)>,
}

// ast::Tuple describes a tuple
#[derive(Clone)]
pub struct Tuple {

    // identifier for the tuple
    pub ident: &'static str,

    // tuple type descriptions
    pub types: Vec<Type>,
}

// ast::Variant describes an enum variant
#[derive(Clone)]
pub enum Variant {

    // no further processing is needed
    Naked,

    // tuple variant description
    Tuple(Vec<Type>),

    // struct variant description
    Struct(Vec<(&'static str,Type)>),
}

// ast::Enum describes an enum
#[derive(Clone)]
pub struct Enum {

    // identifier for the enum
    pub ident: &'static str,

    // enum variant descriptions
    pub variants: Vec<(&'static str,Variant)>,
}

// ast::Const describes a constant value
#[derive(Clone)]
pub struct Const {

    // identifier for the constant
    pub ident: &'static str,

    // type of the constant
    pub type_: Type,

    // constant contents
    pub expr: Expr,
}

// ast::Alias describes a type alias, resolved in prepare pass
#[derive(Clone)]
pub struct Alias {

    // alias identifier
    pub ident: &'static str,

    // type the alias represents
    pub type_: Type,
}

// ast::Module describes the module as it comes from the parser macro
#[derive(Clone)]
pub struct Module {
    pub ident: &'static str,
    pub tuples: Vec<Tuple>,
    pub structs: Vec<Struct>,
    pub extern_structs: Vec<Struct>,
    pub enums: Vec<Enum>,
    pub aliases: Vec<Alias>,
    pub consts: Vec<Const>,
    pub functions: Vec<Function>,
}

pub struct ProcessedModule {
    pub ident: &'static str,
    pub tuples: Vec<Tuple>,
    pub anon_tuple_types: Vec<Vec<Type>>,
    pub structs: Vec<Struct>,
    pub extern_structs: Vec<Struct>,
    pub enum_tuples: Vec<Tuple>,
    pub enum_mappings: Vec<Vec<Vec<usize>>>,
    pub consts: Vec<Const>,
    pub functions: Vec<Function>,
}