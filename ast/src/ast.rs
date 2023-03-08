use std::collections::HashMap;

// The AST is defined so it makes the initial parsing, as well as the transformation steps as easy as possible. All steps use the same AST, even though some nodes only have a function in some steps. Ideally, each transformation is one AST traversal, and we keep translating until no more changes happen. After this, attempting to render the AST into TAC reveals the errors.

// SimpleType: These are types that contain one value and map directly to the target shader language
#[derive(Clone)]
pub enum SimpleType {
}

// Type: all possible type nodes in the AST
#[derive(Clone)]
pub enum Type {

    // inferred type: the actual type should be inferred from the context
    Inferred,

    // void type: nothing, or ()
    Void,

    // numeric literals: the actual type should be inferred from the context
    Integer,Float,

    // simple types: these map directly to the target shader language
    Bool,
    U8,I8,U16,I16,U32,I32,U64,I64,
    F16,F32,F64,

    // anonymous tuple: this should be converted to a regular struct (or resolved entirely)
    AnonTuple(Vec<Type>),

    // array type
    Array(Box<Type>,Box<Expr>),

    // unknown identifier: this should be converted to Struct, Tuple, Enum or remapped alias
    UnknownStructTupleEnumAlias(String),

    // struct type reference: this refers to a struct, a vector or a matrix
    Struct(String),

    // tuple type reference: this refers to a tuple and should ultimately be converted to a struct
    Tuple(String),

    // enum type reference: this refers to an enum and should ultimately be converted to a struct
    Enum(String),

    // alias type reference: this refers to an alias and should ultimately be converted to whatever it points to
    Alias(String),
}

// FieldPat: pattern definition representing one or more fields inside struct or enum struct variant
#[derive(Clone)]
pub enum FieldPat {

    // this field matches anything
    Wildcard,

    // this and following fields match anything
    Rest,

    // this field matches an identifier
    Ident(String),

    // this field matches an identifier and a type
    IdentPat(String,Pat),
}

// VariantPat: pattern definition representing enum variants
#[derive(Clone)]
pub enum VariantPat {

    // no details
    Naked,

    // tuple variant with patterns
    Tuple(Vec<Pat>),

    // struct variant with field patterns
    Struct(Vec<FieldPat>),
}

// Pat: pattern definition, should be resolved completely since no target shader language supports pattern matching like Rust
#[derive(Clone)]
pub enum Pat {

    // matches anything
    Wildcard,

    // this and following patterns match anything
    Rest,

    // hard match: boolean value
    Boolean(bool),

    // hard match: integer value
    Integer(i64),

    // hard match: float value
    Float(f64),

    // matches anonymous tuple
    AnonTuple(Vec<Pat>),

    // matches array
    Array(Vec<Pat>),

    // matches range
    Range(Box<Pat>,Box<Pat>),

    // matches something that can be named
    Ident(String),

    // matches named tuple with patterns
    Tuple(String,Vec<Pat>),

    // matches named struct with field patterns
    Struct(String,Vec<FieldPat>),

    // matches named enum variant with variant patterns
    Variant(String,String,VariantPat),
}

// VariantExpr: enum variant expression, should be resolved completely since no target shader language supports enums like Rust
#[derive(Clone)]
pub enum VariantExpr {

    // no details
    Naked,

    // tuple variant expression
    Tuple(Vec<Expr>),

    // struct variant expression
    Struct(Vec<(String,Expr)>),
}

// Block: code block, contains statements and an optional final expression that will be returned to the super block or module
#[derive(Clone)]
pub struct Block {

    // the statements
    pub stats: Vec<Stat>,

    // optional return expression
    pub expr: Option<Box<Expr>>,
}

// UnaryOp: unary operators
#[derive(Clone)]
pub enum UnaryOp {
    Neg,
    Not,
}

// BinaryOp: binary operators
#[derive(Clone)]
pub enum BinaryOp {
    Mul,Div,Mod,Add,Sub,
    Shl,Shr,And,Or,Xor,
    Eq,NotEq,Greater,Less,GreaterEq,LessEq,
    LogAnd,LogOr,
    Assign,AddAssign,SubAssign,MulAssign,DivAssign,ModAssign,
    AndAssign,OrAssign,XorAssign,ShlAssign,ShrAssign,
}

// Range: range definition (mostly for for statements)
#[derive(Clone)]
pub enum Range {  // resolve?
    Only(Box<Expr>),
    FromTo(Box<Expr>,Box<Expr>),
    FromToIncl(Box<Expr>,Box<Expr>),
    From(Box<Expr>),
    To(Box<Expr>),
    ToIncl(Box<Expr>),
    All,
}

// Expr: all expression nodes
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

    // cloned array literal
    Cloned(Box<Expr>,Box<Expr>),

    // array index
    Index(Box<Expr>,Box<Expr>),

    // expression cast to type
    Cast(Box<Expr>,Box<Type>),

    // anonymous tuple literal, should be converted to struct literal
    AnonTuple(Vec<Expr>),

    // unary operation
    Unary(UnaryOp,Box<Expr>),

    // binary operation
    Binary(Box<Expr>,BinaryOp,Box<Expr>),

    // continue statement
    Continue,

    // break statement
    Break(Option<Box<Expr>>),

    // return statement, value should have same type as block return expression or function result
    Return(Option<Box<Expr>>),

    // nested block
    Block(Block),

    // if statement
    If(Box<Expr>,Block,Option<Box<Expr>>),

    // while statement
    While(Box<Expr>,Block),

    // loop statement
    Loop(Block),

    // if let statement, should be converted to if statement
    IfLet(Vec<Pat>,Box<Expr>,Block,Option<Box<Expr>>),

    // for statement
    For(Vec<Pat>,Range,Block),

    // while let statement, should be converted to while statement
    WhileLet(Vec<Pat>,Box<Expr>,Block),

    // match statement, should be converted to if chain
    Match(Box<Expr>,Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)>),

    // unknown identifier referring to local or const, shoulbe converted to either
    UnknownLocalConst(String),

    // local reference
    Local(String),

    // const reference
    Const(String),

    // named tuple literal or function call, should be converted to either
    UnknownTupleFunctionCall(String,Vec<Expr>),

    // tuple literal, should be converted to struct literal
    Tuple(String,Vec<Expr>),

    // function call
    FunctionCall(String,Vec<Expr>),

    // unknown struct literal, should be converted to struct literal
    UnknownStruct(String,Vec<(String,Expr)>),

    // struct literal
    Struct(String,Vec<Expr>),

    // unknown variant, should be converted to variant
    UnknownVariant(String,String,VariantExpr),

    // variant literal
    Variant(String,usize,VariantExpr),

    // unknown method call, should be converted to method call
    UnknownMethodCall(Box<Expr>,String,Vec<Expr>),

    // method call, methods are only available from the standard library
    MethodCall(Box<Expr>,String,Vec<Expr>),

    // unknown struct field reference, should be converted to struct field reference
    UnknownField(Box<Expr>,String),

    // struct field reference
    Field(Box<Expr>,String,usize),

    // unknown tuple index reference, should be converted to tuple index reference
    UnknownTupleIndex(Box<Expr>,usize),

    // tuple index reference
    TupleIndex(Box<Expr>,String,usize),
}

// Stat; all statement nodes
#[derive(Clone)]
pub enum Stat {

    // let statement, should be destructured into local variables
    Let(Box<Pat>,Box<Type>,Box<Expr>),

    // local variable definition
    Local(String,Box<Type>,Box<Expr>),

    // void expression
    Expr(Box<Expr>),
}

// Local: local variable... not sure if we need this
#[derive(Clone)]
pub struct Local {
    pub ident: String,
    pub type_: Type,
}

// Method: standard library method definition
#[derive(Clone)]
pub struct Method {
    pub from_type: Type,
    pub ident: String,
    pub params: Vec<(String,Type)>,
    pub return_type: Type,
}

// Function: function definition
#[derive(Clone)]
pub struct Function {
    pub ident: String,
    pub params: Vec<(String,Type)>,
    pub return_type: Type,
    pub block: Block,
}

// Tuple; named tuple definition, should be converted to struct definitions
#[derive(Clone)]
pub struct Tuple {
    pub ident: String,
    pub types: Vec<Type>,
}

// Struct: struct definition
#[derive(Clone)]
pub struct Struct {
    pub ident: String,
    pub fields: Vec<(String,Type)>,
}

// Variant: enum variant
#[derive(Clone)]
pub enum Variant {
    Naked,
    Tuple(Vec<Type>),
    Struct(Vec<(String,Type)>),
}

// Enum: enum definition, should be converted to struct definition
#[derive(Clone)]
pub struct Enum {
    pub ident: String,
    pub variants: Vec<(String,Variant)>,
}

// Const: constant definition
#[derive(Clone)]
pub struct Const {
    pub ident: String,
    pub type_: Type,
    pub expr: Expr,
}

// Alias: type alias, should be resolved into the final types
#[derive(Clone)]
pub struct Alias {
    pub ident: String,
    pub type_: Type,
}

// Module: the shader module
#[derive(Clone)]
pub struct Module {
    pub ident: String,
    pub tuples: HashMap<String,Tuple>,
    pub structs: HashMap<String,Struct>,
    pub tuple_structs: HashMap<String,Struct>,
    pub anon_tuple_structs: HashMap<String,Struct>,
    pub extern_structs: HashMap<String,Struct>,
    pub enums: HashMap<String,Enum>,
    pub aliases: HashMap<String,Alias>,
    pub consts: HashMap<String,Const>,
    pub functions: HashMap<String,Function>,
}
