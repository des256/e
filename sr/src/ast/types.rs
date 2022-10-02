use {
    std::{
        collections::HashMap,
        rc::Rc,
        cell::RefCell,
        cmp::PartialEq,
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
    Tuple(Rc<RefCell<Tuple>>),
    Struct(Rc<RefCell<Struct>>),
    Enum(Rc<RefCell<Enum>>),
    Alias(Rc<RefCell<Alias>>),
}

impl PartialEq for Type {
    fn eq(&self,other: &Self) -> bool {
        match (self,other) {
            (Type::Inferred,Type::Inferred) => true,
            (Type::Void,Type::Void) => true,
            (Type::Integer,Type::Integer) => true,
            (Type::Float,Type::Float) => true,
            (Type::Bool,Type::Bool) => true,
            (Type::U8,Type::U8) => true,
            (Type::I8,Type::I8) => true,
            (Type::U16,Type::U16) => true,
            (Type::I16,Type::I16) => true,
            (Type::U32,Type::U32) => true,
            (Type::I32,Type::I32) => true,
            (Type::U64,Type::U64) => true,
            (Type::I64,Type::I64) => true,
            (Type::USize,Type::USize) => true,
            (Type::ISize,Type::ISize) => true,
            (Type::F16,Type::F16) => true,
            (Type::F32,Type::F32) => true,
            (Type::F64,Type::F64) => true,
            (Type::AnonTuple(types),Type::AnonTuple(other_types)) => {
                if types.len() == other_types.len() {
                    let mut equal = true;
                    for i in 0..types.len() {
                        if types[i] != other_types[i] {
                            equal = false;
                            break;
                        }
                    }
                    equal
                }
                else {
                    false
                }
            },
            (Type::Array(type_,expr),Type::Array(other_type,other_expr)) => {
                if type_ == other_type {
                    if let Expr::Integer(size) = **expr {
                        if let Expr::Integer(other_size) = **other_expr {
                            size == other_size
                        }
                        else {
                            false
                        }
                    }
                    else {
                        false
                    }
                }
                else {
                    false
                }
            },
            (Type::UnknownIdent(ident),Type::UnknownIdent(other_ident)) => {
                ident == other_ident
            },
            (Type::Tuple(tuple),Type::Tuple(other_tuple)) => {
                tuple.borrow().ident == other_tuple.borrow().ident
            },
            (Type::Struct(struct_),Type::Struct(other_struct)) => {
                struct_.borrow().ident == other_struct.borrow().ident
            },
            (Type::Enum(enum_),Type::Enum(other_enum)) => {
                enum_.borrow().ident == other_enum.borrow().ident
            },
            (Type::Alias(alias),Type::Alias(other_alias)) => {
                alias.borrow().ident == other_alias.borrow().ident
            },
            _ => false,
        }
    }
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
    Tuple(Rc<RefCell<Tuple>>,Vec<Pat>),
    Struct(Rc<RefCell<Struct>>,Vec<FieldPat>),
    Variant(Rc<RefCell<Enum>>,VariantPat),
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
    Param(Rc<RefCell<Symbol>>),
    Local(Rc<RefCell<Symbol>>),
    Const(Rc<RefCell<Const>>),
    Tuple(Rc<RefCell<Tuple>>,Vec<Expr>),
    Call(Rc<RefCell<Function>>,Vec<Expr>),
    Struct(Rc<RefCell<Struct>>,Vec<Expr>),
    Variant(Rc<RefCell<Enum>>,VariantExpr),
    Method(Box<Expr>,Rc<RefCell<Method>>,Vec<Expr>),
    Field(Rc<RefCell<Struct>>,Box<Expr>,usize),
    TupleIndex(Rc<RefCell<Tuple>>,Box<Expr>,usize),
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

#[derive(Clone)]
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
    pub params: Vec<Rc<RefCell<Symbol>>>,
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

impl Variant {
    pub fn find_struct_element(&self,ident: &str) -> Option<usize> {
        if let Variant::Struct(_,symbols) = self {
            for i in 0..symbols.len() {
                if symbols[i].ident == ident {
                    return Some(i);
                }
            }
        }
        None
    }
}

// enum { variant, ..., variant, }
pub struct Enum {
    pub ident: String,
    pub variants: Vec<Variant>,
}

impl Enum {
    pub fn find_naked_variant(&self,ident: &str) -> Option<usize> {
        for i in 0..self.variants.len() {
            if let Variant::Naked(variant_ident) = &self.variants[i] {
                if ident == variant_ident {
                    return Some(i);
                }
            }
        }
        None
    }

    pub fn find_tuple_variant(&self,ident: &str) -> Option<usize> {
        for i in 0..self.variants.len() {
            if let Variant::Tuple(variant_ident,_) = &self.variants[i] {
                if ident == variant_ident {
                    return Some(i);
                }
            }
        }
        None
    }

    pub fn find_struct_variant(&self,ident: &str) -> Option<usize> {
        for i in 0..self.variants.len() {
            if let Variant::Struct(variant_ident,_) = &self.variants[i] {
                if ident == variant_ident {
                    return Some(i);
                }
            }
        }
        None
    }
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
    pub tuples: HashMap<String,Rc<RefCell<Tuple>>>,
    pub structs: HashMap<String,Rc<RefCell<Struct>>>,
    pub enums: HashMap<String,Rc<RefCell<Enum>>>,
    pub aliases: HashMap<String,Rc<RefCell<Alias>>>,
    pub consts: HashMap<String,Rc<RefCell<Const>>>,
    pub functions: HashMap<String,Rc<RefCell<Function>>>,
}
