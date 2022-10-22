use {
    std::{
        collections::HashMap,
        cmp::PartialEq,
    },
};

#[derive(Clone)]
pub enum Type {
    Inferred,
    Void,
    Integer,
    Float,
    Bool,
    U8,I8,U16,I16,U32,I32,U64,I64,USize,ISize,
    F16,F32,F64,
    AnonTuple(Vec<Type>),
    Array(Box<Type>,Box<Expr>),
    UnknownIdent(String),
    Tuple(String),
    Struct(String),
    Enum(String),
    Alias(String),
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
                tuple == other_tuple
            },
            (Type::Struct(struct_),Type::Struct(other_struct)) => {
                struct_ == other_struct
            },
            (Type::Enum(enum_),Type::Enum(other_enum)) => {
                enum_ == other_enum
            },
            (Type::Alias(alias),Type::Alias(other_alias)) => {
                alias == other_alias
            },
            _ => false,
        }
    }
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
    Naked(String),
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<FieldPat>),
}

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
    Tuple(String,Vec<Pat>),
    Struct(String,Vec<FieldPat>),
    Variant(String,VariantPat),
}

#[derive(Clone)]
pub enum VariantExpr {
    Naked(String),
    Tuple(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
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
    TupleOrCall(String,Vec<Expr>),
    Struct(String,Vec<(String,Expr)>),
    Variant(String,VariantExpr),
    Method(Box<Expr>,String,Vec<Expr>),
    Field(Box<Expr>,String),
    TupleIndex(Box<Expr>,usize),
    Param(String),
    Local(String),
    Const(String),
    Tuple(String,Vec<Expr>),
    Call(String,Vec<Expr>),
    Discriminant(Box<Expr>),
    Destructure(Box<Expr>,usize,usize),
}

#[derive(Clone)]
pub enum Stat {
    Let(Box<Pat>,Box<Type>,Box<Expr>),
    Expr(Box<Expr>),
    Local(String,Box<Type>,Box<Expr>),
}

#[derive(Clone)]
pub struct Symbol {
    pub ident: String,
    pub type_: Type,
}

#[derive(Clone)]
pub struct Method {
    pub from_type: Type,
    pub ident: String,
    pub params: Vec<Symbol>,
    pub type_: Type,
}

#[derive(Clone)]
pub struct Function {
    pub ident: String,
    pub params: Vec<Symbol>,
    pub type_: Type,
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

#[derive(Clone)]
pub enum Variant {
    Naked(String),
    Tuple(String,Vec<Type>),
    Struct(String,Vec<Symbol>),
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

#[derive(Clone)]
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
    pub tuples: HashMap<String,Tuple>,
    pub structs: HashMap<String,Struct>,
    pub enums: HashMap<String,Enum>,
    pub aliases: HashMap<String,Alias>,
    pub consts: HashMap<String,Const>,
    pub functions: HashMap<String,Function>,
}

#[derive(Clone)]
pub struct Source {
    pub ident: String,
    pub tuples: Vec<Tuple>,
    pub structs: Vec<Struct>,
    pub extern_structs: Vec<Struct>,
    pub enums: Vec<Enum>,
    pub aliases: Vec<Alias>,
    pub consts: Vec<Const>,
    pub functions: Vec<Function>,
}
