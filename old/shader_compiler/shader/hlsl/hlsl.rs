use std::{
    rc::Rc,
    collections::HashMap,
};

pub enum Type {

}

pub enum Decl {

}

pub enum Stat {
    Compound(Vec<Stat>),
    Decl(Decl),
    Expr(Expr),
    If(Expr,Stat,Option<Stat>),
    Switch(Expr,Vec<Stat>),
    For(Decl,Expr,Stat,Vec<Stat>),
    While(Expr,Vec<Stat>>),
    Do(Vec<Stat>,Expr),
    Discard,
    Return(Option<Expr>),
    Break,
    Continue,
    Case(Expr),
    Default,
}

pub enum Expr {

}

pub struct Function {

}

pub struct Module {

}

/*
STDLIB:

 */