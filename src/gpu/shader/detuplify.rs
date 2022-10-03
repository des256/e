use {
    crate::*,
    sr::*,
    std::{
        collections::HashMap,
        rc::Rc,
        cell::RefCell,
    },
};

use ast::*;

pub struct Detuplifier {
    pub structs: HashMap<String,Rc<RefCell<Struct>>>,
}

impl Detuplifier {
    pub fn new() -> Detuplifier {
        Detuplifier { 
            structs: HashMap::new(),
        }
    }

    pub fn detuplify_type(&mut self,type_: Type) -> Type {
        match type_ {
            Type::Array(type_,expr) => {
                let new_type = detuplify_type(type_);
                let new_expr = detuplify_expr(expr);
                Type::Array(new_type,new_expr)
            },
            Type::AnonTuple(types) => {
                // TODO: find matching Struct in AnonTuple list
                // TODO: if not found, create new Struct
                // TODO: remap to Type::Struct
            },
            Type::Tuple(tuple) => {
                // TODO: find corresponding struct
                // TODO: remap to Type::Struct
            },
            _ => type_
        }
    }

    pub fn detuplify_pat(&mut self,pat: Pat,should_type: Type) -> Pat {
        match pat {
            Pat::AnonTuple(pats) => {
                // TODO: find matching Struct in AnonTuple list
                // TODO: if not found, create new
                // TODO: remap to Pat::Struct
                pat
            },
            Pat::Array(pats) => {
                // TODO: pass down
                pat
            },
            Pat::Range(lo,hi) => {
                // TODO: pass down
                pat
            },
            Pat::Tuple(tuple,pats) => {
                // TODO: find struct, if not found, create new
                // TODO: remap to Pat::Struct
                pat
            },
            Pat::Struct(struct_,fields) => {
                // TODO: pass down to fields
            },
            Pat::Variant(enum_,variant) => {
                // TODO: pass down to variant
            },
            _ => pat,
        }
    }

    pub fn detuplify_expr(&mut self,expr: Expr) -> Expr {
        match expr {
            Expr::Array(exprs) => {
                // TODO: pass down to exprs
                expr
            },
            Expr::Cloned(expr,count) => {
                // TODO: pass down to expr and expr2
                expr
            },
            Expr::Index(expr,index) => {
                // TODO: pass down to expr and index
                expr
            },
            Expr::Cast(expr,type_) => {
                // TODO: pass down to expr and type_
                expr
            },
            Expr::AnonTuple(exprs) => {
                // TODO: find matching struct in AnonTuple list
                // TODO: if not found, create new
                // TODO: remap to Expr::Struct
            },
            Expr::Unary(op,expr) => {
                // TODO: pass down to expr
            },
            Expr::Binary(expr,op,expr2) => {
                // TODO: pass down to expr and expr2
            },
            Expr::Break(expr) => {
                // TODO: pass down to expr
            },
            Expr::Return(expr) => {
                // TODO: pass down to expr
            },
            Expr::Block(block) => {
                // TODO: pass down to block
            },
            Expr::If
        }
    }

    pub fn detuplify_range(&mut self,range: Range) -> Range {
        // TODO: pass down
        range
    }

    pub fn detuplify_stat(&mut self,stat: Stat) -> Stat {
        // TODO: pass down
        stat
    }

    pub fn detuplify_block(&mut self,block: Block) -> Block {
        // TODO: pass down
        block
    }

    pub fn detuplify_module(&mut self,module: Module) -> Module {
        // TODO: pass down
        module
    }
}
