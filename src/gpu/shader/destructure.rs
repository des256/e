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

pub struct Destructurer { }

impl Destructurer {
    pub fn new() -> Destructurer {
        Destructurer { }
    }

    pub fn destructure_expr(&mut self,expr: Expr) -> Expr {
        // TODO: IfLet, For, WhileLet, Match:
        // TODO:     construct boolean refutability check
        // TODO:     extract destructuring Local nodes
        expr
    }

    pub fn destructure_range(&mut self,range: Range) -> Range {
        // TODO: pass down
        range
    }

    pub fn destructure_stat(&mut self,stat: Stat) -> Stat {
        // TODO: Let:
        // TODO:     extract destructuring Local nodes
        stat
    }

    pub fn destructure_block(&mut self,block: Block) -> Block {
        // TODO: pass down
        block
    }

    pub fn destructure_module(&mut self,module: Module) -> Module {
        module
    }
}
