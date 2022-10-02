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

pub struct Disenumifier { }

impl Disenumifier {
    pub fn new() -> Disenumifier {
        Disenumifier { }
    }

    pub fn disenumify_type(&mut self,type_: Type) -> Type {
        // TODO: Enum => Struct
        type_
    }

    pub fn disenumify_expr(&mut self,expr: Expr) -> Expr {
        // TODO: Variant => Field
        expr
    }

    pub fn disenumify_range(&mut self,range: Range) -> Range {
        // TODO: pass down
        range
    }

    pub fn disenumify_stat(&mut self,stat: Stat) -> Stat {
        // TODO: pass down
        stat
    }

    pub fn disenumify_block(&mut self,block: Block) -> Block {
        // TODO: pass down
        block
    }

    pub fn disenumify_module(&mut self,module: Module) -> Module {
        // TODO: map enums to structs
        module
    }
}
