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
        // TODO: AnonTuple => Struct
        // TODO: Tuple => Struct
        type_
    }

    pub fn detuplify_pat(&mut self,pat: Pat) -> Pat {
        // TODO: AnonTuple => Struct
        // TODO: Tuple => Struct
        pat
    }

    pub fn detuplify_expr(&mut self,expr: Expr) -> Expr {
        // TODO: AnonTuple => Struct
        // TODO: Tuple => Struct
        expr
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
