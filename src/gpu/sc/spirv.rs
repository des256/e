use super::*;

struct Context {
    stdlib: StandardLib,
    module: ProcessedModule,
}

impl Context {

    fn emit_type(&self,type_: &Type) -> Result<Vec<u8>,String> {
        Err("TODO".to_string())
    }
    fn emit_expr(&self,expr: &Expr) -> Result<Vec<u8>,String> {
        Err("TODO".to_string())
    }

    fn emit_block(&self,block: &Block) -> Result<Vec<u8>,String> {
        Err("TODO".to_string())
    }

    fn emit_function(&self,function: &Function) -> Result<Vec<u8>,String> {
        Err("TODO".to_string())
    }
}

pub fn emit_module(module: &ProcessedModule) -> Result<Vec<u8>,String> {

    let mut context = Context {
        stdlib: StandardLib::new(),
        module: module.clone(),
    };

    Err("TODO".to_string())
}
