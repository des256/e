struct Context {
    stdlib: StandardLib,
    module: PreparedModule,
}

impl Context {

    emit_expr(&self,expr: &Expr) -> Result<Vec<u8>,String> {

    }

    emit_block(&self,block: &Block) -> Result<Vec<u8>,String> {

    }

    emit_function(&self,function: &Function) -> Result<Vec<u8>,String> {

    }
}

pub fn emit_module(module: &PreparedModule) -> Result<Vec<u8>,String> {

    let mut context = Context {
        stdlib: StandardLib::new(),
        module: module.clone(),
    }

}
