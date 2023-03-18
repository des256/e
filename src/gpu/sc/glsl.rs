struct Context {
    stdlib: StandardLib,
    module: PreparedModule,
}

impl Context {

    emit_expr(&self,expr: &Expr) -> Result<String,String> {

    }

    emit_block(&self,block: &Block) -> Result<String,String> {

    }

    emit_function(&self,function: &Function) -> Result<String,String> {

    }
}

pub fn emit_module(module: &PreparedModule) -> Result<String,String> {

    let mut context = Context {
        stdlib: StandardLib::new(),
        module: module.clone(),
    }

}