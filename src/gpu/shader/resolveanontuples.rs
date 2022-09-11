/*
// force function return types into the code, resolving types that are too loose, and get rid of anonymous tuple literals

use {
    crate::*,
    std::{
        rc::Rc,
        collections::HashMap,
    },
};

fn resolve_stat(stat: &mut sr::Stat) {
    match stat {
        sr::Stat::Let(variable) => {
            // remove anonymous tuples 
        }
    }
    // - anonymous tuples as function parameters
    // - anonymous tuples as let expressions
    // - anonymous tuple patterns
}

fn resolve_block(block: &mut sr::Block,type_: &sr::Type) {
    for stat in block.stats.iter_mut() {
        resolve_stat(stat);
    }
    if let Some(expr) = &mut block.expr {

        // remove anonymous tuples as block expression
        if let sr::Expr::AnonTuple(exprs) = expr.as_mut() {
            if let sr::Type::Struct(struct_) = &type_ {
                if exprs.len() == struct_.fields.len() {
                    let mut fields: Vec<(String,sr::Expr)> = Vec::new();
                    let mut i = 0usize;
                    for expr in exprs.iter() {
                        if let None = find_tightest_type(&struct_.fields[i].type_,&infer_expr_type(expr)) {
                            panic!("struct {}.{} not compatible with {}",struct_.ident,struct_.fields[i].ident,expr);
                        }
                        fields.push((struct_.fields[i].ident.clone(),expr.clone()));
                        i += 1;
                    }
                    *expr = Box::new(sr::Expr::Struct(Rc::clone(&struct_),fields));
                }
                else {
                    panic!("struct {} not compatible with anonymous tuple literal {}",struct_.ident,expr);
                }
            }
        }
    }
}

pub fn resolve_loose_types(module: &mut sr::Module) {
    let mut new_functions: HashMap<String,Rc<sr::Function>> = HashMap::new();
    for (_,function) in module.functions.iter() {
        let mut new_block = function.block.clone();
        resolve_block(&mut new_block,&function.return_type);
        new_functions.insert(function.ident.clone(),Rc::new(sr::Function { ident: function.ident.clone(),params: function.params.clone(),return_type: function.return_type.clone(),block: new_block, }));
    }
    module.functions = new_functions;
}
*/