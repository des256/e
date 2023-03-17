use {
    std::collections::HashMap,
    super::*,
};

struct Context {
    stdlib: StandardLib,
    module: DestructuredModule,
    enum_tuples: Vec<Tuple>,
    enum_mappings: Vec<Vec<Vec<usize>>>,
}

impl Context {

    fn deenumify_expr(&self,expr: &Expr) -> Result<Expr,String> {
        match expr {
            Expr::Boolean(value) => Ok(Expr::Boolean(*value)),
            Expr::Integer(value) => Ok(Expr::Integer(*value)),
            Expr::Float(value) => Ok(Expr::Float(*value)),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::Array(new_exprs))
            },
            Expr::Cloned(expr,count) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::Cloned(Box::new(new_expr),*count))
            },
            Expr::Index(expr,index_expr) => {
                let new_expr = self.deenumify_expr(expr)?;
                let new_index_expr = self.deenumify_expr(index_expr)?;
                Ok(Expr::Index(Box::new(new_expr),Box::new(new_index_expr)))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::Cast(Box::new(new_expr),type_.clone()))
            },
            Expr::AnonTuple(_) => Err("Expr::AnonTuple shouldn't exist in deenumify pass".to_string()),
            Expr::Unary(op,expr) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::Unary(op.clone(),Box::new(new_expr)))
            },
            Expr::Binary(expr1,op,expr2) => {
                let new_expr1 = self.deenumify_expr(expr1)?;
                let new_expr2 = self.deenumify_expr(expr2)?;
                Ok(Expr::Binary(Box::new(new_expr1),op.clone(),Box::new(new_expr2)))
            },
            Expr::Continue => Ok(Expr::Continue),
            Expr::Break(expr) => {
                if let Some(expr) = expr {
                    Ok(Expr::Break(Some(Box::new(self.deenumify_expr(expr)?))))
                }
                else {
                    Ok(Expr::Break(None))
                }
            },
            Expr::Return(expr) => {
                if let Some(expr) = expr {
                    Ok(Expr::Return(Some(Box::new(self.deenumify_expr(expr)?))))
                }
                else {
                    Ok(Expr::Break(None))
                }
            },
            Expr::Block(block) => {
                let new_block = self.deenumify_block(block)?;
                Ok(Expr::Block(new_block))
            },
            Expr::If(cond_expr,block,else_expr) => {
                let new_cond_expr = self.deenumify_expr(cond_expr)?;
                let new_block = self.deenumify_block(block)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.deenumify_expr(else_expr)?))
                }
                else {
                    None
                };
                Ok(Expr::If(Box::new(new_cond_expr),new_block,new_else_expr))
            },
            Expr::While(cond_expr,block) => {
                let new_cond_expr = self.deenumify_expr(cond_expr)?;
                let new_block = self.deenumify_block(block)?;
                Ok(Expr::While(Box::new(new_cond_expr),new_block))
            },
            Expr::Loop(block) => {
                let new_block = self.deenumify_block(block)?;
                Ok(Expr::Loop(new_block))
            },
            Expr::IfLet(_,_,_,_) => Err("Expr::IfLet should not exist in deenumify pass".to_string()),
            Expr::For(_,_,_) => Err("Expr::For should not exist in deenumify pass".to_string()),
            Expr::WhileLet(_,_,_) => Err("Exor::WhileLet should not exist in deenumify pass".to_string()),
            Expr::Match(_,_) => Err("Expr::Match should not exist in deenumify pass".to_string()),
            Expr::Ident(_) => Err("Expr::Ident should not exist in deenumify pass".to_string()),
            Expr::TupleOrFunction(_,_) => Err("Expr::TupleOrFunction should not exist in deenumify pass".to_string()),
            Expr::Struct(_,_) => Err("Expr::Struct should not exist in deenumify pass".to_string()),
            Expr::Variant(ident,variant_ident,variant_expr) => Ok(Expr::Variant(ident,variant_ident,variant_expr.clone())),
            Expr::MethodRef(from_expr,ident,exprs) => {
                let new_from_expr = self.deenumify_expr(from_expr)?;
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::MethodRef(Box::new(new_from_expr),ident,new_exprs))
            },
            Expr::Field(expr,ident) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::Field(Box::new(new_expr),ident))
            },
            Expr::TupleIndex(expr,index) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::TupleIndex(Box::new(new_expr),*index))
            },
            Expr::StructRef(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::StructRef(ident,new_exprs))
            },
            Expr::TupleRef(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::TupleRef(ident,new_exprs))
            },
            Expr::FunctionRef(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::FunctionRef(ident,new_exprs))
            },
            Expr::AnonTupleRef(index,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.deenumify_expr(expr)?);
                }
                Ok(Expr::AnonTupleRef(*index,new_exprs))
            },
            Expr::ConstRef(ident) => Ok(Expr::ConstRef(ident)),
            Expr::LocalOrParamRef(ident) => Ok(Expr::LocalOrParamRef(ident)),
            Expr::Discriminant(expr,index) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::Binary(Box::new(Expr::Field(Box::new(new_expr),"discr")),BinaryOp::Eq,Box::new(Expr::Integer(*index as i64))))
            },
            Expr::DestructTuple(expr,variant_index,index) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::DestructTuple(Box::new(new_expr),*variant_index,*index))
            },
            Expr::DestructStruct(expr,variant_index,index) => {
                let new_expr = self.deenumify_expr(expr)?;
                Ok(Expr::DestructStruct(Box::new(new_expr),*variant_index,*index))
            },
        }
    }

    fn deenumify_block(&self,block: &Block) -> Result<Block,String> {

    }
}

pub fn deenumify_module(module: &DestructuredModule) -> Result<DeenumifiedModule,String> {
    let mut context = Context {
        stdlib: StandardLib::new(),
        module: (*module).clone(),
        enum_tuples: Vec::new(),
        enum_mappings: Vec::new(),
    };

    for enum_ in module.enums.iter() {

        // this is the merged total type count for all variants
        let mut total_type_counts: HashMap<Type,usize> = HashMap::new();

        for (_,variant) in enum_.variants {

            // calculate how many components of fields of each type there are in this variant
            let mut type_counts: HashMap<Type,usize> = HashMap::new();

            match variant {
                Variant::Naked => { },
                Variant::Tuple(types) => {
                    for type_ in types {
                        if type_counts.contains_key(&type_) {
                            type_counts[&type_] += 1;
                        }
                        else {
                            type_counts.insert(type_,1);
                        }
                    }
                },
                Variant::Struct(fields) => {
                    for (_,type_) in fields {
                        if type_counts.contains_key(&type_) {
                            type_counts[&type_] += 1;
                        }
                        else {
                            type_counts.insert(type_,1);
                        }
                    }
                },
            }

            // merge into total list
            for (type_,count) in type_counts.iter() {
                if total_type_counts.contains_key(type_) {
                    if total_type_counts[type_] < *count {
                        total_type_counts[type_] = *count;
                    }
                }
                else {
                    total_type_counts.insert(type_.clone(),*count);
                }
            }
        }

        // now we know how many items of specific types we have, so start building the tuples
        let mut new_types: Vec<Type> = Vec::new();
        for (type_,count) in total_type_counts.iter() {
            for _ in 0..*count {
                new_types.push(type_.clone());
            }
        }

        // calculate the mapping for each variant
        let mut mappings: Vec<Vec<usize>> = Vec::new();
        for (variant_ident,variant) in enum_.variants {
            match variant {
                Variant::Naked => {
                    mappings.push(Vec::new());
                },
                Variant::Tuple(types) => {
                    let mut mapping: Vec<usize> = Vec::new();
                    let comp = 0usize;
                    for type_ in types {
                        let mut found_slot: Option<usize> = None;
                        for i in 0..new_types.len() {
                            if type_ == new_types[i] {
                                let mut already_assigned = false;
                                for k in 0..mapping.len() {
                                    if mapping[k] == i {
                                        already_assigned = true;
                                        break;
                                    }
                                }
                                if already_assigned {
                                    break;
                                }
                                found_slot = Some(i);
                                break;
                            }
                        }
                        if let Some(index) = found_slot {
                            mapping.push(index);
                        }
                        else {
                            return Err(format!("Unable to find slot for {}::{}.{}",enum_.ident,variant_ident,comp));
                        }
                        comp += 1;
                    }
                    mappings.push(mapping);
                },
                Variant::Struct(fields) => {
                    let mut mapping: Vec<usize> = Vec::new();
                    for (ident,type_) in fields {
                        let mut found_slot: Option<usize> = None;
                        for i in 0..new_types.len() {
                            if type_ == new_types[i] {
                                let mut already_assigned = false;
                                for k in 0..mapping.len() {
                                    if mapping[k] == i {
                                        already_assigned = true;
                                        break;
                                    }
                                }
                                if already_assigned {
                                    break;
                                }
                                found_slot = Some(i);
                                break;
                            }
                        }
                        if let Some(index) = found_slot {
                            mapping.push(index);
                        }
                        else {
                            return Err(format!("Unable to find slot for {}::{}.{}",enum_.ident,variant_ident,ident));
                        }
                    }
                    mappings.push(mapping);
                },
            }
        }
        context.enum_tuples.push(Tuple { ident: enum_.ident.clone(),types: new_types, });
        context.enum_mappings.push(mappings);
    }

    let mut new_consts: Vec<Const> = Vec::new();
    for const_ in context.module.consts {
        let new_expr = context.deenumify_expr(&const_.expr)?;
        new_consts.push(Const { ident: const_.ident.clone(),type_: const_.type_.clone(),expr: new_expr, });
    }

    let mut new_functions: Vec<Function> = Vec::new();
    for function in context.module.functions {
        let new_block = context.deenumify_block(&function.block)?;
        new_functions.push(Function { ident: function.ident.clone(),params: function.params.clone(),return_type: function.return_type.clone(),block: new_block, });
    }

    Ok(DeenumifiedModule {
        ident: context.module.ident,
        tuples: context.module.tuples.clone(),
        structs: context.module.structs.clone(),
        extern_structs: context.module.extern_structs.clone(),
        enum_tuples: context.enum_tuples,
        enum_mappings: context.enum_mappings,
        consts: new_consts,
        functions: new_functions,
        anon_tuple_types: context.module.anon_tuple_types.clone(),
    })
}
