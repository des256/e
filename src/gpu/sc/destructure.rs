use {
    std::collections::HashMap,
    super::*,
};

// - destructure patterns

struct Context {
    consts: HashMap<&'static str,Const>,
    stdlib_consts: HashMap<&'static str,Const>,
}

impl Context {

    fn make_pat_bool(&self,pat: &Pat,scrut: &Expr) -> Result<Option<Expr>,String> {

        match pat {

            Pat::Wildcard | Pat::Rest => Ok(None),

            Pat::Boolean(value) => Ok(Some(
                if *value {
                    scrut.clone()
                }
                else {
                    Expr::Unary(UnaryOp::Not,Box::new(scrut.clone()))
                }
            )),

            Pat::Integer(value) => Ok(Some(
                Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Integer(*value)))
            )),

            Pat::Float(value) => Ok(Some(
                Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Float(*value)))
            )),

            Pat::AnonTuple(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr),));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Array(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(&pats[i],&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64))))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Range(pat_lo,pat_hi) => {
                if let Pat::Integer(value_lo) = **pat_lo {
                    if let Pat::Integer(value_hi) = **pat_hi {
                        Ok(Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Integer(value_lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Integer(value_hi)))),
                            )
                        ))
                    }
                    else {
                        Err("pattern range can only be integers or floats".to_string())
                    }
                }
                else if let Pat::Float(value_lo) = **pat_lo {
                    if let Pat::Float(value_hi) = **pat_hi {
                        Ok(Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Float(value_lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Float(value_hi)))),
                            )
                        ))
                    }
                    else {
                        Err("pattern range can only be integers or floats".to_string())
                    }
                }
                else {
                    Err("pattern range can only be integers or floats".to_string())
                }
            },

            // TODO: if identifier refers to a constant or a constant from stdlib, insert == node
            Pat::Ident(/* ident */_) => /*if self.consts.contains_key(ident) {
                Ok(Some(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Const(ident.clone())))))
            }
            else*/ {
                Ok(None)
            },

            Pat::Tuple(_,pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Struct(_,fields) => {
                let mut accum: Option<Expr> = None;
                for ident_pat in fields.iter() {
                    if let FieldPat::IdentPat(ident,pat) = ident_pat {
                        if let Some(expr) = self.make_pat_bool(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))? {
                            if let Some(accum_inner) = accum {
                                accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                            }
                            else {
                                accum = Some(expr);
                            }
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Variant(_,_,variant_pat) => {
                // TODO: variant_ident should be usize
                let mut accum = Expr::Discriminant(Box::new(scrut.clone()),/* TODO: variant_ident */0);
                match variant_pat {
                    VariantPat::Naked => { },
                    VariantPat::Tuple(pats) => {
                        for i in 0..pats.len() {
                            // TODO: variant_ident should be usize
                            if let Some(expr) = self.make_pat_bool(&pats[i],&Expr::DestructTuple(Box::new(scrut.clone()),/* TODO: variant_ident */0,i))? {
                                accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                            }
                        }
                    },
                    VariantPat::Struct(fields) => {
                        for field in fields.iter() {
                            if let FieldPat::IdentPat(_,pat) = field {
                                // TODO: variant_ident and ident should be usize
                                if let Some(expr) = self.make_pat_bool(pat,&Expr::DestructStruct(Box::new(scrut.clone()),/* TODO: variant_ident */0,/* TODO: ident */0))? {
                                    accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                                }
                            }
                        }
                    },
                }
                Ok(Some(accum))
            },
        }
    }

    fn make_pats_bool(&self,pats: &Vec<Pat>,local: &'static str) -> Result<Option<Expr>,String> {
        let mut accum: Option<Expr> = None;
        for pat in pats.iter() {
            if let Some(expr) = self.make_pat_bool(pat,&Expr::Ident(local))? {
                if let Some(accum_inner) = accum {
                    accum = Some(Expr::Binary(Box::new(accum_inner),BinaryOp::LogOr,Box::new(expr)));
                }
                else {
                    accum = Some(expr);
                }
            }
        }
        Ok(accum)
    }

    fn destructure_pat(&self,pat: &Pat,scrut: &Expr) -> Result<Vec<Stat>,String> {
        let mut stats: Vec<Stat> = Vec::new();
        match pat {
            // TODO: Pat::Const
            Pat::Wildcard | Pat::Rest | Pat::Boolean(_) | Pat::Integer(_) | Pat::Float(_) | Pat::Range(_,_) /*| Pat::Const(_)*/ => { },
            Pat::AnonTuple(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                }
            },
            Pat::Array(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(&pats[i],&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64))))?);
                }
            },
            Pat::Ident(ident) => stats.push(Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(scrut.clone()))),
            Pat::Tuple(_,pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                }
            },
            Pat::Struct(_,fields) => {
                for field in fields.iter() {
                    match field {
                        FieldPat::Wildcard | FieldPat::Rest => { },
                        FieldPat::Ident(ident) => stats.push(
                            Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(Expr::Field(Box::new(scrut.clone()),ident.clone())))
                        ),
                        FieldPat::IdentPat(ident,pat) => stats.append(
                            &mut self.destructure_pat(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))?
                        ),
                    }
                }
            },
            Pat::Variant(_,_,variant_pat) => match variant_pat {
                VariantPat::Naked => { },
                VariantPat::Tuple(pats) => {
                    for i in 0..pats.len() {
                        stats.append(&mut self.destructure_pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                    }
                },
                VariantPat::Struct(field_pats) => {
                    for field_pat in field_pats.iter() {
                        if let FieldPat::IdentPat(ident,pat) = field_pat {
                            stats.append(&mut self.destructure_pat(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))?);
                        }
                    }
                },
            },
        }
        Ok(stats)
    }

    fn destructure_pats(&self,pats: &Vec<Pat>,scrut: &'static str) -> Result<Vec<Stat>,String> {
        let mut stats: Vec<Stat> = Vec::new();
        for pat in pats.iter() {
            stats.append(&mut self.destructure_pat(pat,&Expr::Ident(scrut))?);
        }
        Ok(stats)
    }

    fn process_type(&self,type_: Type) -> Result<Type,String> {
        match type_ {
            Type::Array(type_,expr) => {
                let new_type = self.process_type(*type_)?;
                let new_expr = self.process_expr(*expr)?;
                Ok(Type::Array(Box::new(new_type),Box::new(new_expr)))
            },
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(self.process_type(type_.clone())?);
                }
                Ok(Type::AnonTuple(new_types))
            },
            _ => Ok(type_),
        }
    }

    fn process_block(&self,block: Block) -> Result<Block,String> {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {
                Stat::Let(pat,type_,expr) => {
                    let new_expr = self.process_expr(*expr.clone())?;
                    let expr = Expr::Cast(Box::new(new_expr),(*type_).clone());
                    new_stats.append(&mut self.destructure_pat(&pat,&expr)?);
                },
                Stat::Expr(expr) => {
                    let new_expr = self.process_expr(*expr.clone())?;
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },
                Stat::Local(ident,type_,expr) => {
                    let new_expr = self.process_expr(*expr.clone())?;
                    new_stats.push(Stat::Local(ident.clone(),(*type_).clone(),Box::new(new_expr)));
                },
            }
        }
        let new_expr = if let Some(expr) = block.expr {
            Some(Box::new(self.process_expr(*expr)?))
        }
        else {
            None
        };
        Ok(Block {
            stats: new_stats,
            expr: new_expr,
        })
    }

    fn process_expr(&self,expr: Expr) -> Result<Expr,String> {
        match expr {
            Expr::Boolean(value) => Ok(Expr::Boolean(value)),
            Expr::Integer(value) => Ok(Expr::Integer(value)),
            Expr::Float(value) => Ok(Expr::Float(value)),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr)?);
                }
                Ok(Expr::Array(new_exprs))
            },
            Expr::Cloned(value_expr,count_expr) => {
                let new_value_expr = self.process_expr(*value_expr)?;
                let new_count_expr = self.process_expr(*count_expr)?;
                Ok(Expr::Cloned(Box::new(new_value_expr),Box::new(new_count_expr)))
            },
            Expr::Index(array_expr,index_expr) => {
                let new_array_expr = self.process_expr(*array_expr)?;
                let new_index_expr = self.process_expr(*index_expr)?;
                Ok(Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr)))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.process_expr(*expr)?;
                let new_type = self.process_type(*type_)?;
                Ok(Expr::Cast(Box::new(new_expr),Box::new(new_type)))
            },
            Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr)?);
                }
                Ok(Expr::AnonTuple(new_exprs))
            },
            Expr::Unary(op,expr) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::Unary(op,Box::new(new_expr)))
            },
            Expr::Binary(expr1,op,expr2) => {
                let new_expr1 = self.process_expr(*expr1)?;
                let new_expr2 = self.process_expr(*expr2)?;
                Ok(Expr::Binary(Box::new(new_expr1),op,Box::new(new_expr2)))
            },
            Expr::Continue => Ok(Expr::Continue),
            Expr::Break(expr) => if let Some(expr) = expr {
                Ok(Expr::Break(Some(Box::new(self.process_expr(*expr)?))))
            }
            else {
                Ok(Expr::Break(None))
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                Ok(Expr::Return(Some(Box::new(self.process_expr(*expr)?))))
            }
            else {
                Ok(Expr::Return(None))
            },
            Expr::Block(block) => Ok(Expr::Block(self.process_block(block)?)),
            Expr::If(cond_expr,block,else_expr) => {
                let new_cond_expr = self.process_expr(*cond_expr)?;
                let new_block = self.process_block(block)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.process_expr(*else_expr)?))
                }
                else {
                    None
                };
                Ok(Expr::If(Box::new(new_cond_expr),new_block,new_else_expr))
            },
            Expr::While(cond_expr,block) => {
                let new_cond_expr = self.process_expr(*cond_expr)?;
                let new_block = self.process_block(block)?;
                Ok(Expr::While(Box::new(new_cond_expr),new_block))
            },
            Expr::Loop(block) => Ok(Expr::Loop(self.process_block(block)?)),
            Expr::IfLet(pats,expr,block,else_expr) => {
                let new_expr = self.process_expr(*expr)?;
                let mut new_block = self.process_block(block)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.process_expr(*else_expr)?))
                }
                else {
                    None
                };
                let mut if_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let condition = self.make_pats_bool(&pats,"scrut")?;
                if condition.is_none() {
                    return Err("unable to create boolean condition from if-let patterns".to_string());
                }
                let condition = condition.unwrap();
                let mut then_block = Block {
                    stats: self.destructure_pats(&pats,"scrut")?,
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                if_block.expr = Some(Box::new(Expr::If(Box::new(condition),then_block,new_else_expr)));
                Ok(Expr::Block(if_block))
            },
            Expr::For(pats,range,block) => {
                if pats.len() == 1 {
                    if let Pat::Ident(ident) = &pats[0] {
                        let (expr,op,expr2) = match range {
                            Range::FromTo(expr1,expr2) => (self.process_expr(*expr1)?,BinaryOp::Less,self.process_expr(*expr2)?),
                            Range::FromToIncl(expr1,expr2) => (self.process_expr(*expr1)?,BinaryOp::LessEq,self.process_expr(*expr2)?),
                            _ => { return Err("for range can only be .. or ..=".to_string()) },
                        };
                        let mut stats = block.stats.clone();
                        if let Some(expr) = block.expr {
                            stats.push(Stat::Expr(expr));
                        }
                        let loop_block = Block {
                            stats,
                            expr: Some(Box::new(Expr::Binary(Box::new(Expr::Ident(ident)),BinaryOp::AddAssign,Box::new(Expr::Integer(1))))),
                        };
                        let for_block = Block {
                            stats: vec![
                                Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(expr)),
                            ],
                            expr: Some(Box::new(Expr::While(Box::new(Expr::Binary(Box::new(Expr::Ident(ident)),op,Box::new(expr2))),loop_block))),
                        };
                        Ok(Expr::Block(for_block))
                    }
                    else {
                        Err("for-pattern can only be a single variable".to_string())
                    }
                }
                else {
                    Err("for-pattern can only be a single variable".to_string())
                }
            },
            Expr::WhileLet(pats,expr,block) => {
                let new_expr = self.process_expr(*expr)?;
                let mut new_block = self.process_block(block)?;
                let mut while_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let condition = self.make_pats_bool(&pats,"scrut")?;
                if condition.is_none() {
                    return Err("unable to create boolean condition from while-let patterns".to_string());
                }
                let condition = condition.unwrap();
                let mut then_block = Block {
                    stats: self.destructure_pats(&pats,"scrut")?,
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                while_block.expr = Some(Box::new(Expr::While(Box::new(condition),then_block)));
                Ok(Expr::Block(while_block))
            },
            Expr::Match(expr,arms) => {
                let new_expr = self.process_expr(*expr)?;
                let mut match_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let mut exprs: Vec<Expr> = Vec::new();
                let mut else_expr: Option<Box<Expr>> = None;
                for (pats,_,expr) in arms {
                    // TODO: consider if_expr
                    let new_expr = self.process_expr(*expr)?;
                    if let Some(condition) = self.make_pats_bool(&pats,"scrut")? {
                        let arm_block = Block {
                            stats: self.destructure_pats(&pats,"scrut")?,
                            expr: Some(Box::new(new_expr)),
                        };
                        exprs.push(Expr::If(Box::new(condition),arm_block,None));
                    }
                    else if let None = else_expr {
                        let arm_block = Block {
                            stats: Vec::new(),
                            expr: Some(Box::new(new_expr)),
                        };
                        else_expr = Some(Box::new(Expr::Block(arm_block)));
                    }
                    else {
                        return Err("match expression can only have one wildcard".to_string());
                    }
                }
                let mut result_expr: Option<Box<Expr>> = else_expr;
                for i in 0..exprs.len() {
                    if let Expr::If(condition,block,_) = &exprs[exprs.len() - i - 1] {
                        result_expr = Some(Box::new(Expr::If(condition.clone(),block.clone(),result_expr)));
                    }
                }
                match_block.expr = Some(Box::new(*result_expr.unwrap()));
                Ok(Expr::Block(match_block))
            },
            Expr::Ident(ident) => Ok(Expr::Ident(ident)),
            Expr::TupleOrCall(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.process_expr(expr.clone())?);
                }
                Ok(Expr::TupleOrCall(ident,new_exprs))
            },
            Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(&'static str,Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.process_expr(expr)?));
                }
                Ok(Expr::Struct(ident,new_fields))
            },
            Expr::Variant(enum_ident,variant_ident,variant_expr) => {
                let new_variant_expr = match variant_expr {
                    VariantExpr::Naked => VariantExpr::Naked,
                    VariantExpr::Tuple(exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(self.process_expr(expr.clone())?);
                        }
                        VariantExpr::Tuple(new_exprs)
                    },
                    VariantExpr::Struct(fields) => {
                        let mut new_fields: Vec<(&'static str,Expr)> = Vec::new();
                        for (ident,expr) in fields {
                            new_fields.push((ident,self.process_expr(expr)?));
                        }
                        VariantExpr::Struct(new_fields)
                    },
                };
                Ok(Expr::Variant(enum_ident,variant_ident,new_variant_expr))
            },
            Expr::Method(from_expr,ident,exprs) => {
                let new_from_expr = self.process_expr(*from_expr)?;
                let mut new_exprs: Vec<Expr> =Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr)?);
                }
                Ok(Expr::Method(Box::new(new_from_expr),ident,new_exprs))
            },
            Expr::Field(expr,ident) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::Field(Box::new(new_expr),ident))
            },
            Expr::TupleIndex(expr,index) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::TupleIndex(Box::new(new_expr),index))
            },
        
            Expr::Discriminant(expr,index) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::Discriminant(Box::new(new_expr),index))
            },
            Expr::DestructTuple(expr,variant_index,index) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::DestructTuple(Box::new(new_expr),variant_index,index))
            },
            Expr::DestructStruct(expr,variant_index,index) => {
                let new_expr = self.process_expr(*expr)?;
                Ok(Expr::DestructStruct(Box::new(new_expr),variant_index,index))
            },
        }
    }

    pub fn process_module(module: Module) -> Result<Module,String> {
        let stdlib = StandardLib::new();
        let context = Context {
            consts: module.consts.clone(),
            stdlib_consts: stdlib.consts.clone(),
        };
        let mut new_structs: HashMap<&'static str,Struct> = HashMap::new();
        for struct_ in module.structs.values() {
            let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push((field.0,context.process_type(field.1.clone())?));
            }
            new_structs.insert(struct_.ident,Struct { ident: struct_.ident,fields: new_fields, });
        }
        let mut new_extern_structs: HashMap<&'static str,Struct> = HashMap::new();
        for struct_ in module.extern_structs.values() {
            let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push((field.0,context.process_type(field.1.clone())?));
            }
            new_extern_structs.insert(struct_.ident,Struct { ident: struct_.ident,fields: new_fields, });
        }
        let mut new_enums: HashMap<&'static str,Enum> = HashMap::new();
        for enum_ in module.enums.values() {
            let mut new_variants: Vec<(&'static str,Variant)> = Vec::new();
            for (variant_ident,variant) in enum_.variants.iter() {
                match variant {
                    Variant::Naked => new_variants.push((variant_ident,Variant::Naked)),
                    Variant::Tuple(types) => {
                        let mut new_types: Vec<Type> = Vec::new();
                        for type_ in types.iter() {
                            new_types.push(context.process_type(type_.clone())?);
                        }
                        new_variants.push((variant_ident,Variant::Tuple(new_types)));
                    },
                    Variant::Struct(fields) => {
                        let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
                        for (ident,type_) in fields.iter() {
                            new_fields.push((ident,context.process_type(type_.clone())?));
                        }
                        new_variants.push((variant_ident,Variant::Struct(new_fields)));
                    }
                }   
            }
            new_enums.insert(enum_.ident,Enum { ident: enum_.ident,variants: new_variants,});
        }
        let mut new_aliases: HashMap<&'static str,Alias> = HashMap::new();
        for alias in module.aliases.values() {
            let new_type = context.process_type(alias.type_.clone())?;
            new_aliases.insert(alias.ident,Alias { ident: alias.ident,type_: new_type, });
        }
        let mut new_consts: HashMap<&'static str,Const> = HashMap::new();
        for const_ in module.consts.values() {
            let new_expr = context.process_expr(const_.expr.clone())?;
            let new_type = context.process_type(const_.type_.clone())?;
            new_consts.insert(const_.ident,Const { ident: const_.ident,type_: new_type,expr: new_expr, });
        }
        let mut new_functions: HashMap<&'static str,Function> = HashMap::new();
        for function in module.functions.values() {
            let mut new_params: Vec<(&'static str,Type)> = Vec::new();
            for param in function.params.iter() {
                new_params.push((param.0,context.process_type(param.1.clone())?));
            }
            let new_return_type = context.process_type(function.return_type.clone())?;
            let new_block = context.process_block(function.block.clone())?;
            new_functions.insert(function.ident,Function {
                ident: function.ident,
                params: new_params,
                return_type: new_return_type,
                block: new_block,
            });
        }
        Ok(Module {
            ident: module.ident,
            structs: new_structs,
            extern_structs: new_extern_structs,
            enums: new_enums,
            aliases: new_aliases,
            consts: new_consts,
            functions: new_functions,

            anon_tuple_structs: Vec::new(),
        })
    }
}

pub fn destructure_module(module: Module) -> Result<Module,String> {
    Context::process_module(module)
}
