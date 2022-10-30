use {
    super::*,
    std::collections::HashMap,
};

use ast::*;

struct Context {
    consts: HashMap<String,Const>,
    stdlib_consts: HashMap<String,Const>,
}

impl Context {

    fn make_pat_bool(&self,pat: &Pat,scrut: &Expr) -> Option<Expr> {

        match pat {

            Pat::Wildcard | Pat::Rest => None,
    
            Pat::Boolean(value) => Some(
                if *value {
                    scrut.clone()
                }
                else {
                    Expr::Unary(
                        UnaryOp::Not,
                        Box::new(scrut.clone())
                    )
                }
            ),
    
            Pat::Integer(value) => Some(
                Expr::Binary(
                    Box::new(scrut.clone()),
                    BinaryOp::Eq,
                    Box::new(Expr::Integer(*value))
                )
            ),
    
            Pat::Float(value) => Some(
                Expr::Binary(
                    Box::new(scrut.clone()),
                    BinaryOp::Eq,
                    Box::new(Expr::Float(*value))
                )
            ),
    
            Pat::AnonTuple(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(
                        &pats[i],
                        &Expr::TupleIndex(
                            Box::new(scrut.clone()),
                            i
                        ),
                    ) {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(
                                Box::new(accum_inner.clone()),
                                BinaryOp::LogAnd,
                                Box::new(expr),
                            ));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                accum
            },
    
            Pat::Array(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(
                        &pats[i],
                        &Expr::Index(
                            Box::new(scrut.clone()),
                            Box::new(Expr::Integer(i as i64))
                        ),
                    ) {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(
                                Box::new(accum_inner.clone()),
                                BinaryOp::LogAnd,
                                Box::new(expr),
                            ));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                accum
            },
    
            Pat::Range(pat_lo,pat_hi) => {
                // for now only support Pat::Integer and Pat::Float
                if let Pat::Integer(value_lo) = **pat_lo {
                    if let Pat::Integer(value_hi) = **pat_hi {
                        Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(
                                    Box::new(scrut.clone()),
                                    BinaryOp::GreaterEq,
                                    Box::new(Expr::Integer(value_lo)),
                                )),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(
                                    Box::new(scrut.clone()),
                                    BinaryOp::Less,
                                    Box::new(Expr::Integer(value_hi)),
                                )),
                            )
                        )       
                    }
                    else {
                        panic!("pattern range can only be integers or floats");
                    }
                }
                else if let Pat::Float(value_lo) = **pat_lo {
                    if let Pat::Float(value_hi) = **pat_hi {
                        Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(
                                    Box::new(scrut.clone()),
                                    BinaryOp::GreaterEq,
                                    Box::new(Expr::Float(value_lo)),
                                )),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(
                                    Box::new(scrut.clone()),
                                    BinaryOp::Less,
                                    Box::new(Expr::Float(value_hi)),
                                )),
                            )
                        )
                    }
                    else {
                        panic!("pattern range can only be integers or floats");
                    }
                }
                else {
                    panic!("pattern range can only be integers or floats");
                }
            },
    
            Pat::UnknownIdent(ident) => if self.consts.contains_key(ident) || self.stdlib_consts.contains_key(ident) {
                Some(Expr::Binary(
                    Box::new(scrut.clone()),
                    BinaryOp::Eq,
                    Box::new(Expr::Const(ident.clone()))
                ))
            }
            else {
                None
            },
    
            Pat::Const(ident) => Some(
                Expr::Binary(
                    Box::new(scrut.clone()),
                    BinaryOp::Eq,
                    Box::new(Expr::Const(ident.clone()))
                )
            ),
    
            Pat::Tuple(_,pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.make_pat_bool(
                        &pats[i],
                        &Expr::TupleIndex(Box::new(scrut.clone()),i),
                    ) {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(
                                Box::new(accum_inner.clone()),
                                BinaryOp::LogAnd,
                                Box::new(expr),
                            ));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                accum
            },
    
            Pat::Struct(_,fields) => {
                let mut accum: Option<Expr> = None;
                for identpat in fields.iter() {
                    if let FieldPat::IdentPat(ident,pat) = identpat {
                        if let Some(expr) = self.make_pat_bool(
                            pat,
                            &Expr::Field(
                                Box::new(scrut.clone()),
                                ident.clone()
                            ),
                        ) {
                            if let Some(accum_inner) = accum {
                                accum = Some(Expr::Binary(
                                    Box::new(accum_inner.clone()),
                                    BinaryOp::LogAnd,
                                    Box::new(expr),
                                ));
                            }
                            else {
                                accum = Some(expr);
                            }
                        }
                    }
                }
                accum
            },
    
            Pat::Variant(_,variant) => {
                let variant_ident = match variant {
                    VariantPat::Naked(ident) |
                    VariantPat::Tuple(ident,_) |
                    VariantPat::Struct(ident,_) => ident,
                };
                let mut accum = Expr::Discriminant(Box::new(scrut.clone()),variant_ident.clone());
                match variant {
                    VariantPat::Naked(_) => { },
                    VariantPat::Tuple(_,pats) => {
                        for i in 0..pats.len() {
                            if let Some(expr) = self.make_pat_bool(
                                &pats[i],
                                &Expr::DestructTuple(
                                    Box::new(scrut.clone()),
                                    variant_ident.clone(),
                                    i
                                ),
                            ) {
                                accum = Expr::Binary(
                                    Box::new(accum),
                                    BinaryOp::LogAnd,
                                    Box::new(expr),
                                );
                            }
                        }
                    },
                    VariantPat::Struct(_,fields) => {
                        for field in fields.iter() {
                            if let FieldPat::IdentPat(ident,pat) = field {
                                if let Some(expr) = self.make_pat_bool(
                                    pat,
                                    &Expr::DestructStruct(
                                        Box::new(scrut.clone()),
                                        variant_ident.clone(),
                                        ident.clone()
                                    ),
                                ) {
                                    accum = Expr::Binary(
                                        Box::new(accum),
                                        BinaryOp::LogAnd,
                                        Box::new(expr),
                                    );
                                }
                            }
                        }
                    },
                }
                Some(accum)
            },
        }    
    }

    fn make_pats_bool(&self,pats: &Vec<Pat>,local: &str) -> Option<Expr> {
        let mut accum: Option<Expr> = None;
        for pat in pats.iter() {
            if let Some(expr) = self.make_pat_bool(
                pat,
                &Expr::UnknownIdent(local.to_string())
            ) {
                if let Some(accum_inner) = accum {
                    accum = Some(Expr::Binary(
                        Box::new(accum_inner),
                        BinaryOp::LogOr,
                        Box::new(expr),
                    ));
                }
                else {
                    accum = Some(expr);
                }
            }
        }
        accum    
    }

    fn destructure_pat(&self,pat: &Pat,scrut: &Expr) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        match pat {
    
            Pat::Wildcard |
            Pat::Rest |
            Pat::Boolean(_) |
            Pat::Integer(_) |
            Pat::Float(_) => { },
    
            Pat::AnonTuple(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(
                        &pats[i],
                        &Expr::TupleIndex(
                            Box::new(scrut.clone()),
                            i,
                        )
                    ));
                }
            },
    
            Pat::Array(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(
                        &pats[i],
                        &Expr::Index(
                            Box::new(scrut.clone()),
                            Box::new(Expr::Integer(i as i64)),
                        )
                    ));
                }
            },
    
            Pat::Range(_,_) => { },
    
            Pat::UnknownIdent(ident) => stats.push(Stat::Local(
                ident.clone(),
                Box::new(Type::Inferred),
                Box::new(scrut.clone())
            )),
    
            Pat::Const(_) => { },
    
            Pat::Tuple(_,pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(
                        &pats[i],
                        &Expr::TupleIndex(
                            Box::new(scrut.clone()),
                            i,
                        )
                    ));
                }
            },
    
            Pat::Struct(_,fields) => {
                for field in fields.iter() {
                    match field {
                        FieldPat::Wildcard |
                        FieldPat::Rest => { },
                        FieldPat::Ident(ident) => stats.push(
                            Stat::Local(
                                ident.clone(),
                                Box::new(Type::Inferred),
                                Box::new(Expr::Field(
                                    Box::new(scrut.clone()),
                                    ident.clone(),
                                )),
                            )
                        ),
                        FieldPat::IdentPat(ident,pat) => stats.append(
                            &mut self.destructure_pat(
                                pat,
                                &Expr::Field(
                                    Box::new(scrut.clone()),
                                    ident.clone(),
                                ),
                            )
                        ),
                    }
                }
            },
    
            Pat::Variant(_,variant) => {
                match variant {
                    VariantPat::Naked(_) => { },
                    VariantPat::Tuple(_,pats) => {
                        for i in 0..pats.len() {
                            stats.append(&mut self.destructure_pat(
                                &pats[i],
                                &Expr::TupleIndex(
                                    Box::new(scrut.clone()),
                                    i,
                                ),
                            ));
                        }
                    },
                    VariantPat::Struct(_,fields) => {
                        for field in fields.iter() {
                            if let FieldPat::IdentPat(ident,pat) = field {
                                stats.append(&mut self.destructure_pat(
                                    pat,
                                    &Expr::Field(
                                        Box::new(scrut.clone()),
                                        ident.clone(),
                                    ),
                                ));
                            }
                        }
                    },
                }
            },
        }
    
        stats    
    }

    fn destructure_pats(&self,pats: &Vec<Pat>,scrut: &str) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        for pat in pats.iter() {
            stats.append(&mut self.destructure_pat(
                pat,
                &Expr::UnknownIdent(scrut.to_string())
            ));
        }
        stats
    }

    fn process_type(&self,type_: Type) -> Type {
        match type_ {

            Type::Array(type_,expr) => {
                let new_type = self.process_type(*type_);
                let new_expr = self.process_expr(*expr);
                Type::Array(
                    Box::new(new_type),
                    Box::new(new_expr),
                )
            },

            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types {
                    new_types.push(self.process_type(type_));
                }
                Type::AnonTuple(new_types)
            },

            _ => type_,
        }
    }

    fn process_block(&self,block: Block) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats {
            match stat {

                Stat::Let(pat,type_,expr) => {
                    let new_expr = self.process_expr(*expr);
                    let expr = Expr::Cast(
                        Box::new(new_expr.clone()),
                        Box::new(*type_),
                    );
                    new_stats.append(&mut self.destructure_pat(
                        &pat,
                        &expr,
                    ));
                },

                Stat::Expr(expr) => {
                    let new_expr = self.process_expr(*expr);
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },

                Stat::Local(ident,type_,expr) => {
                    let new_expr = self.process_expr(*expr);
                    new_stats.push(Stat::Local(
                        ident.clone(),
                        Box::new(*type_),
                        Box::new(new_expr),
                    ));
                },
            }
        }
        let new_expr = if let Some(expr) = block.expr {
            Some(Box::new(self.process_expr(*expr)))
        }
        else {
            None
        };
        Block {
            stats: new_stats,
            expr: new_expr,
        }
    }

    fn process_expr(&self,expr: Expr) -> Expr {
        match expr {

            Expr::Boolean(value) => Expr::Boolean(value),

            Expr::Integer(value) => Expr::Integer(value),

            Expr::Float(value) => Expr::Float(value),

            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Array(new_exprs)
            },

            Expr::Cloned(expr,expr2) => {
                let new_expr = self.process_expr(*expr);
                let new_expr2 = self.process_expr(*expr2);
                Expr::Cloned(
                    Box::new(new_expr),
                    Box::new(new_expr2),
                )
            },

            Expr::Index(expr,expr2) => {
                let new_expr = self.process_expr(*expr);
                let new_expr2 = self.process_expr(*expr2);
                Expr::Index(
                    Box::new(new_expr),
                    Box::new(new_expr2),
                )
            },

            Expr::Cast(expr,type_) => {
                let new_expr = self.process_expr(*expr);
                Expr::Cast(
                    Box::new(new_expr),
                    Box::new(*type_),
                )
            },

            Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::AnonTuple(new_exprs)
            },

            Expr::Unary(op,expr) => {
                let new_expr = self.process_expr(*expr);
                Expr::Unary(op,Box::new(new_expr))
            },

            Expr::Binary(expr,op,expr2) => {
                let new_expr = self.process_expr(*expr);
                let new_expr2 = self.process_expr(*expr2);
                Expr::Binary(
                    Box::new(new_expr),
                    op,
                    Box::new(new_expr2),
                )
            },

            Expr::Continue => Expr::Continue,

            Expr::Break(expr) => if let Some(expr) = expr {
                Expr::Break(Some(Box::new(self.process_expr(*expr))))
            }
            else {
                Expr::Break(None)
            },

            Expr::Return(expr) => if let Some(expr) = expr {
                Expr::Return(Some(Box::new(self.process_expr(*expr))))
            }
            else {
                Expr::Return(None)
            },

            Expr::Block(block) => Expr::Block(self.process_block(block)),

            Expr::If(expr,block,else_expr) => {
                let new_expr = self.process_expr(*expr);
                let new_block = self.process_block(block);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.process_expr(*else_expr)))
                }
                else {
                    None
                };
                Expr::If(
                    Box::new(new_expr),
                    new_block,
                    new_else_expr,
                )
            },

            Expr::While(expr,block) => {
                let new_expr = self.process_expr(*expr);
                let new_block = self.process_block(block);
                Expr::While(
                    Box::new(new_expr),
                    new_block,
                )
            },

            Expr::Loop(block) => {
                let new_block = self.process_block(block);
                Expr::Loop(new_block)
            },

            Expr::IfLet(pats,expr,block,else_expr) => {
                let new_expr = self.process_expr(*expr);
                let mut new_block = self.process_block(block);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.process_expr(*else_expr)))
                }
                else {
                    None
                };
                let mut if_block = Block {
                    stats: vec![
                        Stat::Local(
                            "scrut".to_string(),
                            Box::new(Type::Inferred),
                            Box::new(new_expr),
                        )
                    ],
                    expr: None,
                };
                let condition = self.make_pats_bool(&pats,"scrut").expect("unable to create boolean condition from if-let patterns");
                let mut then_block = Block {
                    stats: self.destructure_pats(&pats,"scrut"),
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                if_block.expr = Some(Box::new(
                    Expr::If(
                        Box::new(condition),
                        then_block,
                        new_else_expr,
                    )
                ));
                Expr::Block(if_block)
            },

            Expr::For(pats,range,block) => {
                if pats.len() == 1 {
                    if let Pat::UnknownIdent(ident) = &pats[0] {
                        let (expr,op,expr2) = match range {
                            Range::FromTo(expr,expr2) => (
                                self.process_expr(*expr),
                                BinaryOp::Less,
                                self.process_expr(*expr2),
                            ),
                            Range::FromToIncl(expr,expr2) => (
                                self.process_expr(*expr),
                                BinaryOp::LessEq,
                                self.process_expr(*expr2),
                            ),
                            _ => panic!("for range can only be .. or ..="),
                        };
                        let mut stats = block.stats.clone();
                        if let Some(expr) = block.expr {
                            stats.push(Stat::Expr(expr));
                        }
                        let loop_block = Block {
                            stats,
                            expr: Some(Box::new(
                                Expr::Binary(
                                    Box::new(Expr::UnknownIdent(ident.clone())),
                                    BinaryOp::AddAssign,
                                    Box::new(Expr::Integer(1)),
                                )
                            )),
                        };
                        let for_block = Block {
                            stats: vec![
                                Stat::Local(
                                    ident.clone(),
                                    Box::new(Type::Inferred),
                                    Box::new(expr),
                                ),
                            ],
                            expr: Some(Box::new(
                                Expr::While(
                                    Box::new(
                                        Expr::Binary(
                                            Box::new(Expr::UnknownIdent(ident.clone())),
                                            op,
                                            Box::new(expr2),
                                        )
                                    ),
                                    loop_block
                                )
                            )),
                        };
                        Expr::Block(for_block)
                    }
                    else {
                        panic!("for-pattern can only be a single variable");
                    }
                }
                else {
                    panic!("for-pattern can only be a single variable");
                }
            },

            Expr::WhileLet(pats,expr,block) => {
                let new_expr = self.process_expr(*expr);
                let mut new_block = self.process_block(block);
                let mut while_block = Block {
                    stats: vec![
                        Stat::Local(
                            "scrut".to_string(),
                            Box::new(Type::Inferred),
                            Box::new(new_expr),
                        )
                    ],
                    expr: None,
                };
                let condition = self.make_pats_bool(&pats,"scrut").expect("unable to create boolean condition from if-let patterns");
                let mut then_block = Block {
                    stats: self.destructure_pats(&pats,"scrut"),
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                while_block.expr = Some(Box::new(
                    Expr::While(
                        Box::new(condition),
                        then_block,
                    )
                ));
                Expr::Block(while_block)
            },

            Expr::Match(expr,arms) => {
                let new_expr = self.process_expr(*expr);
                let mut match_block = Block {
                    stats: vec![
                        Stat::Local(
                            "scrut".to_string(),
                            Box::new(Type::Inferred),
                            Box::new(new_expr),
                        )
                    ],
                    expr: None,
                };
                let mut exprs: Vec<Expr> = Vec::new();
                let mut else_expr: Option<Box<Expr>> = None;
                for (pats,_,expr) in arms {
                    // TODO: if_expr does what exactly?
                    let new_expr = self.process_expr(*expr);
                    if let Some(condition) = self.make_pats_bool(&pats,"scrut") {
                        let arm_block = Block {
                            stats: self.destructure_pats(&pats,"scrut"),
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
                        panic!("match expression can only have one wildcard");
                    }
                }
                let mut result_expr: Option<Box<Expr>> = else_expr;
                for i in 0..exprs.len() {
                    if let Expr::If(condition,block,_) = &exprs[exprs.len() - i - 1] {
                        result_expr = Some(Box::new(Expr::If(condition.clone(),block.clone(),result_expr)));
                    }
                }
                match_block.expr = Some(Box::new(*result_expr.unwrap()));
                Expr::Block(match_block)
            },

            Expr::UnknownIdent(ident) => Expr::UnknownIdent(ident),

            Expr::TupleOrCall(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::TupleOrCall(ident,new_exprs)
            },

            Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.process_expr(expr)));
                }
                Expr::Struct(ident,new_fields)
            },

            Expr::Variant(enum_ident,variant) => {
                let new_variant = match variant {
                    VariantExpr::Naked(ident) => VariantExpr::Naked(ident),
                    VariantExpr::Tuple(ident,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs {
                            new_exprs.push(self.process_expr(expr));
                        }
                        VariantExpr::Tuple(ident,new_exprs)
                    },
                    VariantExpr::Struct(ident,fields) => {
                        let mut new_fields: Vec<(String,Expr)> = Vec::new();
                        for (ident,expr) in fields {
                            new_fields.push((ident.clone(),self.process_expr(expr)));
                        }
                        VariantExpr::Struct(ident,new_fields)
                    },
                };
                Expr::Variant(enum_ident,new_variant)
            },

            Expr::Method(expr,ident,exprs) => {
                let new_expr = self.process_expr(*expr);
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Method(Box::new(new_expr),ident,new_exprs)
            },

            Expr::Field(expr,ident) => {
                let new_expr = self.process_expr(*expr);
                Expr::Field(Box::new(new_expr),ident)
            },

            Expr::TupleIndex(expr,index) => {
                let new_expr = self.process_expr(*expr);
                Expr::TupleIndex(Box::new(new_expr),index)
            },

            Expr::Param(ident) => Expr::Param(ident),

            Expr::Local(ident) => Expr::Local(ident),

            Expr::Const(ident) => Expr::Const(ident),

            Expr::Tuple(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Tuple(ident,new_exprs)
            },

            Expr::Call(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Call(ident,new_exprs)
            },

            Expr::Discriminant(expr,ident) => {
                let new_expr = self.process_expr(*expr);
                Expr::Discriminant(Box::new(new_expr),ident)
            },

            Expr::DestructTuple(expr,variant_ident,index) => {
                let new_expr = self.process_expr(*expr);
                Expr::DestructTuple(Box::new(new_expr),variant_ident,index)
            },

            Expr::DestructStruct(expr,variant_ident,ident) => {
                let new_expr = self.process_expr(*expr);
                Expr::DestructStruct(Box::new(new_expr),variant_ident,ident)
            },
        }
    }

    pub fn process_module(module: RustModule) -> DestructuredModule {

        // instantiate to get access to the constants
        let stdlib = StandardLib::new();

        // create context with clones of original constant lists, this is only used to identify the constants, not do anything else with them
        let context = Context {
            consts: module.consts.clone(),
            stdlib_consts: stdlib.consts.clone(),
        };

        // user named tuples
        let mut new_tuples: HashMap<String,Tuple> = HashMap::new();
        for tuple in module.tuples.values() {
            let mut new_types: Vec<Type> = Vec::new();
            for type_ in tuple.types.iter() {
                new_types.push(context.process_type(type_.clone()));
            }
            new_tuples.insert(
                tuple.ident.clone(),
                Tuple {
                    ident: tuple.ident.clone(),
                    types: new_types,
                }
            );
        }

        // user structs
        let mut new_structs: HashMap<String,Struct> = HashMap::new();
        for struct_ in module.structs.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(field.type_.clone()),
                });
            }
            new_structs.insert(
                struct_.ident.clone(),
                Struct {
                    ident: struct_.ident.clone(),
                    fields: new_fields,
                }
            );
        }

        // user external structs
        let mut new_extern_structs: HashMap<String,Struct> = HashMap::new();
        for struct_ in module.extern_structs.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(field.type_.clone()),
                });
            }
            new_extern_structs.insert(
                struct_.ident.clone(),
                Struct {
                    ident: struct_.ident.clone(),
                    fields: new_fields,
                }
            );
        }

        // user enums
        let mut new_enums: HashMap<String,Enum> = HashMap::new();
        for enum_ in module.enums.values() {
            let mut new_variants: Vec<Variant> = Vec::new();
            for variant in enum_.variants.iter() {
                match variant {

                    Variant::Naked(ident) => new_variants.push(Variant::Naked(ident.clone())),

                    Variant::Tuple(ident,types) => {
                        let mut new_types: Vec<Type> = Vec::new();
                        for type_ in types {
                            new_types.push(context.process_type(type_.clone()));
                        }
                        new_variants.push(Variant::Tuple(ident.clone(),new_types));
                    },

                    Variant::Struct(ident,fields) => {
                        let mut new_fields: Vec<Symbol> = Vec::new();
                        for field in fields {
                            new_fields.push(
                                Symbol {
                                    ident: field.ident.clone(),
                                    type_: context.process_type(field.type_.clone()),
                                }
                            );
                        }
                        new_variants.push(Variant::Struct(ident.clone(),new_fields));
                    },
                }
            }
            new_enums.insert(
                enum_.ident.clone(),
                Enum {
                    ident: enum_.ident.clone(),
                    variants: new_variants,
                }
            );
        }

        // user aliases
        let mut new_aliases: HashMap<String,Alias> = HashMap::new();
        for alias in module.aliases.values() {
            let new_type = context.process_type(alias.type_.clone());
            new_aliases.insert(
                alias.ident.clone(),
                Alias {
                    ident: alias.ident.clone(),
                    type_: new_type,
                }
            );
        }

        // user constants
        let mut new_consts: HashMap<String,Const> = HashMap::new();
        for const_ in module.consts.values() {
            let new_expr = context.process_expr(const_.expr.clone());
            new_consts.insert(
                const_.ident.clone(),
                Const {
                    ident: const_.ident.clone(),
                    type_: const_.type_.clone(),
                    expr: new_expr,
                }
            );
        }

        // user functions
        let mut new_functions: HashMap<String,Function> = HashMap::new();
        for function in module.functions.values() {
            let mut new_params: Vec<Symbol> = Vec::new();
            for param in function.params.iter() {
                new_params.push(Symbol {
                    ident: param.ident.clone(),
                    type_: context.process_type(param.type_.clone()),
                });
            }
            let new_type = context.process_type(function.type_.clone());
            let new_block = context.process_block(function.block.clone());
            new_functions.insert(function.ident.clone(),Function {
                ident: function.ident.clone(),
                params: new_params,
                type_: new_type,
                block: new_block,
            });
        }

        DestructuredModule {
            ident: module.ident.clone(),
            tuples: new_tuples,
            structs: new_structs,
            extern_structs: new_extern_structs,
            enums: new_enums,
            aliases: new_aliases,
            consts: new_consts,
            functions: new_functions,
        }
    }
}

pub fn destructure_module(module: RustModule) -> DestructuredModule {
    Context::process_module(module)
}
