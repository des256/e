use {
    super::*,
    std::collections::HashMap,
};

use ast::*;

pub trait Destructure {
    fn destructure(self,consts: &HashMap<String,Const>) -> Self;
}

fn make_pat_bool(pat: &Pat,scrut: &Expr,consts: &HashMap<String,Const>) -> Option<Expr> {
    match pat {

        Pat::Wildcard | Pat::Rest => None,

        Pat::Boolean(value) => Some(
            if *value {
                scrut.clone()
            }
            else {
                Expr::Unary(UnaryOp::Not,Box::new(scrut.clone()))
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
                if let Some(expr) = make_pat_bool(
                    &pats[i],
                    &Expr::TupleIndex(Box::new(scrut.clone()),i),
                    consts,
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
                if let Some(expr) = make_pat_bool(
                    &pats[i],
                    &Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64))),
                    consts,
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

        Pat::UnknownIdent(ident) => if consts.contains_key(ident) {
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
                if let Some(expr) = make_pat_bool(
                    &pats[i],
                    &Expr::TupleIndex(Box::new(scrut.clone()),i),
                    consts,
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
                    if let Some(expr) = make_pat_bool(
                        pat,
                        &Expr::Field(
                            Box::new(scrut.clone()),
                            ident.clone()
                        ),
                        consts,
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
                        if let Some(expr) = make_pat_bool(&pats[i],&Expr::DestructTuple(Box::new(scrut.clone()),variant_ident.clone(),i),consts) {
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
                            if let Some(expr) = make_pat_bool(pat,&Expr::DestructStruct(Box::new(scrut.clone()),variant_ident.clone(),ident.clone()),consts) {
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

fn make_pats_bool(pats: &Vec<Pat>,local: &str,consts: &HashMap<String,Const>) -> Option<Expr> {
    let mut accum: Option<Expr> = None;
    for pat in pats.iter() {
        if let Some(expr) = make_pat_bool(pat,&Expr::UnknownIdent(local.to_string()),consts) {
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

fn destructure_pat(pat: &Pat,scrut: &Expr) -> Vec<Stat> {
    let mut stats: Vec<Stat> = Vec::new();
    match pat {

        Pat::Wildcard |
        Pat::Rest |
        Pat::Boolean(_) |
        Pat::Integer(_) |
        Pat::Float(_) => { },

        Pat::AnonTuple(pats) => {
            for i in 0..pats.len() {
                stats.append(&mut destructure_pat(
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
                stats.append(&mut destructure_pat(
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

        Pat::Tuple(ident,pats) => {
            for i in 0..pats.len() {
                stats.append(&mut destructure_pat(
                    &pats[i],
                    &Expr::TupleIndex(
                        Box::new(scrut.clone()),
                        i,
                    )
                ));
            }
        },

        Pat::Struct(ident,fields) => {
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
                        &mut destructure_pat(
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

        Pat::Variant(ident,variant) => {
            match variant {
                VariantPat::Naked(ident) => { },
                VariantPat::Tuple(ident,pats) => {
                    for i in 0..pats.len() {
                        stats.append(&mut destructure_pat(
                            &pats[i],
                            &Expr::TupleIndex(
                                Box::new(scrut.clone()),
                                i,
                            ),
                        ));
                    }
                },
                VariantPat::Struct(ident,fields) => {
                    for field in fields.iter() {
                        if let FieldPat::IdentPat(ident,pat) = field {
                            stats.append(&mut destructure_pat(
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

fn destructure_pats(pats: &Vec<Pat>,scrut: &str) -> Vec<Stat> {
    let mut stats: Vec<Stat> = Vec::new();
    for pat in pats.iter() {
        stats.append(&mut destructure_pat(
            pat,
            &Expr::UnknownIdent(scrut.to_string())
        ));
    }
    stats
}

impl Destructure for Block {
    fn destructure(self,consts: &HashMap<String,Const>) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in self.stats.iter() {
            match stat {

                Stat::Let(pat,type_,expr) => {
                    let new_expr = expr.destructure(consts);
                    let expr = Expr::Cast(
                        Box::new(new_expr.clone()),
                        Box::new(**type_),
                    );
                    new_stats.append(&mut destructure_pat(
                        pat,
                        &expr,
                    ));
                },

                Stat::Expr(expr) => {
                    let new_expr = expr.destructure(consts);
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },

                Stat::Local(ident,type_,expr) => {
                    let new_expr = expr.destructure(consts);
                    new_stats.push(Stat::Local(
                        ident.clone(),
                        Box::new(**type_),
                        Box::new(new_expr),
                    ));
                },
            }
        }
        let mut new_expr = if let Some(expr) = self.expr {
            Some(Box::new(expr.destructure(consts)))
        }
        else {
            None
        };
        Block {
            stats: new_stats,
            expr: new_expr,
        }
    }
}

impl Destructure for Expr {
    fn destructure(self,consts: &HashMap<String,Const>) -> Expr {
        match self {

            Expr::Boolean(value) => Expr::Boolean(value),

            Expr::Integer(value) => Expr::Integer(value),

            Expr::Float(value) => Expr::Float(value),

            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::Array(new_exprs)
            },

            Expr::Cloned(expr,expr2) => {
                let new_expr = expr.destructure(consts);
                let new_expr2 = expr2.destructure(consts);
                Expr::Cloned(
                    Box::new(new_expr),
                    Box::new(new_expr2),
                )
            },

            Expr::Index(expr,expr2) => {
                let new_expr = expr.destructure(consts);
                let new_expr2 = expr2.destructure(consts);
                Expr::Index(
                    Box::new(new_expr),
                    Box::new(new_expr2),
                )
            },

            Expr::Cast(expr,type_) => {
                let new_expr = expr.destructure(consts);
                Expr::Cast(
                    Box::new(new_expr),
                    Box::new(*type_),
                )
            },

            Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::AnonTuple(new_exprs)
            },

            Expr::Unary(op,expr) => {
                let new_expr = expr.destructure(consts);
                Expr::Unary(op,Box::new(new_expr))
            },

            Expr::Binary(expr,op,expr2) => {
                let new_expr = expr.destructure(consts);
                let new_expr2 = expr2.destructure(consts);
                Expr::Binary(
                    Box::new(new_expr),
                    op,
                    Box::new(new_expr2),
                )
            },

            Expr::Continue => Expr::Continue,

            Expr::Break(expr) => if let Some(expr) = expr {
                Expr::Break(Some(Box::new(expr.destructure(consts))))
            }
            else {
                Expr::Break(None)
            },

            Expr::Return(expr) => if let Some(expr) = expr {
                Expr::Return(Some(Box::new(expr.destructure(consts))))
            }
            else {
                Expr::Return(None)
            },

            Expr::Block(block) => Expr::Block(block.destructure(consts)),

            Expr::If(expr,block,else_expr) => {
                let new_expr = expr.destructure(consts);
                let new_block = block.destructure(consts);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(else_expr.destructure(consts)))
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
                let new_expr = expr.destructure(consts);
                let new_block = block.destructure(consts);
                Expr::While(
                    Box::new(new_expr),
                    new_block,
                )
            },

            Expr::Loop(block) => {
                let new_block = block.destructure(consts);
                Expr::Loop(new_block)
            },

            Expr::IfLet(pats,expr,block,else_expr) => {
                let new_expr = expr.destructure(consts);
                let mut new_block = block.destructure(consts);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(else_expr.destructure(consts)))
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
                let condition = make_pats_bool(&pats,"scrut",consts).expect("unable to create boolean condition from if-let patterns");
                let mut then_block = Block {
                    stats: destructure_pats(&pats,"scrut"),
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
                // TODO: translate patterns, maybe turn to Expr::While
            },

            Expr::WhileLet(pats,expr,block) => {
                let new_expr = expr.destructure(consts);
                let mut new_block = block.destructure(consts);
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
                let condition = make_pats_bool(&pats,"scrut",consts).expect("unable to create boolean condition from if-let patterns");
                let mut then_block = Block {
                    stats: destructure_pats(&pats,"scrut"),
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
                let new_expr = expr.destructure(consts);
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
                for (pats,if_expr,expr) in arms.iter() {
                    // TODO: if_expr does what exactly?
                    let new_expr = expr.destructure(consts);
                    if let Some(condition) = make_pats_bool(pats,"scrut",consts) {
                        let arm_block = Block {
                            stats: destructure_pats(pats,"scrut"),
                            expr: Some(Box::new(new_expr)),
                        };
                        exprs.push(Expr::If(Box::new(condition),arm_block,None));
                    }
                    else if let None = else_expr {
                        let arm_block = Block {
                            stats: Vec::new(),
                            expr: Some(Box::new(*expr.clone())),
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
                        result_expr = Some(Box::new(Expr::If(*condition,*block,result_expr)));
                    }
                }
                *result_expr.unwrap()
            },

            Expr::UnknownIdent(ident) => Expr::UnknownIdent(ident),

            Expr::TupleOrCall(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::TupleOrCall(ident,new_exprs)
            },

            Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for (ident,expr) in fields.iter() {
                    new_fields.push((ident.clone(),expr.destructure(consts)));
                }
                Expr::Struct(ident,new_fields)
            },

            Expr::Variant(enum_ident,variant) => {
                let mut new_variant = match variant {
                    VariantExpr::Naked(ident) => VariantExpr::Naked(ident),
                    VariantExpr::Tuple(ident,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(expr.destructure(consts));
                        }
                        VariantExpr::Tuple(ident,new_exprs)
                    },
                    VariantExpr::Struct(ident,fields) => {
                        let mut new_fields: Vec<(String,Expr)> = Vec::new();
                        for (ident,expr) in fields.iter() {
                            new_fields.push((ident.clone(),expr.destructure(consts)));
                        }
                        VariantExpr::Struct(ident,new_fields)
                    },
                };
                Expr::Variant(enum_ident,new_variant)
            },

            Expr::Method(expr,ident,exprs) => {
                let new_expr = expr.destructure(consts);
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::Method(Box::new(new_expr),ident,new_exprs)
            },

            Expr::Field(expr,ident) => {
                let new_expr = expr.destructure(consts);
                Expr::Field(Box::new(new_expr),ident)
            },

            Expr::TupleIndex(expr,index) => {
                let new_expr = expr.destructure(consts);
                Expr::TupleIndex(Box::new(new_expr),index)
            },

            Expr::Param(ident) => Expr::Param(ident),

            Expr::Local(ident) => Expr::Local(ident),

            Expr::Const(ident) => Expr::Const(ident),

            Expr::Tuple(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::Tuple(ident,new_exprs)
            },

            Expr::Call(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.destructure(consts));
                }
                Expr::Call(ident,new_exprs)
            },

            Expr::Discriminant(expr,ident) => {
                let new_expr = expr.destructure(consts);
                Expr::Discriminant(Box::new(new_expr),ident)
            },

            Expr::DestructTuple(expr,variant_ident,index) => {
                let new_expr = expr.destructure(consts);
                Expr::DestructTuple(Box::new(new_expr),variant_ident,index)
            },

            Expr::DestructStruct(expr,variant_ident,ident) => {
                let new_expr = expr.destructure(consts);
                Expr::DestructStruct(expr,variant_ident,ident)
            },
        }
    }
}
