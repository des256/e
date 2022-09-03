// convert all tuples to structs (none of the target languages have tuples)

use {
    crate::*,
    std::cell::RefCell,
};

fn detuplify_expr(module: &Module,context: &mut Context,expr: &mut Expr) {
    match expr {
        Expr::Boolean(_) => { },
        Expr::Integer(_) => { },
        Expr::Float(_) => { },
        Expr::Ident(_) => { },  // shouldn't exist
        Expr::Local(_,_) => { },
        Expr::Param(_,_) => { },
        Expr::Const(_,_) => { },
        Expr::Array(exprs) => {
            for expr in exprs.iter_mut() {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Cloned(expr,expr2) => {
            detuplify_expr(module,context,expr);
            detuplify_expr(module,context,expr2);
        },
        Expr::Struct(_,fields) => {
            for (_,expr) in fields.iter_mut() {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Tuple(ident,exprs) => {
            let mut fields: Vec<(String,Expr)> = Vec::new();
            let mut i = 0usize;
            for expr in exprs.iter_mut() {
                detuplify_expr(module,context,expr);
                fields.push((format!("_{}",i),expr.clone()));
                i += 1;
            }
            *expr = Expr::Struct(format!("Tuple{}",ident),fields);
        },
        Expr::AnonTuple(exprs) => {
            let mut fields: Vec<(String,RefCell<Type>)> = Vec::new();
            let mut i = 0usize;
            for expr in exprs.iter_mut() {
                fields.push((format!("_{}",i),RefCell::new(find_type(module,context,expr))));
                i += 1;
            }
            let mut anon_tuple_count = module.anon_tuple_count.borrow_mut();
            module.structs.borrow_mut().insert(format!("Tuple{}",anon_tuple_count),fields);
            *anon_tuple_count += 1;
        },
        Expr::Variant(_,_) => {
            // TODO: enums is a whole different can of worms...
            /*match variant {
                VariantExpr::Naked(ident) => { },
                VariantExpr::Tuple(ident,exprs) => {
                    for mut expr in exprs {
                        detuplify_expr(module,context,expr);
                    }
                },
                VariantExpr::Struct(ident,fields) => {
                    for (ident,mut expr) in fields {
                        detuplify_expr(module,context,&mut expr);
                    }
                },
            }*/
        },
        Expr::Call(_,exprs) => {
            for expr in exprs.iter_mut() {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Field(expr,_) => {
            detuplify_expr(module,context,expr);
        },
        Expr::TupleIndex(expr,_) => {
            detuplify_expr(module,context,expr);
            // TODO: convert to Expr::Field
        },
        Expr::Cast(expr,ty) => {
            detuplify_expr(module,context,expr);
            detuplify_type(module,context,ty);
        },
        Expr::Neg(expr) | Expr::Not(expr) => {
            detuplify_expr(module,context,expr);
        },
        Expr::Index(expr,expr2) |
        Expr::Mul(expr,expr2) |
        Expr::Div(expr,expr2) |
        Expr::Mod(expr,expr2) |
        Expr::Add(expr,expr2) |
        Expr::Sub(expr,expr2) |
        Expr::Shl(expr,expr2) |
        Expr::Shr(expr,expr2) |
        Expr::And(expr,expr2) |
        Expr::Or(expr,expr2) |
        Expr::Xor(expr,expr2) |
        Expr::Eq(expr,expr2) |
        Expr::NotEq(expr,expr2) |
        Expr::Greater(expr,expr2) |
        Expr::Less(expr,expr2) |
        Expr::GreaterEq(expr,expr2) |
        Expr::LessEq(expr,expr2) |
        Expr::LogAnd(expr,expr2) |
        Expr::LogOr(expr,expr2) |
        Expr::Assign(expr,expr2) |
        Expr::AddAssign(expr,expr2) |
        Expr::SubAssign(expr,expr2) |
        Expr::MulAssign(expr,expr2) |
        Expr::DivAssign(expr,expr2) |
        Expr::ModAssign(expr,expr2) |
        Expr::AndAssign(expr,expr2) |
        Expr::OrAssign(expr,expr2) |
        Expr::XorAssign(expr,expr2) |
        Expr::ShlAssign(expr,expr2) |
        Expr::ShrAssign(expr,expr2) => {
            detuplify_expr(module,context,expr);
            detuplify_expr(module,context,expr2);
        },
        Expr::Continue => { },
        Expr::Break(expr) => {
            if let Some(expr) = expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Return(expr) => {
            if let Some(expr) = expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Block(block) => {
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::If(expr,block,else_expr) => {
            detuplify_expr(module,context,expr);
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
            if let Some(else_expr) = else_expr {
                detuplify_expr(module,context,else_expr);
            }
        },
        Expr::IfLet(pats,expr,block,else_expr) => {
            for pat in pats {
                detuplify_pat(module,pat);
            }
            detuplify_expr(module,context,expr);
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
            if let Some(else_expr) = else_expr {
                detuplify_expr(module,context,else_expr);
            }
        },
        Expr::Loop(block) => {
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::For(pats,range,block) => {
            for pat in pats {
                detuplify_pat(module,pat);
            }
            match range {
                Range::Only(expr) => {
                    detuplify_expr(module,context,expr);
                },
                Range::FromTo(expr,expr2) => {
                    detuplify_expr(module,context,expr);
                    detuplify_expr(module,context,expr2);
                },
                Range::FromToIncl(expr,expr2) => {
                    detuplify_expr(module,context,expr);
                    detuplify_expr(module,context,expr2);
                },
                Range::From(expr) => {
                    detuplify_expr(module,context,expr);
                },
                Range::To(expr) => {
                    detuplify_expr(module,context,expr);
                },
                Range::ToIncl(expr) => {
                    detuplify_expr(module,context,expr);
                },
                Range::All => { },
            }
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::While(expr,block) => {
            detuplify_expr(module,context,expr);
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::WhileLet(pats,expr,block) => {
            for pat in pats {
                detuplify_pat(module,pat);
            }
            detuplify_expr(module,context,expr);
            for stat in block.stats.iter_mut() {
                detuplify_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                detuplify_expr(module,context,expr);
            }
        },
        Expr::Match(expr,arms) => {
            detuplify_expr(module,context,expr);
            for (pats,if_expr,expr) in arms {
                for pat in pats {
                    detuplify_pat(module,pat);
                }
                if let Some(if_expr) = if_expr {
                    detuplify_expr(module,context,if_expr);
                }
                detuplify_expr(module,context,expr);
            }
        },
    }
}

fn detuplify_stat(module: &Module,context: &mut Context,stat: &mut Stat) {
    match stat {
        Stat::Let(pat,ty,expr) => {
            detuplify_pat(module,pat);
            if let Some(ty) = ty {
                detuplify_type(module,context,ty);
            }
            detuplify_expr(module,context,expr);
        },
        Stat::Expr(expr) => {
            detuplify_expr(module,context,expr);
        }
    }
}

fn detuplify_type(module: &Module,context: &mut Context,ty: &mut Type) {
    match ty {
        Type::Inferred => { },
        Type::Integer => { },
        Type::Float => { },
        Type::Void => { },
        Type::Base(_) => { },
        Type::Ident(_) => panic!("detuplify_type: Type::Ident shouldn't exist"),
        Type::Struct(_) => { },
        Type::Tuple(_) => {
            // TODO: convert to Type::Struct
        },
        Type::Enum(_) => { },
        Type::Array(ty,expr) => {
            detuplify_type(module,context,ty);
            detuplify_expr(module,context,expr);
        },
        Type::AnonTuple(types) => {
            for ty in types {
                detuplify_type(module,context,ty);
            }
            // TODO: convert to Type::Struct
        },
    }
}

fn detuplify_pat(module: &Module,pat: &mut Pat) {
    match pat {
        Pat::Wildcard => { },
        Pat::Rest => { },
        Pat::Boolean(_) => { },
        Pat::Integer(_) => { },
        Pat::Float(_) => { },
        Pat::Ident(_) => { },
        Pat::Const(_,_) => { },
        Pat::Struct(_,identpats) => {
            for identpat in identpats.iter_mut() {
                match identpat {
                    IdentPat::Wildcard => { },
                    IdentPat::Rest => { },
                    IdentPat::Ident(_) => { },
                    IdentPat::IdentPat(_,pat) => {
                        detuplify_pat(module,pat);
                    },
                }
            }
        },
        Pat::Tuple(_,pats) => {
            for pat in pats.iter_mut() {
                detuplify_pat(module,pat);
            }
            // TODO: convert to Pat::Struct
        },
        Pat::Array(pats) => {
            for pat in pats.iter_mut() {
                detuplify_pat(module,pat);
            }
        },
        Pat::AnonTuple(pats) => {
            for pat in pats.iter_mut() {
                detuplify_pat(module,pat);
            }
            // TODO: convert to Pat::Struct
        },
        Pat::Variant(_,variantpat) => {
            match variantpat {
                VariantPat::Naked(_) => { },
                VariantPat::Tuple(_,pats) => {
                    for pat in pats.iter_mut() {
                        detuplify_pat(module,pat);
                    }
                    // convert to VariantPat::Struct
                },
                VariantPat::Struct(_,identpats) => {
                    for identpat in identpats {
                        match identpat {
                            IdentPat::Wildcard => { },
                            IdentPat::Rest => { },
                            IdentPat::Ident(_) => { },
                            IdentPat::IdentPat(_,pat) => {
                                detuplify_pat(module,pat);
                            },
                        }                       
                    }
                },
            }  
        },
        Pat::Range(pat,pat2) => {
            detuplify_pat(module,pat);
            detuplify_pat(module,pat2);
        },
    }
}

pub fn detuplify_module(module: &Module) {
    let mut context = Context {
        params: Vec::new(),
        locals: Vec::new(),
    };
    for (_,(_,return_type,block)) in module.functions.borrow_mut().iter_mut() {
        let mut return_type = return_type.borrow_mut();
        detuplify_type(module,&mut context,&mut return_type);
        let mut block = block.borrow_mut();
        for stat in block.stats.iter_mut() {
            detuplify_stat(module,&mut context,stat);
        }
        if let Some(expr) = &mut block.expr {
            detuplify_expr(module,&mut context,expr);
        }
    }
    for (_,fields) in module.structs.borrow_mut().iter_mut() {
        for (_,ty) in fields {
            let mut ty = ty.borrow_mut();
            detuplify_type(module,&mut context,&mut ty);
        }
    }
    for (_,types) in module.tuples.borrow_mut().iter_mut() {
        for ty in types {
            let mut ty = ty.borrow_mut();
            detuplify_type(module,&mut context,&mut ty);
        }
    }
    for (_,variants) in module.enums.borrow_mut().iter_mut() {
        for variant in variants {
            match variant {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for mut ty in types {
                        detuplify_type(module,&mut context,&mut ty);
                    }
                    // TODO: convert to Variant::Struct
                },
                Variant::Struct(_,fields) => {
                    for (_,ty) in fields.iter_mut() {
                        detuplify_type(module,&mut context,ty);
                    }
                },
            }
        }
    }
    for (_,(ty,expr)) in module.consts.borrow_mut().iter_mut() {
        let mut ty = ty.borrow_mut();
        let mut expr = expr.borrow_mut();
        detuplify_type(module,&mut context,&mut ty);
        detuplify_expr(module,&mut context,&mut expr);
    }
}
