// resolve all Expr::Ident, Type::Ident and Pat::Idents

use {
    crate::*,
    std::collections::HashMap,
};

struct Context {
    locals: HashMap<String,Type>,
    params: HashMap<String,Type>,
}

fn resolve_expr(module: &Module,context: &mut Context,expr: &mut Expr) {
    match expr {
        Expr::Boolean(_) => { },
        Expr::Integer(_) => { },
        Expr::Float(_) => { },
        Expr::Ident(ident) => {
            if module.consts.borrow().contains_key(ident) {
                let consts = module.consts.borrow();
                let ty = consts[ident].0.borrow_mut();
                *expr = Expr::Const(ident.clone(),ty.clone());
            }
            else if context.locals.contains_key(ident) {
                *expr = Expr::Local(ident.clone(),context.locals[ident].clone());
            }
            else if context.params.contains_key(ident) {
                *expr = Expr::Param(ident.clone(),context.params[ident].clone());
            }
        },
        Expr::Local(_,_) => { },
        Expr::Param(_,_) => { },
        Expr::Const(_,_) => { },
        Expr::Array(exprs) => {
            for expr in exprs {
                resolve_expr(module,context,expr);
            }
        },
        Expr::Cloned(expr,expr2) => {
            resolve_expr(module,context,expr);
            resolve_expr(module,context,expr2);
        },
        Expr::Struct(_,fields) => {
            for (_,expr) in fields {
                resolve_expr(module,context,expr);
            }
        },
        Expr::Tuple(ident,exprs) => {
            for expr in exprs.iter_mut() {
                resolve_expr(module,context,expr);
            }
            if module.functions.borrow().contains_key(ident) {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(expr.clone());
                }
                *expr = Expr::Call(ident.clone(),new_exprs);
            }
        },
        Expr::AnonTuple(exprs) => {
            for expr in exprs {
                resolve_expr(module,context,expr);
            }
        },
        Expr::Variant(_,variant) => {
            match variant {
                VariantExpr::Naked(_) => { },
                VariantExpr::Tuple(_,exprs) => {
                    for expr in exprs {
                        resolve_expr(module,context,expr);
                    }
                },
                VariantExpr::Struct(_,fields) => {
                    for (_,expr) in fields {
                        resolve_expr(module,context,expr);
                    }
                },
            }
        },
        Expr::Call(_,exprs) => {
            for mut expr in exprs {
                resolve_expr(module,context,&mut expr);
            }
        },
        Expr::Field(expr,_) => {
            resolve_expr(module,context,expr);
        },
        Expr::TupleIndex(expr,_) => {
            resolve_expr(module,context,expr);
        },
        Expr::Cast(expr,ty) => {
            resolve_expr(module,context,expr);
            resolve_type(module,context,ty);
        },
        Expr::Neg(expr) |
        Expr::Not(expr) => {
            resolve_expr(module,context,expr);
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
            resolve_expr(module,context,expr);
            resolve_expr(module,context,expr2);
        },
        Expr::Continue => { },
        Expr::Break(expr) => {
            if let Some(expr) = expr {
                resolve_expr(module,context,expr);
            }
        },
        Expr::Return(expr) => {
            if let Some(expr) = expr {
                resolve_expr(module,context,expr);
            }
        },
        Expr::Block(block) => {
            let local_frame = context.locals.clone();
            for stat in block.stats.iter_mut() {
                resolve_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
        },
        Expr::If(expr,block,else_expr) => {
            resolve_expr(module,context,expr);
            let local_frame = context.locals.clone();
            for stat in block.stats.iter_mut() {
                resolve_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
            if let Some(else_expr) = else_expr {
                resolve_expr(module,context,else_expr);
            }
        },
        Expr::IfLet(pats,expr,block,else_expr) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
            resolve_expr(module,context,expr);
            let local_frame = context.locals.clone();
            for mut stat in block.stats.iter_mut() {
                resolve_stat(module,context,&mut stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
            if let Some(else_expr) = else_expr {
                resolve_expr(module,context,else_expr);
            }
        },
        Expr::Loop(block) => {
            let local_frame = context.locals.clone();
            for mut stat in block.stats.iter_mut() {
                resolve_stat(module,context,&mut stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
        },
        Expr::For(pats,range,block) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
            match range {
                Range::Only(expr) => {
                    resolve_expr(module,context,expr);
                },
                Range::FromTo(expr,expr2) => {
                    resolve_expr(module,context,expr);
                    resolve_expr(module,context,expr2);
                },
                Range::FromToIncl(expr,expr2) => {
                    resolve_expr(module,context,expr);
                    resolve_expr(module,context,expr2);
                },
                Range::From(expr) => {
                    resolve_expr(module,context,expr);
                },
                Range::To(expr) => {
                    resolve_expr(module,context,expr);
                },
                Range::ToIncl(expr) => {
                    resolve_expr(module,context,expr);
                },
                Range::All => { },
            }
            for stat in block.stats.iter_mut() {
                resolve_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
        },
        Expr::While(expr,block) => {
            resolve_expr(module,context,expr);
            let local_frame = context.locals.clone();
            for stat in block.stats.iter_mut() {
                resolve_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
        },
        Expr::WhileLet(pats,expr,block) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
            resolve_expr(module,context,expr);
            let local_frame = context.locals.clone();
            for stat in block.stats.iter_mut() {
                resolve_stat(module,context,stat);
            }
            if let Some(expr) = &mut block.expr {
                resolve_expr(module,context,expr);
            }
            context.locals = local_frame;
        },
        Expr::Match(expr,arms) => {
            resolve_expr(module,context,expr);
            for (pats,if_expr,expr) in arms {
                for pat in pats {
                    resolve_pat(module,context,pat);
                }
                if let Some(if_expr) = if_expr {
                    resolve_expr(module,context,if_expr);
                }
                resolve_expr(module,context,expr);
            }
        },
    }
}

fn resolve_stat(module: &Module,context: &mut Context,stat: &mut Stat) {
    match stat {
        Stat::Let(pat,ty,expr) => {
            resolve_pat(module,context,pat);
            if let Some(ty) = ty {
                resolve_type(module,context,ty);
            }
            resolve_expr(module,context,expr);
        },
        Stat::Expr(expr) => {
            resolve_expr(module,context,expr);
        }
    }
}

fn resolve_type(module: &Module,context: &mut Context,ty: &mut Type) {
    match ty {
        Type::Inferred => { },
        Type::Integer => { },
        Type::Float => { },
        Type::Void => { },
        Type::Base(_) => { },
        Type::Ident(ident) => {
            if module.structs.borrow().contains_key(ident) {
                *ty = Type::Struct(ident.clone());
            }
            else if module.tuples.borrow().contains_key(ident) {
                *ty = Type::Tuple(ident.clone());
            }
            else if module.enums.borrow().contains_key(ident) {
                *ty = Type::Enum(ident.clone());
            }
        },
        Type::Struct(_) => { },
        Type::Tuple(_) => { },
        Type::Enum(_) => { },
        Type::Array(ty,expr) => {
            resolve_type(module,context,ty);
            resolve_expr(module,context,expr);
        },
        Type::AnonTuple(types) => {
            for ty in types {
                resolve_type(module,context,ty);
            }
        },
    }
}

fn resolve_pat(module: &Module,context: &mut Context,pat: &mut Pat) {
    match pat {
        Pat::Wildcard => { },
        Pat::Rest => { },
        Pat::Boolean(_) => { },
        Pat::Integer(_) => { },
        Pat::Float(_) => { },
        Pat::Ident(ident) => {
            if module.consts.borrow().contains_key(ident) {
                let consts = module.consts.borrow();
                let ty = consts[ident].0.borrow_mut();
                *pat = Pat::Const(ident.clone(),ty.clone());
            }
        },
        Pat::Const(_,_) => { },
        Pat::Struct(_,identpats) => {
            for identpat in identpats {
                match identpat {
                    IdentPat::Wildcard => { },
                    IdentPat::Rest => { },
                    IdentPat::Ident(_) => { },
                    IdentPat::IdentPat(_,pat) => {
                        resolve_pat(module,context,pat);
                    },
                }
            }
        },
        Pat::Tuple(_,pats) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
        },
        Pat::Array(pats) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
        },
        Pat::AnonTuple(pats) => {
            for pat in pats {
                resolve_pat(module,context,pat);
            }
        },
        Pat::Variant(_,variantpat) => {
            match variantpat {
                VariantPat::Naked(_) => { },
                VariantPat::Tuple(_,pats) => {
                    for pat in pats {
                        resolve_pat(module,context,pat);
                    }
                },
                VariantPat::Struct(_,identpats) => {
                    for identpat in identpats {
                        match identpat {
                            IdentPat::Wildcard => { },
                            IdentPat::Rest => { },
                            IdentPat::Ident(_) => { },
                            IdentPat::IdentPat(_,pat) => {
                                resolve_pat(module,context,pat);
                            },
                        }                       
                    }
                },
            }  
        },
        Pat::Range(pat,pat2) => {
            resolve_pat(module,context,pat);
            resolve_pat(module,context,pat2);
        },
    }
}

pub fn resolve_module(module: &Module) {
    let mut context = Context {
        locals: HashMap::new(),
        params: HashMap::new(),       
    };
    for (_,(params,return_type,block)) in module.functions.borrow().iter() {
        for (ident,ty) in params {
            *context.params.get_mut(ident).unwrap() = ty.clone();
        }
        let mut return_type = return_type.borrow_mut();
        resolve_type(module,&mut context,&mut return_type);
        let local_frame = context.locals.clone();
        let mut block = block.borrow_mut();
        for stat in block.stats.iter_mut() {
            resolve_stat(module,&mut context,stat);
        }
        if let Some(expr) = &mut block.expr {
            resolve_expr(module,&mut context,expr);
        }
        context.locals = local_frame;
        context.params.clear();
    }
    for (_,fields) in module.structs.borrow_mut().iter_mut() {
        for (_,ty) in fields.iter_mut() {
            let mut ty = ty.borrow_mut();
            resolve_type(module,&mut context,&mut ty);
        }
    }
    for (_,types) in module.tuples.borrow_mut().iter_mut() {
        for ty in types {
            let mut ty = ty.borrow_mut();
            resolve_type(module,&mut context,&mut ty);
        }
    }
    for (_,variants) in module.enums.borrow_mut().iter_mut() {
        for variant in variants.iter_mut() {
            match variant {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for mut ty in types {
                        resolve_type(module,&mut context,&mut ty);
                    }
                },
                Variant::Struct(_,fields) => {
                    for (_,ty) in fields {
                        resolve_type(module,&mut context,ty);
                    }
                },
            }
        }
    }
    for (_,(ty,expr)) in module.consts.borrow_mut().iter_mut() {
        let mut ty = ty.borrow_mut();
        resolve_type(module,&mut context,&mut ty);
        let mut expr = expr.borrow_mut();
        resolve_expr(module,&mut context,&mut expr);
    }
}
