// resolve all Expr::Ident, Type::Ident and Pat::Idents

use {
    crate::*,
    std::collections::HashMap,
};

struct Context {
    pub locals: HashMap<String,Type>,
    pub params: HashMap<String,Type>,
}

fn resolve_block(module: &Module,context: &mut Context,block: &Block) -> Block {
    let local_frame = context.locals.clone();
    let mut new_stats: Vec<Stat> = Vec::new();
    for stat in block.stats.iter() {
        new_stats.push(resolve_stat(module,context,stat));
    }
    let new_expr = if let Some(expr) = &block.expr {
        Some(Box::new(resolve_expr(module,context,expr)))
    }
    else {
        None
    };
    context.locals = local_frame;
    Block { stats: new_stats,expr: new_expr, }
}

fn resolve_expr(module: &Module,context: &mut Context,expr: &Expr) -> Expr {
    match expr {
        Expr::Boolean(value) => Expr::Boolean(*value),
        Expr::Integer(value) => Expr::Integer(*value),
        Expr::Float(value) => Expr::Float(*value),
        Expr::Base(ident,fields) => {
            let mut new_fields: Vec<(String,Expr)> = Vec::new();
            for (ident,expr) in fields {
                new_fields.push((ident.clone(),resolve_expr(module,context,expr)));
            }
            Expr::Base(ident.clone(),new_fields)
        },
        Expr::Ident(ident) => {
            if module.consts.contains_key(ident) {
                let ty = &module.consts[ident].0;
                Expr::Const(ident.clone(),ty.clone())
            }
            else if context.locals.contains_key(ident) {
                Expr::Local(ident.clone(),context.locals[ident].clone())
            }
            else if context.params.contains_key(ident) {
                Expr::Param(ident.clone(),context.params[ident].clone())
            }
            else {
                panic!("unknown identifier {}",ident);
            }
        },
        Expr::Local(ident,ty) => Expr::Local(ident.clone(),ty.clone()),
        Expr::Param(ident,ty) => Expr::Param(ident.clone(),ty.clone()),
        Expr::Const(ident,ty) => Expr::Const(ident.clone(),ty.clone()),
        Expr::Array(exprs) => {
            let mut new_exprs: Vec<Expr> = Vec::new();
            for expr in exprs {
                new_exprs.push(resolve_expr(module,context,expr));
            }
            Expr::Array(new_exprs)
        },
        Expr::Cloned(expr,expr2) => Expr::Cloned(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Struct(ident,fields) => {
            let mut new_fields: Vec<(String,Expr)> = Vec::new();
            for (ident,expr) in fields {
                new_fields.push((ident.clone(),resolve_expr(module,context,expr)));
            }
            Expr::Struct(ident.clone(),new_fields)
        },
        Expr::Tuple(ident,exprs) => {
            let mut new_exprs: Vec<Expr> = Vec::new();
            for expr in exprs {
                new_exprs.push(resolve_expr(module,context,expr));
            }
            if module.functions.contains_key(ident) {
                Expr::Call(ident.clone(),new_exprs)
            }
            else {
                Expr::Tuple(ident.clone(),new_exprs)
            }
        },
        Expr::AnonTuple(exprs) => {
            let mut new_exprs: Vec<Expr> = Vec::new();
            for expr in exprs {
                new_exprs.push(resolve_expr(module,context,expr));
            }
            Expr::AnonTuple(new_exprs)
        },
        Expr::Variant(ident,variantexpr) => {
            let new_variantexpr = match variantexpr {
                VariantExpr::Naked(ident) => VariantExpr::Naked(ident.clone()),
                VariantExpr::Tuple(ident,exprs) => {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for expr in exprs {
                        new_exprs.push(resolve_expr(module,context,expr));
                    }
                    VariantExpr::Tuple(ident.clone(),new_exprs)
                },
                VariantExpr::Struct(ident,fields) => {
                    let mut new_fields: Vec<(String,Expr)> = Vec::new();
                    for (ident,expr) in fields {
                        new_fields.push((ident.clone(),resolve_expr(module,context,expr)));
                    }
                    VariantExpr::Struct(ident.clone(),new_fields)
                },
            };
            Expr::Variant(ident.clone(),new_variantexpr)
        },
        Expr::Call(ident,exprs) => {
            let mut new_exprs: Vec<Expr> = Vec::new();
            for expr in exprs {
                new_exprs.push(resolve_expr(module,context,expr));
            }
            Expr::Call(ident.clone(),new_exprs)
        },
        Expr::Field(expr,ident) => Expr::Field(Box::new(resolve_expr(module,context,expr)),ident.clone()),
        Expr::TupleIndex(expr,index) => Expr::TupleIndex(Box::new(resolve_expr(module,context,expr)),*index),
        Expr::Cast(expr,ty) => Expr::Cast(Box::new(resolve_expr(module,context,expr)),resolve_type(module,context,ty)),
        Expr::Neg(expr) => Expr::Neg(Box::new(resolve_expr(module,context,expr))),
        Expr::Not(expr) => Expr::Not(Box::new(resolve_expr(module,context,expr))),
        Expr::Index(expr,expr2) => Expr::Index(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Mul(expr,expr2) => Expr::Mul(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Div(expr,expr2) => Expr::Div(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Mod(expr,expr2) => Expr::Mod(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Add(expr,expr2) => Expr::Add(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Sub(expr,expr2) => Expr::Sub(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Shl(expr,expr2) => Expr::Shl(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Shr(expr,expr2) => Expr::Shr(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::And(expr,expr2) => Expr::And(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Or(expr,expr2) => Expr::Or(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Xor(expr,expr2) => Expr::Xor(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Eq(expr,expr2) => Expr::Eq(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::NotEq(expr,expr2) => Expr::NotEq(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Greater(expr,expr2) => Expr::Greater(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Less(expr,expr2) => Expr::Less(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::GreaterEq(expr,expr2) => Expr::GreaterEq(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::LessEq(expr,expr2) => Expr::LessEq(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::LogAnd(expr,expr2) => Expr::LogAnd(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::LogOr(expr,expr2) => Expr::LogOr(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Assign(expr,expr2) => Expr::Assign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::AddAssign(expr,expr2) => Expr::AddAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::SubAssign(expr,expr2) => Expr::SubAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::MulAssign(expr,expr2) => Expr::MulAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::DivAssign(expr,expr2) => Expr::DivAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::ModAssign(expr,expr2) => Expr::ModAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::AndAssign(expr,expr2) => Expr::AndAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::OrAssign(expr,expr2) => Expr::OrAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::XorAssign(expr,expr2) => Expr::XorAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::ShlAssign(expr,expr2) => Expr::ShlAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::ShrAssign(expr,expr2) => Expr::ShrAssign(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
        Expr::Continue => Expr::Continue,
        Expr::Break(expr) => if let Some(expr) = expr {
            Expr::Break(Some(Box::new(resolve_expr(module,context,expr))))
        }
        else {
            Expr::Break(None)
        },
        Expr::Return(expr) => if let Some(expr) = expr {
            Expr::Return(Some(Box::new(resolve_expr(module,context,expr))))
        }
        else {
            Expr::Return(None)
        },
        Expr::Block(block) => Expr::Block(resolve_block(module,context,block)),
        Expr::If(expr,block,else_expr) => {
            let new_expr = resolve_expr(module,context,expr);
            let new_block = resolve_block(module,context,block);
            if let Some(else_expr) = else_expr {
                Expr::If(Box::new(new_expr),new_block,Some(Box::new(resolve_expr(module,context,else_expr))))
            }
            else {
                Expr::If(Box::new(new_expr),new_block,None)
            }
        },
        Expr::IfLet(pats,expr,block,else_expr) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            let new_expr = resolve_expr(module,context,expr);
            let new_block = resolve_block(module,context,block);
            if let Some(else_expr) = else_expr {
                Expr::IfLet(new_pats,Box::new(new_expr),new_block,Some(Box::new(resolve_expr(module,context,else_expr))))
            }
            else {
                Expr::IfLet(new_pats,Box::new(new_expr),new_block,None)
            }
        },
        Expr::Loop(block) => Expr::Loop(resolve_block(module,context,block)),
        Expr::For(pats,range,block) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            let new_range = match range {
                Range::Only(expr) => Range::Only(Box::new(resolve_expr(module,context,expr))),
                Range::FromTo(expr,expr2) => Range::FromTo(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
                Range::FromToIncl(expr,expr2) => Range::FromToIncl(Box::new(resolve_expr(module,context,expr)),Box::new(resolve_expr(module,context,expr2))),
                Range::From(expr) => Range::From(Box::new(resolve_expr(module,context,expr))),
                Range::To(expr) => Range::To(Box::new(resolve_expr(module,context,expr))),
                Range::ToIncl(expr) => Range::ToIncl(Box::new(resolve_expr(module,context,expr))),
                Range::All => Range::All,
            };
            let new_block = resolve_block(module,context,block);
            Expr::For(new_pats,new_range,new_block)
        },
        Expr::While(expr,block) => Expr::While(Box::new(resolve_expr(module,context,expr)),resolve_block(module,context,block)),
        Expr::WhileLet(pats,expr,block) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            let new_expr = resolve_expr(module,context,expr);
            let new_block = resolve_block(module,context,block);
            Expr::WhileLet(new_pats,Box::new(new_expr),new_block)
        },
        Expr::Match(expr,arms) => {
            let new_expr = resolve_expr(module,context,expr);
            let mut new_arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
            for (pats,if_expr,expr) in arms {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(resolve_pat(module,context,pat));
                }
                let new_if_expr = if let Some(if_expr) = if_expr {
                    Some(Box::new(resolve_expr(module,context,if_expr)))
                }
                else {
                    None
                };
                let new_expr = resolve_expr(module,context,expr);
                new_arms.push((new_pats,new_if_expr,Box::new(new_expr)))
            }
            Expr::Match(Box::new(new_expr),new_arms)
        },
    }
}

fn resolve_stat(module: &Module,context: &mut Context,stat: &Stat) -> Stat {
    match stat {
        Stat::Let(pat,ty,expr) => {
            let new_pat = resolve_pat(module,context,pat);
            let new_ty = if let Some(ty) = ty {
                Some(resolve_type(module,context,ty))
            }
            else {
                None
            };
            let new_expr = resolve_expr(module,context,expr);
            Stat::Let(new_pat,new_ty,Box::new(new_expr))
        },
        Stat::Expr(expr) => Stat::Expr(Box::new(resolve_expr(module,context,expr))),
    }
}

fn resolve_type(module: &Module,context: &mut Context,ty: &Type) -> Type {
    match ty {
        Type::Inferred => Type::Inferred,
        Type::Integer => Type::Integer,
        Type::Float => Type::Float,
        Type::Void => Type::Void,
        Type::Base(base_type) => Type::Base(base_type.clone()),
        Type::Ident(ident) => {
            if module.structs.contains_key(ident) {
                Type::Struct(ident.clone())
            }
            else if module.tuples.contains_key(ident) {
                Type::Tuple(ident.clone())
            }
            else if module.enums.contains_key(ident) {
                Type::Enum(ident.clone())
            }
            else {
                panic!("unknown type {}",ident);
            }
        },
        Type::Struct(ident) => Type::Struct(ident.clone()),
        Type::Tuple(ident) => Type::Tuple(ident.clone()),
        Type::Enum(ident) => Type::Enum(ident.clone()),
        Type::Array(ty,expr) => Type::Array(Box::new(resolve_type(module,context,ty)),Box::new(resolve_expr(module,context,expr))),
        Type::AnonTuple(types) => {
            let mut new_types: Vec<Type> = Vec::new();
            for ty in types {
                new_types.push(resolve_type(module,context,ty));
            }
            Type::AnonTuple(new_types)
        },
    }
}

fn resolve_pat(module: &Module,context: &mut Context,pat: &Pat) -> Pat {
    match pat {
        Pat::Wildcard => Pat::Wildcard,
        Pat::Rest => Pat::Rest,
        Pat::Boolean(value) => Pat::Boolean(*value),
        Pat::Integer(value) => Pat::Integer(*value),
        Pat::Float(value) => Pat::Float(*value),
        Pat::Ident(ident) => {
            if module.consts.contains_key(ident) {
                let ty = &module.consts[ident].0;
                Pat::Const(ident.clone(),ty.clone())
            }
            else {
                Pat::Ident(ident.clone())
            }
        },
        Pat::Const(ident,ty) => Pat::Const(ident.clone(),ty.clone()),
        Pat::Struct(ident,identpats) => {
            let mut new_identpats: Vec<IdentPat> = Vec::new();
            for identpat in identpats {
                new_identpats.push(match identpat {
                    IdentPat::Wildcard => IdentPat::Wildcard,
                    IdentPat::Rest => IdentPat::Rest,
                    IdentPat::Ident(ident) => IdentPat::Ident(ident.clone()),
                    IdentPat::IdentPat(ident,pat) => IdentPat::IdentPat(ident.clone(),resolve_pat(module,context,pat)),
                });
            }
            Pat::Struct(ident.clone(),new_identpats)
        },
        Pat::Tuple(ident,pats) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            Pat::Tuple(ident.clone(),new_pats)
        },
        Pat::Array(pats) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            Pat::Array(new_pats)
        },
        Pat::AnonTuple(pats) => {
            let mut new_pats: Vec<Pat> = Vec::new();
            for pat in pats {
                new_pats.push(resolve_pat(module,context,pat));
            }
            Pat::AnonTuple(new_pats)
        },
        Pat::Variant(ident,variantpat) => {
            let new_variantpat = match variantpat {
                VariantPat::Naked(ident) => VariantPat::Naked(ident.clone()),
                VariantPat::Tuple(ident,pats) => {
                    let mut new_pats: Vec<Pat> = Vec::new();
                    for pat in pats {
                        new_pats.push(resolve_pat(module,context,pat));
                    }
                    VariantPat::Tuple(ident.clone(),new_pats)
                },
                VariantPat::Struct(ident,identpats) => {
                    let mut new_identpats: Vec<IdentPat> = Vec::new();
                    for identpat in identpats {
                        new_identpats.push(match identpat {
                            IdentPat::Wildcard => IdentPat::Wildcard,
                            IdentPat::Rest => IdentPat::Rest,
                            IdentPat::Ident(ident) => IdentPat::Ident(ident.clone()),
                            IdentPat::IdentPat(ident,pat) => IdentPat::IdentPat(ident.clone(),resolve_pat(module,context,pat)),
                        });
                    }
                    VariantPat::Struct(ident.clone(),new_identpats)
                },
            };
            Pat::Variant(ident.clone(),new_variantpat)
        },
        Pat::Range(pat,pat2) => Pat::Range(Box::new(resolve_pat(module,context,pat)),Box::new(resolve_pat(module,context,pat2))),
    }
}

pub fn resolve_module(module: &Module) -> Module{
    let mut context = Context {
        locals: HashMap::new(),
        params: HashMap::new(),       
    };

    let mut new_functions: HashMap<String,(Vec<(String,Type)>,Type,Block)> = HashMap::new();
    for (ident,(params,return_type,block)) in module.functions.iter() {
        let mut new_params: Vec<(String,Type)> = Vec::new();
        for (ident,ty) in params {
            let new_type = resolve_type(module,&mut context,ty);
            new_params.push((ident.clone(),new_type.clone()));
            context.params.insert(ident.clone(),new_type);
        }
        let new_return_type = resolve_type(module,&mut context,return_type);
        let new_block = resolve_block(module,&mut context,block);
        context.params.clear();
        new_functions.insert(ident.clone(),(new_params,new_return_type,new_block));
    }

    let mut new_structs: HashMap<String,Vec<(String,Type)>> = HashMap::new();
    for (ident,fields) in &module.structs {
        let mut new_fields: Vec<(String,Type)> = Vec::new();
        for (ident,ty) in fields {
            new_fields.push((ident.clone(),resolve_type(module,&mut context,&ty)));
        }
        new_structs.insert(ident.clone(),new_fields);
    }

    let mut new_tuples: HashMap<String,Vec<Type>> = HashMap::new();
    for (ident,types) in &module.tuples {
        let mut new_types: Vec<Type> = Vec::new();
        for ty in types {
            new_types.push(resolve_type(module,&mut context,&ty));
        }
        new_tuples.insert(ident.clone(),new_types);
    }

    let mut new_enums: HashMap<String,Vec<Variant>> = HashMap::new();
    for (ident,variants) in &module.enums {
        let mut new_variants: Vec<Variant> = Vec::new();
        for variant in variants {
            new_variants.push(match variant {
                Variant::Naked(ident) => Variant::Naked(ident.clone()),
                Variant::Tuple(ident,types) => {
                    let mut new_types: Vec<Type> = Vec::new();
                    for ty in types {
                        new_types.push(resolve_type(module,&mut context,&ty));
                    }
                    Variant::Tuple(ident.clone(),new_types)
                },
                Variant::Struct(ident,fields) => {
                    let mut new_fields: Vec<(String,Type)> = Vec::new();
                    for (ident,ty) in fields {
                        new_fields.push((ident.clone(),resolve_type(module,&mut context,ty)));
                    }
                    Variant::Struct(ident.clone(),new_fields)
                },
            });
        }
        new_enums.insert(ident.clone(),new_variants);
    }

    let mut new_consts: HashMap<String,(Type,Expr)> = HashMap::new();
    for (ident,(ty,expr)) in &module.consts {
        new_consts.insert(ident.clone(),(resolve_type(module,&mut context,&ty),resolve_expr(module,&mut context,&expr)));        
    }

    Module {
        ident: module.ident.clone(),
        functions: new_functions,
        structs: new_structs,
        tuples: new_tuples,
        enums: new_enums,
        consts: new_consts,
    }
}
