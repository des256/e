use {
    crate::*,
    std::collections::HashMap,
};

fn find_tightest_type(type1: &sr::Type,type2: &sr::Type) -> Option<sr::Type> {

    // if both types are identical, take one of them
    if type1 == type2 {
        Some(type1.clone())
    }
    
    // if one is sr::Type::Inferred, take the other one
    else if let sr::Type::Inferred = type1 {
        Some(type2.clone())
    }
    else if let sr::Type::Inferred = type2 {
        Some(type1.clone())
    }

    // if one is sr::Type::Integer, upcast to sr::Type::Float or sr::Type::Base
    else if let sr::Type::Integer = type1 {
        match type2 {
            sr::Type::Float => Some(sr::Type::Float),
            sr::Type::Base(bt) => match bt {
                sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                _ => None,
            },
            _ => None,
        }
    }
    else if let sr::Type::Integer = type2 {
        match type1 {
            sr::Type::Float => Some(sr::Type::Float),
            sr::Type::Base(bt) => match bt {
                sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                _ => None,
            },
            _ => None,
        }
    }

    // if one is sr::Type::Float, upcast to sr::Type::Base
    else if let sr::Type::Float = type1 {
        if let sr::Type::Base(bt) = type2 {
            match bt {
                sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                _ => None,
            }
        }
        else {
            None
        }
    }
    else if let sr::Type::Float = type2 {
        if let sr::Type::Base(bt) = type1 {
            match bt {
                sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                _ => None,
            }
        }
        else {
            None
        }
    }

    // anything else is not compatible
    else {
        None
    }
}

pub(crate) fn infer_expr_type(module: &sr::Module,anon_tuple_structs: &mut HashMap<String,Vec<(String,sr::Type)>>,locals: &HashMap<String,sr::Type>,expr: &sr::Expr) -> sr::Type {
    match expr {
        sr::Expr::Boolean(_) => sr::Type::Boolean,
        sr::Expr::Integer(_) => sr::Type::Integer,
        sr::Expr::Float(_) => sr::Type::Float,
        sr::Expr::Base(bt,_) => sr::Type::Base(bt.clone()),
        sr::Expr::Ident(ident) => panic!("unable to infer type of {}",ident),
        sr::Expr::Local(ident) => panic!("unable to infer type of local {}",ident),
        sr::Expr::Param(ident) => panic!("unable to infer type of param {}",ident),
        sr::Expr::Const(ident) => panic!("unable to infer type of const {}",ident),
        sr::Expr::Array(exprs) => {
            let mut ty: sr::Type = sr::Type::Inferred;
            for expr in exprs {
                ty = find_tightest_type(&ty,&infer_expr_type(module,anon_tuple_structs,expr)).expect(&format!("in array, {} has incompatible type",expr));
            }
            sr::Type::Array(Box::new(ty),Box::new(sr::Expr::Integer(exprs.len() as i64)))
        },
        sr::Expr::Cloned(expr,expr2) => {
            let ty = infer_expr_type(module,anon_tuple_structs,expr);
            sr::Type::Array(Box::new(ty),Box::new(*expr2.clone()))
        },
        sr::Expr::Struct(ident,_) => sr::Type::Struct(ident.clone()),
        sr::Expr::Variant(ident,_) => sr::Type::Enum(ident.clone()),
        sr::Expr::Call(ident,_) => {
            let function = &module.functions[ident];
            function.1.clone()
        },
        sr::Expr::Field(expr,ident) => {
            let ty = infer_expr_type(module,anon_tuple_structs,expr);
            let mut found_ty: Option<sr::Type> = None;
            if let sr::Type::Struct(struct_ident) = ty {
                if module.structs.contains_key(&struct_ident) {
                    let struct_ = &module.structs[&struct_ident];
                    for (field_ident,ty) in struct_ {
                        if field_ident == ident {
                            found_ty = Some(ty.clone());
                            break;
                        }
                    }
                }
                else if module.anon_tuple_structs.contains_key(&struct_ident) {
                    let anon_tuple_struct = &module.anon_tuple_structs[&struct_ident];
                    for (field_ident,ty) in anon_tuple_struct {
                        if field_ident == ident {
                            found_ty = Some(ty.clone());
                            break;
                        }
                    }
                }
            }
            else {
                panic!("type of {} should be struct",expr);
            }
            if let Some(ty) = found_ty {
                ty
            }
            else {
                panic!("unable to find type of {}.{}",expr,ident);
            }
        },
        sr::Expr::Index(expr,_) => {
            let ty = infer_expr_type(module,anon_tuple_structs,expr);
            if let sr::Type::Array(ty,_) = ty {
                *ty.clone()
            }
            else {
                panic!("type of {} should be array",expr);
            }
        },
        sr::Expr::Cast(_,ty) => ty.clone(),
        sr::Expr::AnonTuple(exprs) => {
            let mut types: Vec<sr::Type> = Vec::new();
            for expr in exprs {
                types.push(infer_expr_type(module,anon_tuple_structs,expr));
            }
            let ident = make_anon_tuple_struct(module,anon_tuple_structs,types);
            sr::Type::Struct(ident)
        },
        sr::Expr::Neg(expr) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::Not(expr) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::Mul(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Div(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Mod(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Add(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Sub(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Shl(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Shr(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::And(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Or(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Xor(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Eq(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::NotEq(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Greater(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Less(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::GreaterEq(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::LessEq(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::LogAnd(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::LogOr(expr,expr2) => find_tightest_type(&infer_expr_type(module,anon_tuple_structs,expr),&infer_expr_type(module,anon_tuple_structs,expr2)).expect(&format!("{} and {} have incompatible types",expr,expr2)),
        sr::Expr::Assign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr2),
        sr::Expr::AddAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::SubAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::MulAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::DivAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::ModAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::AndAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::OrAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::XorAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::ShlAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::ShrAssign(expr,expr2) => infer_expr_type(module,anon_tuple_structs,expr),
        sr::Expr::Continue => sr::Type::Void,
        sr::Expr::Break(expr) => {
            if let Some(expr) = expr {
                infer_expr_type(module,anon_tuple_structs,expr)
            }
            else {
                sr::Type::Void
            }
        },
        sr::Expr::Return(expr) => {
            if let Some(expr) = expr {
                infer_expr_type(module,anon_tuple_structs,expr)
            }
            else {
                sr::Type::Void
            }
        },
        sr::Expr::Block(block) => {
            if let Some(expr) = &block.expr {
                infer_expr_type(module,anon_tuple_structs,&expr)
            }
            else {
                sr::Type::Void
            }
        },
        sr::Expr::If(expr,block,else_expr) => {
            let mut ty: sr::Type = sr::Type::Inferred;
            if let Some(expr) = &block.expr {
                ty = infer_expr_type(module,anon_tuple_structs,&expr);
            }
            if let Some(else_expr) = else_expr {
                ty = find_tightest_type(&ty,&infer_expr_type(module,anon_tuple_structs,else_expr)).expect(&format!("{} and {} have incompatible types",expr,else_expr));
            }
            ty
        },
        sr::Expr::IfLet(pats,expr,block,else_expr) => {
            let mut ty: sr::Type = sr::Type::Inferred;
            if let Some(expr) = &block.expr {
                ty = infer_expr_type(module,anon_tuple_structs,&expr);
            }
            if let Some(else_expr) = else_expr {
                ty = find_tightest_type(&ty,&infer_expr_type(module,anon_tuple_structs,else_expr)).expect(&format!("{} and {} have incompatible types",expr,else_expr));
            }
            ty
        },
        sr::Expr::Loop(block) => sr::Type::Void,
        sr::Expr::For(pats,range,block) => sr::Type::Void,
        sr::Expr::While(expr,block) => sr::Type::Void,
        sr::Expr::WhileLet(pats,expr,block) => sr::Type::Void,
        sr::Expr::Match(expr,arms) => {
            let mut ty: sr::Type = sr::Type::Inferred;
            for (pats,if_expr,expr) in arms {
                ty = find_tightest_type(&ty,&infer_expr_type(module,anon_tuple_structs,expr)).expect(&format!("in match expression, {} has incompatible type",expr));
            }
            ty
        },
    }
}
