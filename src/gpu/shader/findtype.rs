// find type of expression

/*
use crate::*;

pub struct Context {
    pub locals: Vec<String>,
    pub params: Vec<String>,
}

pub fn tightest(type1: &Type,type2: &Type) -> Option<Type> {
    if *type1 == *type2 {
        Some(type1.clone())
    }
    else {
        match type1 {
            Type::Inferred => Some(type2.clone()),
            Type::Integer => match type2 {
                Type::Inferred | Type::Integer => Some(Type::Integer),
                Type::Float => Some(Type::Float),
                Type::Base(base_type) => match base_type {
                    sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                    sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type2.clone()),
                    _ => None,
                },
                _ => None,
            },
            Type::Float => match type2 {
                Type::Inferred | Type::Integer | Type::Float => Some(Type::Float),
                Type::Base(base_type) => match base_type {
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type2.clone()),
                    _ => None,
                },
                _ => None,
            },
            _ => match type2 {
                Type::Inferred => Some(type1.clone()),
                Type::Integer => match type1 {
                    Type::Inferred | Type::Integer => Some(Type::Integer),
                    Type::Float => Some(Type::Float),
                    Type::Base(base_type) => match base_type {
                        sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                        sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                        sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type1.clone()),
                        _ => None,
                    },
                    _ => None,
                },
                Type::Float => match type1 {
                    Type::Inferred | Type::Integer | Type::Float => Some(Type::Float),
                    Type::Base(base_type) => match base_type {
                        sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type1.clone()),
                        _ => None,
                    },
                    _ => None,
                },
                _ => None,
            }
        }
    }
}

pub fn find_type(module: &Module,context: &mut Context,expr: &Expr) -> Type {
    match expr {
        Expr::Boolean(_) => Type::Base(sr::BaseType::Bool),
        Expr::Integer(_) => Type::Integer,
        Expr::Float(_) => Type::Float,
        Expr::Base(base_type,_) => Type::Base(base_type.clone()),
        Expr::Ident(_) => Type::Void,
        Expr::Local(_,ty) => ty.clone(),
        Expr::Param(_,ty) => ty.clone(),
        Expr::Const(_,ty) => ty.clone(),
        Expr::Array(exprs) => {
            let mut result = Type::Inferred;
            for expr in exprs {
                if let Some(ty) = tightest(&result,&find_type(module,context,expr)) {
                    result = ty;
                }
            }
            result
        },
        Expr::Cloned(expr,_) => find_type(module,context,expr),
        Expr::Struct(ident,_) => Type::Struct(ident.clone()),
        Expr::Tuple(ident,_) => Type::Tuple(ident.clone()),
        Expr::AnonTuple(exprs) => {
            let mut types: Vec<Type> = Vec::new();
            for expr in exprs.iter() {
                types.push(find_type(module,context,expr));
            }
            Type::AnonTuple(types)
        },
        Expr::Variant(ident,_) => Type::Enum(ident.clone()),
        Expr::Call(ident,_) => module.functions.borrow()[ident].1.borrow().clone(),
        Expr::Field(expr,ident) => {
            if let Type::Struct(struct_ident) = find_type(module,context,expr) {
                let mut result: Option<Type> = None;
                for (param_ident,ty) in module.structs.borrow()[&struct_ident].iter() {
                    if *param_ident == *ident {
                        result = Some(ty.borrow().clone());
                        break;
                    }
                }
                result.unwrap() // shouldn't happen
            }
            else {
                Type::Void // shouldn't happen
            }
        },
        Expr::TupleIndex(expr,index) => {
            if let Type::Tuple(tuple_ident) = find_type(module,context,expr) {
                module.tuples.borrow()[&tuple_ident][*index as usize].borrow().clone()
            }
            else {
                Type::Void // shouldn't happen
            }
        },
        Expr::Index(expr,_) => {
            if let Type::Array(ty,_) = find_type(module,context,expr) {
                *ty
            }
            else {
                Type::Void // shouldn't happen
            }
        },
        Expr::Cast(_,ty) => ty.clone(),
        Expr::Neg(expr) => find_type(module,context,expr),
        Expr::Not(expr) => find_type(module,context,expr),
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
        Expr::AddAssign(expr,expr2) |
        Expr::SubAssign(expr,expr2) |
        Expr::MulAssign(expr,expr2) |
        Expr::DivAssign(expr,expr2) |
        Expr::ModAssign(expr,expr2) |
        Expr::AndAssign(expr,expr2) |
        Expr::OrAssign(expr,expr2) |
        Expr::XorAssign(expr,expr2) |
        Expr::ShlAssign(expr,expr2) |
        Expr::ShrAssign(expr,expr2) => if let Some(ty) = tightest(&find_type(module,context,expr),&find_type(module,context,expr2)) {
            ty
        }
        else {
            Type::Void // this shouldn't happen
        },
        Expr::Assign(_,expr2) => find_type(module,context,expr2),
        Expr::Continue => Type::Void,
        Expr::Break(expr) => {
            if let Some(expr) = expr {
                find_type(module,context,expr)
            }
            else {
                Type::Void
            }
        },
        Expr::Return(expr) => {
            if let Some(expr) = expr {
                find_type(module,context,expr)
            }
            else {
                Type::Void
            }
        },
        Expr::Block(block) => {
            let local_frame = context.locals.clone();
            let result = if let Some(expr) = &block.expr {
                find_type(module,context,expr)
            }
            else {
                Type::Void
            };
            context.locals = local_frame;
            result
        },
        Expr::If(_,block,else_expr) | Expr::IfLet(_,_,block,else_expr) => {
            let local_frame = context.locals.clone();
            let mut result = if let Some(expr) = &block.expr {
                find_type(module,context,&expr)
            }
            else {
                Type::Void
            };
            context.locals = local_frame;
            if let Some(else_expr) = else_expr {
                result = if let Some(ty) = tightest(&result,&find_type(module,context,else_expr)) {
                    ty
                }
                else {
                    Type::Void // shouldn't happen
                }
            }
            result
        },
        Expr::Loop(block) => {
            let local_frame = context.locals.clone();
            let result = if let Some(expr) = &block.expr {
                find_type(module,context,&expr)
            }
            else {
                Type::Void
            };
            context.locals = local_frame;
            result
        },
        Expr::For(_,_,_) => {            
            // TODO: find tightest type of any Expr::Break in block
            Type::Void
        },
        Expr::While(_,_) | Expr::WhileLet(_,_,_) => {
            // TODO: find tightest type of any Expr::Break in block
            Type::Void
        },
        Expr::Match(_,arms) => {
            // TODO: find tightest type of all arm expr
            let mut result = Type::Inferred;
            for (_,_,expr) in arms {
                if let Some(ty) = tightest(&result,&find_type(module,context,expr)) {
                    result = ty;
                }
            }
            result
        },
    }
}
*/