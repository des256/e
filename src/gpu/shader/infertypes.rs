// for each block statement/expression, figure out the type of the expression

use {
    crate::*,
    std::rc::Rc,
};

pub fn find_tightest_type(type1: &sr::Type,type2: &sr::Type) -> Option<sr::Type> {

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

pub fn infer_block_type(block: &sr::Block) -> sr::Type {
    for stat in block.stats.iter() {
        infer_stat_type(stat);
    }
    let mut result = sr::Type::Void;
    if let Some(expr) = block.expr.as_ref() {
        result = infer_expr_type(&expr);
    }
    result
}

pub fn infer_expr_type(expr: &sr::Expr) -> sr::Type {
    match expr {
        sr::Expr::Boolean(_) => sr::Type::Boolean,
        sr::Expr::Integer(_) => sr::Type::Integer,
        sr::Expr::Float(_) => sr::Type::Float,
        sr::Expr::Base(bt,_) => sr::Type::Base(bt.clone()),
        sr::Expr::UnknownIdent(ident) => panic!("unknown identifier {}",ident),
        sr::Expr::Const(const_) => const_.type_.clone(),
        sr::Expr::Local(local) => local.type_.clone(),
        sr::Expr::Param(param) => param.type_.clone(),
        sr::Expr::Array(exprs) => {
            let mut type_ = sr::Type::Inferred;
            for expr in exprs.iter() {
                type_ = find_tightest_type(&type_,&infer_expr_type(expr)).expect(&format!("array element types incompatible at {}",expr));
            }
            sr::Type::Array(Box::new(type_),Box::new(sr::Expr::Integer(exprs.len() as i64)))
        },
        sr::Expr::Cloned(expr,_) => infer_expr_type(expr),
        sr::Expr::UnknownStruct(ident,_) => panic!("unknown struct {}",ident),
        sr::Expr::Struct(struct_,_) => sr::Type::Struct(Rc::clone(&struct_)),
        sr::Expr::UnknownCall(ident,_) => panic!("unknown function call {}",ident),
        sr::Expr::Call(function,_) => function.return_type.clone(),
        sr::Expr::UnknownVariant(ident,variantexpr) => panic!("unknown enum variant {}::{}",ident,variantexpr),
        sr::Expr::Variant(enum_,_) => sr::Type::Enum(Rc::clone(&enum_)),
        sr::Expr::Field(expr,ident) => if let sr::Type::Struct(struct_) = infer_expr_type(expr) {
            let mut type_ = sr::Type::Inferred;
            for field in struct_.fields.iter() {
                if field.ident == *ident {
                    type_ = field.type_.clone();
                    break;
                }
            }
            if let sr::Type::Inferred = type_ {
                panic!("unknown field {} of struct {}",ident,struct_.ident);
            }
            type_
        }
        else {
            panic!("cannot access field of non-struct {}",expr);
        },
        sr::Expr::Index(expr,_) => if let sr::Type::Array(type_,_) = infer_expr_type(expr) {
            *type_
        }
        else {
            panic!("cannot index non-array {}",expr);
        },
        sr::Expr::Cast(_,type_) => *type_.clone(),
        sr::Expr::AnonTuple(_) => sr::Type::Inferred, // TODO: this should cause a problem at the top where a type will be forced onto the AnonTuple from a function param or return value
        sr::Expr::Neg(expr) => infer_expr_type(expr),
        sr::Expr::Not(expr) => infer_expr_type(expr),
        sr::Expr::Mul(expr,expr2) |
        sr::Expr::Div(expr,expr2) |
        sr::Expr::Mod(expr,expr2) |
        sr::Expr::Add(expr,expr2) |
        sr::Expr::Sub(expr,expr2) |
        sr::Expr::Shl(expr,expr2) |
        sr::Expr::Shr(expr,expr2) |
        sr::Expr::And(expr,expr2) |
        sr::Expr::Or(expr,expr2) |
        sr::Expr::Xor(expr,expr2) |
        sr::Expr::Assign(expr,expr2) |
        sr::Expr::AddAssign(expr,expr2) |
        sr::Expr::SubAssign(expr,expr2) |
        sr::Expr::MulAssign(expr,expr2) |
        sr::Expr::DivAssign(expr,expr2) |
        sr::Expr::ModAssign(expr,expr2) |
        sr::Expr::AndAssign(expr,expr2) |
        sr::Expr::OrAssign(expr,expr2) |
        sr::Expr::XorAssign(expr,expr2) |
        sr::Expr::ShlAssign(expr,expr2) |
        sr::Expr::ShrAssign(expr,expr2) => find_tightest_type(&infer_expr_type(expr),&infer_expr_type(expr2)).expect(&format!("types of {} and {} incompatible",expr,expr2)),
        sr::Expr::Eq(expr,expr2) |
        sr::Expr::NotEq(expr,expr2) |
        sr::Expr::Greater(expr,expr2) |
        sr::Expr::Less(expr,expr2) |
        sr::Expr::GreaterEq(expr,expr2) |
        sr::Expr::LessEq(expr,expr2) |
        sr::Expr::LogAnd(expr,expr2) |
        sr::Expr::LogOr(expr,expr2) => {
            find_tightest_type(&infer_expr_type(expr),&infer_expr_type(expr2)).expect(&format!("types of {} and {} incompatible",expr,expr2));
            sr::Type::Boolean
        },
        sr::Expr::Continue => sr::Type::Void,
        sr::Expr::Break(expr) |
        sr::Expr::Return(expr) => if let Some(expr) = expr {
            infer_expr_type(expr)
        }
        else {
            sr::Type::Void
        },
        sr::Expr::Block(block) => infer_block_type(block),
        sr::Expr::If(expr,block,else_expr) => {
            if let sr::Type::Boolean = infer_expr_type(expr) {
                let mut type_ = infer_block_type(block);
                if let Some(else_expr) = else_expr {
                    type_ = find_tightest_type(&type_,&infer_expr_type(else_expr)).expect(&format!("types of {} and {} incompatible",expr,else_expr));
                }
                type_
            }
            else {
                panic!("if-expression {} should be boolean",expr);
            }
        },
        sr::Expr::IfLet(_,expr,block,else_expr) => {
            // TODO: patterns...
            let mut type_ = infer_block_type(block);
            if let Some(else_expr) = else_expr {
                type_ = find_tightest_type(&type_,&infer_expr_type(else_expr)).expect(&format!("types of {} and {} incompatible",expr,else_expr));
            }
            type_
        },
        sr::Expr::Loop(block) => infer_block_type(block),
        sr::Expr::For(_,_,block) => {
            // TODO: patterns...
            infer_block_type(block)
        },
        sr::Expr::While(expr,block) => {
            if let sr::Type::Boolean = infer_expr_type(expr) {
                infer_block_type(block)
            }
            else {
                panic!("while-expression {} should be boolean",expr);
            }
        },
        sr::Expr::WhileLet(_,_,block) => {
            // TODO: patterns...
            infer_block_type(block)
        },
        sr::Expr::Match(_,_) => {
            // TODO: patterns...
            sr::Type::Void
        },
    }
}

pub fn infer_stat_type(stat: &sr::Stat) -> sr::Type {
    match stat {
        sr::Stat::Let(variable) => {
            if let Some(expr) = &variable.value {
                let mut type_ = infer_expr_type(expr);
                type_ = find_tightest_type(&variable.type_,&type_).expect(&format!("local variable {} not compatible with expression {}",variable.ident,expr));
                type_
            }
            else {
                variable.type_.clone()
            }
        },
        sr::Stat::Expr(expr,_) => {
            infer_expr_type(expr)
        },
    }
}
