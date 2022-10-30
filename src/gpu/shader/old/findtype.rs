use {
    super::*,
    super::ast::*,
};

pub trait FindType {
    fn find_type(&self,context: &Context) -> Type;
}

pub fn tightest(type1: &Type,type2: &Type) -> Option<Type> {
    match (type1,type2) {
        (Type::Alias,_) | (_,Type::Alias) => None,
        (Type::UnknownIdent(_),_) | (_,Type::UnknownIdent(_)) => None,
        (Type::Tuple(_),_) | (_,Type::Tuple(_)) => None,
        (Type::Enum(_),_) | (_,Type::Enum(_)) => None,
        (Type::Inferred,_) => Some(type2.clone()),
        (_,Type::Inferred) => Some(type1.clone()),
        (Type::Integer,_) => match type2 {
            Type::Float => Some(Type::Float),
            Type::U8 => Some(Type::U8),
            Type::I8 => Some(Type::I8),
            Type::U16 => Some(Type::U16),
            Type::I16 => Some(Type::I16),
            Type::U32 => Some(Type::U32),
            Type::I32 => Some(Type::I32),
            Type::U64 => Some(Type::U64),
            Type::I64 => Some(Type::I64),
            Type::USize => Some(Type::USize),
            Type::ISize => Some(Type::ISize),
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        },
        (_,Type::Integer) => match type1 {
            Type::Float => Some(Type::Float),
            Type::U8 => Some(Type::U8),
            Type::I8 => Some(Type::I8),
            Type::U16 => Some(Type::U16),
            Type::I16 => Some(Type::I16),
            Type::U32 => Some(Type::U32),
            Type::I32 => Some(Type::I32),
            Type::U64 => Some(Type::U64),
            Type::I64 => Some(Type::I64),
            Type::USize => Some(Type::USize),
            Type::ISize => Some(Type::ISize),
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        },
        (Type::Float,_) => match type2 {
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        },
        (_,Type::Float) => match type1 {
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        },
        _ => if type1 == type2 {
            Some(type1.clone())
        }
        else {
            None
        },
    }
}

impl FindType for Block {
    fn find_type(&self,context: &Context) -> Type {
        if let Some(expr) = &self.expr {
            expr.find_type(context)
        }
        else {
            Type::Void
        }
    }
}

impl FindType for Expr {
    fn find_type(&self,context: &Context) -> Type {
        match self {
            Expr::Boolean(_) => Type::Bool,
            Expr::Integer(_) => Type::Integer,
            Expr::Float(_) => Type::Float,
            Expr::Array(exprs) => {
                let mut type_ = Type::Inferred;
                for expr in exprs.iter() {
                    type_ = tightest(&type_,&expr.find_type(context)).expect(&format!("cannot infer type of array {}",self));
                }
                Type::Array(Box::new(type_),Box::new(Expr::Integer(exprs.len() as i64)))
            },
            Expr::Cloned(expr,expr2) => Type::Array(Box::new(expr.find_type(context)),expr2.clone()),
            Expr::Index(expr,_) => if let Type::Array(type_,_) = expr.find_type(context) { *type_.clone() } else { Type::Inferred },
            Expr::Cast(_,type_) => *type_.clone(),
            Expr::AnonTuple(exprs) => {
                let mut types: Vec<Type> = Vec::new();
                for expr in exprs {
                    types.push(expr.find_type(context));
                }
                Type::AnonTuple(types)
            },
            Expr::Unary(_,expr) => expr.find_type(context),
            Expr::Binary(lhs,op,rhs) => {
                match op {
                    BinaryOp::Mul |
                    BinaryOp::Div |
                    BinaryOp::Mod |
                    BinaryOp::Add |
                    BinaryOp::Sub |
                    BinaryOp::Shl |
                    BinaryOp::Shr |
                    BinaryOp::And |
                    BinaryOp::Or |
                    BinaryOp::Xor |
                    BinaryOp::LogAnd |
                    BinaryOp::LogOr => tightest(&lhs.find_type(context),&rhs.find_type(context)).expect(&format!("operands of {} not compatible",op)),
                    BinaryOp::Eq |
                    BinaryOp::NotEq |
                    BinaryOp::Greater |
                    BinaryOp::Less |
                    BinaryOp::GreaterEq |
                    BinaryOp::LessEq => Type::Bool,
                    BinaryOp::Assign |
                    BinaryOp::AddAssign |
                    BinaryOp::SubAssign |
                    BinaryOp::MulAssign |
                    BinaryOp::DivAssign |
                    BinaryOp::ModAssign |
                    BinaryOp::AndAssign |
                    BinaryOp::OrAssign |
                    BinaryOp::XorAssign |
                    BinaryOp::ShlAssign |
                    BinaryOp::ShrAssign => lhs.find_type(context),
                }
            },
            Expr::Continue => Type::Void,
            Expr::Break(_) => Type::Void,
            Expr::Return(_) => Type::Void,
            Expr::Block(block) => block.find_type(context),
            Expr::If(_,block,else_expr) => {
                let type_ = block.find_type(context);
                if let Some(else_expr) = else_expr {
                    tightest(&type_,&else_expr.find_type(context)).expect("if block and else-expression not compatible")
                }
                else {
                    type_
                }
            },
            Expr::While(_,_) => Type::Void,
            Expr::Loop(_) => Type::Void,
            Expr::IfLet(_,_,block,else_expr) => {
                let type_ = block.find_type(context);
                if let Some(else_expr) = else_expr {
                    tightest(&type_,&else_expr.find_type(context)).expect("if let block and else-expression not compatible")
                }
                else {
                    type_
                }
            },
            Expr::For(_,_,_) => Type::Void,
            Expr::WhileLet(_,_,_) => Type::Void,
            Expr::Match(_,arms) => {
                let mut type_ = Type::Inferred;
                for (_,_,expr) in arms {
                    type_ = tightest(&type_,&expr.find_type(context)).expect("match arms not compatible");
                }
                type_
            },
            Expr::UnknownIdent(_) => Type::Inferred,
            Expr::TupleOrCall(_,_) => Type::Inferred,
            Expr::Struct(ident,_) => Type::Struct(ident.clone()),
            Expr::Variant(ident,_) => Type::Enum(ident.clone()),
            Expr::Method(from_expr,ident,_) => {
                let type_ = from_expr.find_type(context);
                if context.stdlib.methods.contains_key(ident) {
                    let mut found: Option<Type> = None;
                    let methods = context.stdlib.methods[ident];
                    for method in methods.iter() {
                        if method.from_type == type_ {
                            found = Some(method.type_);
                            break;
                        }
                    }
                    if let Some(type_) = found {
                        type_
                    }
                    else {
                        panic!("method {} invalid for {}",ident,from_expr);
                    }
                }
                else {
                    panic!("unknown method {}",ident);
                }
            },
            Expr::Field(expr,ident) => if let Type::Struct(struct_ident) = expr.find_type(context) {
                if context.stdlib.structs.contains_key(&struct_ident) {
                    let mut found: Option<Type> = None;
                    let fields = context.stdlib.structs[&struct_ident].fields;
                    for field in fields.iter() {
                        if field.ident == *ident {
                            found = Some(field.type_);
                            break;
                        }
                    }
                    if let Some(type_) = found {
                        type_
                    }
                    else {
                        panic!("field {} does not exist on {}",ident,expr);
                    }
                }
                else if context.structs.contains_key(&struct_ident) {
                    let mut found: Option<Type> = None;
                    let fields = context.stdlib.structs[&struct_ident].fields;
                    for field in fields.iter() {
                        if field.ident == *ident {
                            found = Some(field.type_);
                            break;
                        }
                    }
                    if let Some(type_) = found {
                        type_
                    }
                    else {
                        panic!("field {} does not exist on {}",ident,expr);
                    }
                }
                else {
                    panic!("unknown struct in {} ({})",expr,struct_ident);
                }
            }
            else {
                panic!("{} not a struct",expr);
            },
            Expr::Param(ident) => if context.params.contains_key(ident) {
                context.params[ident].type_
            }
            else {
                panic!("unknown parameter {}",ident);
            },
            Expr::Local(ident) => if context.locals.contains_key(ident) {
                context.locals[ident].type_
            }
            else {
                panic!("unknown local variable {}",ident);
            },
            Expr::Const(ident) => if context.stdlib.consts.contains_key(ident) {
                context.stdlib.consts[ident].type_
            }
            else if context.consts.contains_key(ident) {
                context.consts[ident].type_
            }
            else {
                panic!("unknown constant {}",ident);
            },
            Expr::Tuple(ident,_) => Type::Tuple(ident.clone()),
            Expr::Call(ident,exprs) => if context.stdlib.functions.contains_key(ident) {
                let functions = context.stdlib.functions[ident];
                let mut found: Option<Type> = None;
                for function in functions.iter() {
                    let mut params_same = true;
                    if exprs.len() != function.params.len() {
                        params_same = false;
                    }
                    else {
                        for i in 0..exprs.len() {
                            if let None = tightest(&exprs[i].find_type(context),&function.params[i].type_,context) {
                                params_same = false;
                                break;
                            }
                        }
                        if params_same {
                            found = Some(function.type_.clone());
                            break;
                        }
                    }
                }
                if let Some(type_) = found {
                    type_
                }
                else {
                    panic!("function {} not found for given parameters",ident);
                }
            }
            else if context.functions.contains_key(ident) {
                let function = context.functions[ident];
                let mut params_same = true;
                if exprs.len() == function.params.len() {
                    for i in 0..exprs.len() {
                        if let None = tightest(&exprs[i].find_type(context),&function.params[i].type_,context) {
                            params_same = false;
                            break;
                        }
                    }
                    if params_same {
                        function.type_.clone()
                    }
                    else {
                        panic!("function {} parameters invalid",ident);
                    }
                }
                else {
                    panic!("function {} number of parameters wrong",ident);
                }
            }
            else {
                panic!("unknown function {}",ident);
            },
            Expr::Discriminant(_) => Type::USize,
            Expr::Destructure(expr,variant_index,index) => if let Type::Enum(ident) = expr.find_type(context) {
                if context.stdlib.enums.contains_key(&ident) {
                    let enum_ = context.stdlib.enums[&ident];
                    match enum_.variants[*variant_index] {
                        Variant::Naked(_) => Type::Void,
                        Variant::Tuple(_,types) => types[*index].clone(),
                        Variant::Struct(_,fields) => fields[*index].type_.clone(),
                    }
                }
                else if context.enums.contains_key(&ident) {
                    let enum_ = context.enums[&ident];
                    match enum_.variants[*variant_index] {
                        Variant::Naked(_) => Type::Void,
                        Variant::Tuple(_,types) => types[*index].clone(),
                        Variant::Struct(_,fields) => fields[*index].type_.clone(),
                    }
                }
                else {
                    panic!("unknown enum {}",ident);
                }
            }
            else {
                panic!("expression {} not an enum",expr);
            },
        }
    }
}
