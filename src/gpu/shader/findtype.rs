use {
    sr::*,
    std::{
        rc::Rc,
    },
};

use ast::*;

pub trait FindType {
    fn find_type(&self) -> Type;
}

pub fn tightest(type1: &Type,type2: &Type) -> Option<Type> {

    let mut type1 = type1.clone();
    let mut type2 = type2.clone();
    while let Type::Alias(alias) = type1 {
        type1 = alias.borrow().type_.clone();
    }
    while let Type::Alias(alias) = type2 {
        type2 = alias.borrow().type_.clone();
    }
    
    if type1 == type2 {
        Some(type1.clone())
    }

    else if let Type::Inferred = type1 {
        Some(type2.clone())
    }
    else if let Type::Inferred = type2 {
        Some(type1.clone())
    }

    else if let Type::Integer = type1 {
        match type2 {
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
        }
    }
    else if let Type::Integer = type2 {
        match type1 {
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
        }
    }

    else if let Type::Float = type1 {
        match type2 {
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        }
    }
    else if let Type::Float = type2 {
        match type1 {
            Type::F16 => Some(Type::F16),
            Type::F32 => Some(Type::F32),
            Type::F64 => Some(Type::F64),
            _ => None,
        }
    }

    else {
        None
    }
}

impl FindType for Block {
    fn find_type(&self) -> Type {
        if let Some(expr) = &self.expr {
            expr.find_type()
        }
        else {
            Type::Void
        }
    }
}

impl FindType for Expr {
    fn find_type(&self) -> Type {
        match self {
            Expr::Boolean(_) => Type::Bool,
            Expr::Integer(_) => Type::Integer,
            Expr::Float(_) => Type::Float,
            Expr::Array(exprs) => {
                let mut type_ = Type::Inferred;
                for expr in exprs.iter() {
                    type_ = tightest(&type_,&expr.find_type()).expect(&format!("cannot infer type of array {}",self));
                }
                Type::Array(Box::new(type_),Box::new(Expr::Integer(exprs.len() as i64)))
            },
            Expr::Cloned(expr,expr2) => Type::Array(Box::new(expr.find_type()),expr2.clone()),
            Expr::Index(expr,_) => if let Type::Array(type_,_) = expr.find_type() { *type_.clone() } else { Type::Inferred },
            Expr::Cast(_,type_) => *type_.clone(),
            Expr::AnonTuple(exprs) => {
                let mut types: Vec<Type> = Vec::new();
                for expr in exprs {
                    types.push(expr.find_type());
                }
                Type::AnonTuple(types)
            },
            Expr::Unary(_,expr) => expr.find_type(),
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
                    BinaryOp::LogOr => tightest(&lhs.find_type(),&rhs.find_type()).expect(&format!("operands of {} not compatible",op)),
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
                    BinaryOp::ShrAssign => lhs.find_type(),
                }
            },
            Expr::Continue => Type::Void,
            Expr::Break(_) => Type::Void,
            Expr::Return(_) => Type::Void,
            Expr::Block(block) => block.find_type(),
            Expr::If(_,block,else_expr) => {
                let type_ = block.find_type();
                if let Some(else_expr) = else_expr {
                    tightest(&type_,&else_expr.find_type()).expect("if block and else-expression not compatible")
                }
                else {
                    type_
                }
            },
            Expr::While(_,_) => Type::Void,
            Expr::Loop(_) => Type::Void,
            Expr::IfLet(_,_,block,else_expr) => {
                let type_ = block.find_type();
                if let Some(else_expr) = else_expr {
                    tightest(&type_,&else_expr.find_type()).expect("if let block and else-expression not compatible")
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
                    type_ = tightest(&type_,&expr.find_type()).expect("match arms not compatible");
                }
                type_
            },
            Expr::UnknownIdent(_) => Type::Inferred,
            Expr::UnknownTupleOrCall(_,_) => Type::Inferred,
            Expr::UnknownStruct(_,_) => Type::Inferred,
            Expr::UnknownVariant(_,_) => Type::Inferred,
            Expr::UnknownMethod(_,_,_) => Type::Inferred,
            Expr::UnknownField(_,_) => Type::Inferred,
            Expr::UnknownTupleIndex(_,_) => Type::Inferred,
            Expr::Param(param) => param.borrow().type_.clone(),
            Expr::Local(local) => local.borrow().type_.clone(),
            Expr::Const(const_) => const_.borrow().type_.clone(),
            Expr::Tuple(tuple,_) => Type::Tuple(Rc::clone(&tuple)),
            Expr::Call(function,_) => function.borrow().type_.clone(),
            Expr::Struct(struct_,_) => Type::Struct(Rc::clone(&struct_)),
            Expr::Variant(enum_,_) => Type::Enum(Rc::clone(&enum_)),
            Expr::Method(_,method,_) => method.borrow().type_.clone(),
            Expr::Field(struct_,_,index) => struct_.borrow().fields[*index].type_.clone(),
            Expr::TupleIndex(tuple,_,index) => tuple.borrow().types[*index].clone(),
        }
    }
}
