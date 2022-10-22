use {
    super::ast::*,
};

pub trait EvalBool {
    fn eval_bool(&self) -> Option<bool>;
}

pub trait EvalInt {
    fn eval_int(&self) -> Option<i64>;
}

pub trait EvalFloat {
    fn eval_float(&self) -> Option<f64>;
}

impl EvalBool for Expr {
    fn eval_bool(&self) -> Option<bool> {
        match self {
            Expr::Boolean(value) => Some(*value),
            Expr::AnonTuple(exprs) => {
                if exprs.len() == 1 {
                    if let Some(value) = exprs[0].eval_bool() {
                        Some(value)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                }
            },
            Expr::Unary(op,expr) => match op {
                UnaryOp::Not => if let Some(value) = expr.eval_bool() {
                    Some(!value)
                }
                else {
                    None
                },
                _ => None,
            },
            Expr::Binary(expr,op,expr2) => match op {
                BinaryOp::LogAnd => if let Some(value) = expr.eval_bool() {
                    if let Some(value2) = expr2.eval_bool() {
                        Some(value && value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::LogOr => if let Some(value) = expr.eval_bool() {
                    if let Some(value2) = expr2.eval_bool() {
                        Some(value || value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Eq => if let Some(value) = expr.eval_bool() {
                    if let Some(value2) = expr2.eval_bool() {
                        Some(value == value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value == value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value == value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::NotEq => if let Some(value) = expr.eval_bool() {
                    if let Some(value2) = expr2.eval_bool() {
                        Some(value != value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value != value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value != value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Greater => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value > value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value > value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Less => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value < value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value < value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::GreaterEq => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value >= value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value >= value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::LessEq => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr.eval_int() {
                        Some(value <= value2)
                    }
                    else {
                        None
                    }
                }
                else if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr.eval_float() {
                        Some(value <= value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Assign => expr2.eval_bool(),
                _ => None,
            },
            Expr::Block(block) => if let Some(expr) = block.expr {
                expr.eval_bool()
            }
            else {
                None
            },
            Expr::If(expr,block,else_expr) => if let Some(value) = expr.eval_bool() {
                if value {
                    if let Some(expr) = block.expr {
                        expr.eval_bool()
                    }
                    else {
                        None
                    }    
                }
                else {
                    if let Some(else_expr) = else_expr {
                        else_expr.eval_bool()
                    }
                    else {
                        None
                    }
                }
            }
            else {
                None
            },
            _ => None,
        }
    }
}

impl EvalInt for Expr {
    fn eval_int(&self) -> Option<i64> {
        match self {
            Expr::Integer(value) => Some(*value),
            Expr::AnonTuple(exprs) => {
                if exprs.len() == 1 {
                    if let Some(value) = exprs[0].eval_int() {
                        Some(value)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                }
            },
            Expr::Unary(op,expr) => match op {
                UnaryOp::Neg => if let Some(value) = expr.eval_int() {
                    Some(-value)
                }
                else {
                    None
                },
                _ => None,
            },
            Expr::Binary(expr,op,expr2) => match op {
                BinaryOp::Mul | BinaryOp::MulAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value * value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Div | BinaryOp::DivAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value / value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Mod | BinaryOp::ModAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value % value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Add | BinaryOp::AddAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value + value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Sub | BinaryOp::SubAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value - value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Shl | BinaryOp::ShlAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value << value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Shr | BinaryOp::ShrAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value >> value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::And | BinaryOp::AndAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value & value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Or | BinaryOp::OrAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value | value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Xor | BinaryOp::XorAssign => if let Some(value) = expr.eval_int() {
                    if let Some(value2) = expr2.eval_int() {
                        Some(value ^ value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Assign => expr2.eval_int(),
                _ => None,
            },
            Expr::Block(block) => if let Some(expr) = block.expr {
                expr.eval_int()
            }
            else {
                None
            },
            Expr::If(expr,block,else_expr) => if let Some(value) = expr.eval_bool() {
                if value {
                    if let Some(expr) = block.expr {
                        expr.eval_int()
                    }
                    else {
                        None
                    }    
                }
                else {
                    if let Some(else_expr) = else_expr {
                        else_expr.eval_int()
                    }
                    else {
                        None
                    }
                }
            }
            else {
                None
            },
            _ => None,
       }
    }
}

impl EvalFloat for Expr {
    fn eval_float(&self) -> Option<f64> {
        match self {
            Expr::Integer(value) => Some(*value as f64),
            Expr::Float(value) => Some(*value),
            Expr::AnonTuple(exprs) => {
                if exprs.len() == 1 {
                    if let Some(value) = exprs[0].eval_float() {
                        Some(value)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                }
            },
            Expr::Unary(op,expr) => match op {
                UnaryOp::Neg => if let Some(value) = expr.eval_float() {
                    Some(-value)
                }
                else {
                    None
                },
                _ => None,
            },
            Expr::Binary(expr,op,expr2) => match op {
                BinaryOp::Mul | BinaryOp::MulAssign => if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr2.eval_float() {
                        Some(value * value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Div | BinaryOp::DivAssign => if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr2.eval_float() {
                        Some(value / value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Add | BinaryOp::AddAssign => if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr2.eval_float() {
                        Some(value + value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Sub | BinaryOp::SubAssign => if let Some(value) = expr.eval_float() {
                    if let Some(value2) = expr2.eval_float() {
                        Some(value - value2)
                    }
                    else {
                        None
                    }
                }
                else {
                    None
                },
                BinaryOp::Assign => expr2.eval_float(),
                _ => None,
            },
            Expr::Block(block) => if let Some(expr) = block.expr {
                expr.eval_float()
            }
            else {
                None
            },
            Expr::If(expr,block,else_expr) => if let Some(value) = expr.eval_bool() {
                if value {
                    if let Some(expr) = block.expr {
                        expr.eval_float()
                    }
                    else {
                        None
                    }    
                }
                else {
                    if let Some(else_expr) = else_expr {
                        else_expr.eval_float()
                    }
                    else {
                        None
                    }
                }
            }
            else {
                None
            },
            _ => None,
       }
    }
}
