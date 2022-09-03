// evaluate expression to various types

use crate::*;

pub fn evaluate_integer(module: &Module,expr: &Expr) -> Option<isize> {
    match expr {
        Expr::Integer(value) => Some(value as isize),
        Expr::Ident(ident) => panic!("evaluate_integer: Expr::Ident should already be resolved"),
        Expr::Const(ident,ty) => evaluate_integer(module.consts[ident].1),
        Expr::Tuple(ident,exprs) => panic!("evaluate_integer: Expr::Tuple should already be converted"),
        Expr::AnonTuple(exprs) => panic!("evaluate_integer: Expr::AnonTuple should already be converted"),
        Expr::Variant(ident,variant) => panic!("evaluate_integer: TODO: Expr::Variant"),
        Expr::Call(ident,exprs) => panic!("evaluate_integer: TODO: evaluate function call?"),
        Expr::TupleIndex(expr,index) => panic!("evaluate_integer: Expr::TupleIndex should already be converted"),
        Expr::Index(expr,expr2) => panic!("evaluate_integer: TODO: find constant array element"),
        Expr::Cast(expr,ty) => evaluate_integer(module,expr),
        Expr::Neg(expr) => if let Some(result) = evaluate_integer(module,expr) {
            Some(-result)
        }
        else {
            None
        },
        Expr::Not(expr) => if let Some(result) = evaluate_integer(module,expr) {
            Some(!result)
        }
        else {
            None
        },
        Expr::Mul(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result * result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::Div(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result / result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::Mod(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result % result2)
            }
            else {
                None
            }
        }
        else {
            None
        },        
        Expr::Add(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result + result2)
            }
            else {
                None
            }
        }
        else {
            None
        },        
        Expr::Sub(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result - result2)
            }
            else {
                None
            }
        }
        else {
            None
        },        
        Expr::Shl(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result << result2)
            }
            else {
                None
            }
        }
        else {
            None
        },        
        Expr::Shr(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result >> result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::And(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result & result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::Or(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result | result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::Xor(expr,expr2) => if let Some(result) = evaluate_integer(module,expr) {
            if let Some(result2) = evaluate_integer(module,expr2) {
                Some(result ^ result2)
            }
            else {
                None
            }
        }
        else {
            None
        },
        Expr::Block(block) => if let Some(expr) = block.expr {
            evaluate_integer(module,block.expr)
        }
        else {
            None
        },
        Expr::If(expr,block,else_expr) | Expr::IfLet(_,expr,block,else_expr) => panic!("evaluate_integer: TODO: Expr::If or Expr::IfLet"),
        Expr::Match(expr,arms) => panic!("evaluate_integer: TODO: Expr::Match"),
    }
}
