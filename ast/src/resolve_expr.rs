use super::*;

impl Resolver {

    // resolve expr when the type is already known
    pub fn resolve_should_expr(&mut self,expr: &Expr,should_type: &Type) -> Expr {
        match expr {
            _ => expr.clone(),
        }
    }

    // resolve expr when the type is not known
    pub fn resolve_expr(&mut self,expr: &Expr) -> Expr {
        match expr {
            _ => expr.clone(),
       }
    }
}