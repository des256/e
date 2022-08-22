use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_stat(&mut self) -> ast::Stat {
        if self.ident("let") {
            let pat = self.parse_pat();
            let ty = if self.punct(':') {
                Some(Box::new(self.parse_type()))
            }
            else {
                None
            };
            let expr = if self.punct('=') {
                Some(Box::new(self.parse_expr()))
            }
            else {
                None
            };
            ast::Stat::Let(pat,ty,expr)
        }
        else {
            let expr = self.parse_expr();
            self.punct(';');
            ast::Stat::Expr(Box::new(expr))
        }
    }
}

impl Display for ast::Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            ast::Stat::Let(pat,ty,expr) => {
                write!(f,"let {}",pat)?;
                if let Some(ty) = ty {
                    write!(f,": {}",ty)?;
                }
                if let Some(expr) = expr {
                    write!(f," = {}",expr)?;
                }
                write!(f,";")
            },
            ast::Stat::Expr(expr) => write!(f,"{}",expr),
        }
    }
}
