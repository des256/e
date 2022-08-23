use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_stat(&mut self) -> sr::Stat {
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
            sr::Stat::Let(pat,ty,expr)
        }
        else {
            let expr = self.parse_expr();
            self.punct(';');
            sr::Stat::Expr(Box::new(expr))
        }
    }
}

impl Display for sr::Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            sr::Stat::Let(pat,ty,expr) => {
                write!(f,"let {}",pat)?;
                if let Some(ty) = ty {
                    write!(f,": {}",ty)?;
                }
                if let Some(expr) = expr {
                    write!(f," = {}",expr)?;
                }
                write!(f,";")
            },
            sr::Stat::Expr(expr) => write!(f,"{}",expr),
        }
    }
}
