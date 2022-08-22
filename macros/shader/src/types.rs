use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_type(&mut self) -> ast::Type {
        let mut result = if self.punct('_') {
            ast::Type::Inferred
        }
        else if let Some(mut symbol) = self.any_ident() {
            if self.punct('<') {
                let subtype = self.any_ident().unwrap();
                self.punct('>');
                ast::Type::Symbol(format!("{}<{}>",symbol,subtype))
            }
            else {
                ast::Type::Symbol(symbol)
            }
        }
        else {
            panic!(format!("type not supported: {:?}",self.current));
        };
        while let Some(subparser) = self.group('[') {
            let expr = self.parse_expr();
            result = ast::Type::Array(Box::new(result),Box::new(expr));
        }
        result
    }
}

impl Display for ast::Type {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            ast::Type::Array(ty,expr) => write!(f,"{}[{}]",ty,expr),
            ast::Type::Symbol(symbol) => write!(f,"{}",symbol),
            ast::Type::Inferred => write!(f,"_"),
        }
    }
}
