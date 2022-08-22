use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_pat(&mut self) -> ast::Pat {
        if self.punct('_') {
            ast::Pat::Wildcard
        }
        else if self.punct2('.','.') {
            ast::Pat::Rest
        }
        else if let Some(literal) = self.literal() {
            ast::Pat::Literal(literal.to_string())
        }
        else if let Some(mut parser) = self.group('[') {
            let mut pats: Vec<ast::Pat> = Vec::new();
            while !parser.done() {
                pats.push(parser.parse_pat());
                parser.punct(',');
            }
            ast::Pat::Slice(pats)
        }
        else if let Some(symbol) = self.any_ident() {
            ast::Pat::Symbol(symbol)
        }
        else {
            panic!("pattern not supported");
        }
    }
}

impl Display for ast::Pat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            ast::Pat::Wildcard => write!(f,"_"),
            ast::Pat::Rest => write!(f,".."),
            ast::Pat::Literal(literal) => write!(f,"{}",literal),
            ast::Pat::Slice(pats) => {
                write!(f,"[")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,",")?;
                    }
                    write!(f,"{}",pat)?;
                    first_pat = false;
                }
                write!(f,"]")
            },
            ast::Pat::Symbol(symbol) => write!(f,"{}",symbol),
        }
    }
}
