use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_pat(&mut self) -> sr::Pat {
        if self.punct('_') {
            sr::Pat::Wildcard
        }
        else if self.punct2('.','.') {
            sr::Pat::Rest
        }
        else if let Some(literal) = self.literal() {
            sr::Pat::Literal(literal.to_string())
        }
        else if let Some(mut parser) = self.group('[') {
            let mut pats: Vec<sr::Pat> = Vec::new();
            while !parser.done() {
                pats.push(parser.parse_pat());
                parser.punct(',');
            }
            sr::Pat::Slice(pats)
        }
        else if let Some(symbol) = self.any_ident() {
            sr::Pat::Symbol(symbol)
        }
        else {
            panic!("pattern not supported");
        }
    }
}

impl Display for sr::Pat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            sr::Pat::Wildcard => write!(f,"_"),
            sr::Pat::Rest => write!(f,".."),
            sr::Pat::Literal(literal) => write!(f,"{}",literal),
            sr::Pat::Slice(pats) => {
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
            sr::Pat::Symbol(symbol) => write!(f,"{}",symbol),
        }
    }
}
