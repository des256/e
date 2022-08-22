use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_item(&mut self) -> ast::Item {
        if self.ident("mod") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut items: Vec<ast::Item> = Vec::new();
            if let Some(mut parser) = self.group('{') {
                while !parser.done() {
                    items.push(parser.parse_item());
                }
            }
            else {
                self.punct(';');
            }
            ast::Item::Module(symbol,items)
        }
        else if self.ident("fn") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut params: Vec<(ast::Pat,Box<ast::Type>)> = Vec::new();
            if let Some(mut parser) = self.group('(') {
                while !parser.done() {
                    let pat = parser.parse_pat();
                    parser.punct(':');
                    let ty = parser.parse_type();
                    params.push((pat,Box::new(ty)));
                    parser.punct(';');
                }
            }
            let return_ty = if self.punct2('-','>') {
                Some(Box::new(self.parse_type()))
            }
            else {
                None
            };
            let mut stats: Vec<ast::Stat> = Vec::new();
            let mut parser = self.group('{').expect("{ expected");
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            ast::Item::Function(symbol,params,return_ty,stats)
        }
        else {
            panic!("item not supported");
        }
    }
}

impl Display for ast::Item {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            ast::Item::Module(symbol,items) => {
                write!(f,"mod {} {{ ",symbol)?;
                for item in items {
                    write!(f,"{} ",item)?;
                }
                write!(f,"}}")
            },
            ast::Item::Function(symbol,params,return_ty,stats) => {
                write!(f,"fn {}(",symbol)?;
                let mut first_param = true;
                for (pat,ty) in params {
                    if !first_param {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",pat,ty)?;
                    first_param = false;
                }
                write!(f,") ")?;
                if let Some(ty) = return_ty {
                    write!(f,"-> {} ",ty)?;
                }
                write!(f,"{{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
        }
    }
}
