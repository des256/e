use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_item(&mut self) -> sr::Item {
        if self.ident("pub") {
            self.group('(');
        }
        if self.ident("mod") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut items: Vec<sr::Item> = Vec::new();
            if let Some(mut parser) = self.group('{') {
                while !parser.done() {
                    items.push(parser.parse_item());
                }
            }
            else {
                self.punct(';');
            }
            sr::Item::Module(symbol,items)
        }
        else if self.ident("fn") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut params: Vec<(sr::Pat,Box<sr::Type>)> = Vec::new();
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
            let mut stats: Vec<sr::Stat> = Vec::new();
            let mut parser = self.group('{').expect("{ expected");
            while !parser.done() {
                stats.push(parser.parse_stat());
            }
            sr::Item::Function(symbol,params,return_ty,stats)
        }
        else if self.ident("struct") {
            let symbol = self.any_ident().expect("identifier expected");
            let mut fields: Vec<(String,Box<sr::Type>)> = Vec::new();
            let mut parser = self.group('{').expect("{ expected");
            while !parser.done() {
                if parser.ident("pub") {
                    parser.group('(');
                }
                let symbol = parser.any_ident().expect("identifier expected");
                parser.punct(':');
                let ty = parser.parse_type();
                fields.push((symbol,Box::new(ty)));
                parser.punct(',');
            }
            sr::Item::Struct(symbol,fields)
        }
        else {
            panic!("item not supported");
        }
    }
}

impl Display for sr::Item {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            sr::Item::Module(symbol,items) => {
                write!(f,"mod {} {{ ",symbol)?;
                for item in items {
                    write!(f,"{} ",item)?;
                }
                write!(f,"}}")
            },
            sr::Item::Function(symbol,params,return_ty,stats) => {
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
            sr::Item::Struct(symbol,fields) => {
                write!(f,"struct {} {{",symbol)?;
                let mut first_field = true;
                for (symbol,ty) in fields {
                    if !first_field {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",symbol,ty)?;
                    first_field = false;
                }
                write!(f,"}}")
            },
        }
    }
}
