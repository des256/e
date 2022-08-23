use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Parser {
    pub fn parse_type(&mut self) -> sr::Type {
        let mut result = if self.punct('_') {
            sr::Type::Inferred
        }
        else if let Some(symbol) = self.any_ident() {
            // generics are only strictly allowed to address basic types like Vec3<f32>, Vec4<u8>, etc. Any ::'s are removed.
            self.punct2(':',':');
            if self.punct('<') {
                let subtype = self.any_ident().unwrap();
                self.punct('>');
                sr::Type::Symbol(format!("{}<{}>",symbol,subtype))
            }
            else {
                sr::Type::Symbol(symbol)
            }
        }
        else if let Some(mut parser) = self.group('(') {
            let mut types: Vec<sr::Type> = Vec::new();
            while !parser.done() {
                types.push(parser.parse_type());
                parser.punct(',');
            }
            sr::Type::Tuple(types)
        }
        else {
            panic!("type not supported: {:?}",self.current);
        };
        while let Some(mut parser) = self.group('[') {
            let expr = parser.parse_expr();
            result = sr::Type::Array(Box::new(result),Box::new(expr));
        }
        result
    }
}

impl Display for sr::Type {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            sr::Type::Array(ty,expr) => write!(f,"{}[{}]",ty,expr),
            sr::Type::Tuple(types) => {
                write!(f,"(")?;
                let mut first_type = true;
                for ty in types {
                    if !first_type {
                        write!(f,",")?;
                    }
                    write!(f,"{}",ty)?;
                    first_type = false;
                }
                write!(f,")")
            },
            sr::Type::Symbol(symbol) => write!(f,"{}",symbol),
            sr::Type::Inferred => write!(f,"_"),
        }
    }
}
