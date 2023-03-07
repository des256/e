use super::*;

impl Parser {

    pub(crate) fn type_(&mut self) -> Type {
        if let Some(ident) = self.ident() {
            match ident.as_str() {
                "bool" => Type::Bool,
                "u8" => Type::U8,
                "i8" => Type::I8,
                "u16" => Type::U16,
                "i16" => Type::I16,
                "u32" => Type::U32,
                "i32" => Type::I32,
                "u64" => Type::U64,
                "i64" => Type::I64,
                "f16" => Type::F16,
                "f32" => Type::F32,
                "f64" => Type::F64,
                _ => {
                    self.punct2(':',':');
                    if self.punct('<') {
                        let type_ = self.type_();
                        self.punct('>');
                        Type::Struct(format!("{}<{}>",ident,type_))
                    }
                    else {
                        Type::UnknownStructTupleEnumAlias(ident)
                    }
                }
            }
        }

        else if let Some(types) = self.paren_types() {
            Type::AnonTuple(types)
        }

        else if let Some(mut parser) = self.group('[') {
            let type_ = parser.type_();
            parser.punct(';');
            let expr = parser.expr();
            Type::Array(Box::new(type_),Box::new(expr))
        }

        else {
            self.fatal(&format!("Type expected"));
        }
    }
}
