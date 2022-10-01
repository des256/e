use {
    crate::*,
    sr::*,
};

impl Parser {

    pub(crate) fn type_(&mut self) -> ast::Type {
        if let Some(ident) = self.ident() {
            match ident.as_str() {
                "bool" => ast::Type::Bool,
                "u8" => ast::Type::U8,
                "i8" => ast::Type::I8,
                "u16" => ast::Type::U16,
                "i16" => ast::Type::I16,
                "u32" => ast::Type::U32,
                "i32" => ast::Type::I32,
                "u64" => ast::Type::U64,
                "i64" => ast::Type::I64,
                "usize" => ast::Type::USize,
                "isize" => ast::Type::ISize,
                "f16" => ast::Type::F16,
                "f32" => ast::Type::F32,
                "f64" => ast::Type::F64,
                _ => {
                    let mut full = ident;
                    self.punct2(':',':');
                    if self.punct('<') {
                        full += "<";
                        if let Some(ident) = self.ident() {
                            full += &ident;
                            if self.punct('>') {
                                full += ">";
                            }
                            else {
                                self.fatal(&format!("closing '>' expected instead of '{}'",ident));
                            }
                        }
                        else {
                            self.fatal(&format!("identifier expected instead of '{}'",full));
                        }
                    }
                    ast::Type::UnknownIdent(full)
                }
            }
        }

        else if let Some(types) = self.paren_types() {
            ast::Type::AnonTuple(types)
        }

        else if let Some(mut parser) = self.group('[') {
            let type_ = parser.type_();
            parser.punct(';');
            let expr = parser.expr();
            ast::Type::Array(Box::new(type_),Box::new(expr))
        }

        else {
            self.fatal(&format!("Type expected"));
        }
    }
}
