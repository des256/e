use {
    crate::*,
};

impl Parser {

    pub(crate) fn parse_type(&mut self) -> Type {

        // identifier (base type or anything named)
        if let Some(ident) = self.ident() {
            match ident.as_str() {
                "bool" => Type::Base(BaseType::Bool),
                "u8" => Type::Base(BaseType::U8),
                "u16" => Type::Base(BaseType::U16),
                "u32" => Type::Base(BaseType::U32),
                "u64" => Type::Base(BaseType::U64),
                "i8" => Type::Base(BaseType::I8),
                "i16" => Type::Base(BaseType::I16),
                "i32" => Type::Base(BaseType::I32),
                "i64" => Type::Base(BaseType::I64),
                "f16" => Type::Base(BaseType::F16),
                "f32" => Type::Base(BaseType::F32),
                "f64" => Type::Base(BaseType::F64),
                "Vec2" => {
                    if !self.punct('<') {
                        panic!("< expected after Vec2");
                    }
                    if let Some(ident) = self.ident() {
                        let result = match ident.as_str() {
                            "bool" => Type::Base(BaseType::Vec2Bool),
                            "u8" => Type::Base(BaseType::Vec2U8),
                            "u16" => Type::Base(BaseType::Vec2U16),
                            "u32" => Type::Base(BaseType::Vec2U32),
                            "u64" => Type::Base(BaseType::Vec2U64),
                            "i8" => Type::Base(BaseType::Vec2I8),
                            "i16" => Type::Base(BaseType::Vec2I16),
                            "i32" => Type::Base(BaseType::Vec2I32),
                            "i64" => Type::Base(BaseType::Vec2I64),
                            "f16" => Type::Base(BaseType::Vec2F16),
                            "f32" => Type::Base(BaseType::Vec2F32),
                            "f64" => Type::Base(BaseType::Vec2F64),
                            _ => panic!("Vec2<> can not be made from {}",ident),
                        };
                        if !self.punct('>') {
                            panic!("> expected");
                        }
                        result
                    }
                    else {
                        panic!("Vec2 can only be made from basic types");
                    }
                },
                "Vec3" => {
                    if !self.punct('<') {
                        panic!("< expected after Vec3");
                    }
                    if let Some(ident) = self.ident() {
                        let result = match ident.as_str() {
                            "bool" => Type::Base(BaseType::Vec3Bool),
                            "u8" => Type::Base(BaseType::Vec3U8),
                            "u16" => Type::Base(BaseType::Vec3U16),
                            "u32" => Type::Base(BaseType::Vec3U32),
                            "u64" => Type::Base(BaseType::Vec3U64),
                            "i8" => Type::Base(BaseType::Vec3I8),
                            "i16" => Type::Base(BaseType::Vec3I16),
                            "i32" => Type::Base(BaseType::Vec3I32),
                            "i64" => Type::Base(BaseType::Vec3I64),
                            "f16" => Type::Base(BaseType::Vec3F16),
                            "f32" => Type::Base(BaseType::Vec3F32),
                            "f64" => Type::Base(BaseType::Vec3F64),
                            _ => panic!("Vec3<> can not be made from {}",ident),
                        };
                        if !self.punct('>') {
                            panic!("> expected");
                        }
                        result
                    }
                    else {
                        panic!("Vec2 can only be made from basic types");
                    }
                },
                "Vec4" => {
                    if !self.punct('<') {
                        panic!("< expected after Vec4");
                    }
                    if let Some(ident) = self.ident() {
                        let result = match ident.as_str() {
                            "bool" => Type::Base(BaseType::Vec4Bool),
                            "u8" => Type::Base(BaseType::Vec4U8),
                            "u16" => Type::Base(BaseType::Vec4U16),
                            "u32" => Type::Base(BaseType::Vec4U32),
                            "u64" => Type::Base(BaseType::Vec4U64),
                            "i8" => Type::Base(BaseType::Vec4I8),
                            "i16" => Type::Base(BaseType::Vec4I16),
                            "i32" => Type::Base(BaseType::Vec4I32),
                            "i64" => Type::Base(BaseType::Vec4I64),
                            "f16" => Type::Base(BaseType::Vec4F16),
                            "f32" => Type::Base(BaseType::Vec4F32),
                            "f64" => Type::Base(BaseType::Vec4F64),
                            _ => panic!("Vec4<> can not be made from {}",ident),
                        };
                        if !self.punct('>') {
                            panic!("> expected");
                        }
                        result
                    }
                    else {
                        panic!("Vec4 can only be made from basic types");
                    }
                },
                "Color" => {
                    if !self.punct('<') {
                        panic!("< expected after Color");
                    }
                    if let Some(ident) = self.ident() {
                        let result = match ident.as_str() {
                            "u8" => Type::Base(BaseType::ColorU8),
                            "u16" => Type::Base(BaseType::ColorU16),
                            "f16" => Type::Base(BaseType::ColorF16),
                            "f32" => Type::Base(BaseType::ColorF32),
                            "f64" => Type::Base(BaseType::ColorF64),
                            _ => panic!("Color<> can not be made from {}",ident),
                        };
                        if !self.punct('>') {
                            panic!("> expected");
                        }
                        result
                    }
                    else {
                        panic!("Color can only be made from basic types");
                    }
                },
                _ => {
                    Type::Ident(ident)
                },
            }
        }

        // array type
        else if let Some(mut parser) = self.group('[') {
            let element_type = parser.parse_type();
            parser.punct(';');
            let expr = parser.parse_expr();
            Type::Array(Box::new(element_type),Box::new(expr))
        }

        // anonymous tuple type
        else if let Some(types) = self.parse_paren_types() {
            Type::AnonTuple(types)
        }

        else {
            panic!("type expected");
        }
    }
}
