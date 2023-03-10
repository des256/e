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
                        match (ident.as_str(),type_.clone()) {
                            ("Vec2",Type::Bool) => Type::Vec2Bool,
                            ("Vec2",Type::U8) => Type::Vec2U8,
                            ("Vec2",Type::I8) => Type::Vec2I8,
                            ("Vec2",Type::U16) => Type::Vec2U16,
                            ("Vec2",Type::I16) => Type::Vec2I16,
                            ("Vec2",Type::U32) => Type::Vec2U32,
                            ("Vec2",Type::I32) => Type::Vec2I32,
                            ("Vec2",Type::U64) => Type::Vec2U64,
                            ("Vec2",Type::I64) => Type::Vec2I64,
                            ("Vec2",Type::F16) => Type::Vec2F16,
                            ("Vec2",Type::F32) => Type::Vec2F32,
                            ("Vec2",Type::F64) => Type::Vec2F64,
                            ("Vec3",Type::Bool) => Type::Vec3Bool,
                            ("Vec3",Type::U8) => Type::Vec3U8,
                            ("Vec3",Type::I8) => Type::Vec3I8,
                            ("Vec3",Type::U16) => Type::Vec3U16,
                            ("Vec3",Type::I16) => Type::Vec3I16,
                            ("Vec3",Type::U32) => Type::Vec3U32,
                            ("Vec3",Type::I32) => Type::Vec3I32,
                            ("Vec3",Type::U64) => Type::Vec3U64,
                            ("Vec3",Type::I64) => Type::Vec3I64,
                            ("Vec3",Type::F16) => Type::Vec3F16,
                            ("Vec3",Type::F32) => Type::Vec3F32,
                            ("Vec3",Type::F64) => Type::Vec3F64,
                            ("Vec4",Type::Bool) => Type::Vec4Bool,
                            ("Vec4",Type::U8) => Type::Vec4U8,
                            ("Vec4",Type::I8) => Type::Vec4I8,
                            ("Vec4",Type::U16) => Type::Vec4U16,
                            ("Vec4",Type::I16) => Type::Vec4I16,
                            ("Vec4",Type::U32) => Type::Vec4U32,
                            ("Vec4",Type::I32) => Type::Vec4I32,
                            ("Vec4",Type::U64) => Type::Vec4U64,
                            ("Vec4",Type::I64) => Type::Vec4I64,
                            ("Vec4",Type::F16) => Type::Vec4F16,
                            ("Vec4",Type::F32) => Type::Vec4F32,
                            ("Vec4",Type::F64) => Type::Vec4F64,
                            ("Mat2x2",Type::F32) => Type::Mat2x2F32,
                            ("Mat2x2",Type::F64) => Type::Mat2x2F64,
                            ("Mat2x3",Type::F32) => Type::Mat2x3F32,
                            ("Mat2x3",Type::F64) => Type::Mat2x3F64,
                            ("Mat2x4",Type::F32) => Type::Mat2x4F32,
                            ("Mat2x4",Type::F64) => Type::Mat2x4F64,
                            ("Mat3x2",Type::F32) => Type::Mat3x2F32,
                            ("Mat3x2",Type::F64) => Type::Mat3x2F64,
                            ("Mat3x3",Type::F32) => Type::Mat3x3F32,
                            ("Mat3x3",Type::F64) => Type::Mat3x3F64,
                            ("Mat3x4",Type::F32) => Type::Mat3x4F32,
                            ("Mat3x4",Type::F64) => Type::Mat3x4F64,
                            ("Mat4x2",Type::F32) => Type::Mat4x2F32,
                            ("Mat4x2",Type::F64) => Type::Mat4x2F64,
                            ("Mat4x3",Type::F32) => Type::Mat4x3F32,
                            ("Mat4x3",Type::F64) => Type::Mat4x3F64,
                            ("Mat4x4",Type::F32) => Type::Mat4x4F32,
                            ("Mat4x4",Type::F64) => Type::Mat4x4F64,
                            _ => self.fatal(&format!("Unknown type {}<{}>",ident,type_)),
                        }
                    }
                    else {
                        Type::Ident(ident)
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
