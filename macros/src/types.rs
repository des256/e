use super::*;

impl Parser {

    pub(crate) fn type_(&mut self) -> Result<Type,String> {
        if let Some(ident) = self.ident() {
            match ident.as_str() {
                "bool" => Ok(Type::Bool),
                "u8" => Ok(Type::U8),
                "i8" => Ok(Type::I8),
                "u16" => Ok(Type::U16),
                "i16" => Ok(Type::I16),
                "u32" => Ok(Type::U32),
                "i32" => Ok(Type::I32),
                "u64" => Ok(Type::U64),
                "i64" => Ok(Type::I64),
                "f16" => Ok(Type::F16),
                "f32" => Ok(Type::F32),
                "f64" => Ok(Type::F64),
                _ => {
                    self.punct2(':',':');
                    if self.punct('<') {
                        let type_ = self.type_()?;
                        self.punct('>');
                        match (ident.as_str(),type_.clone()) {
                            ("Vec2",Type::Bool) => Ok(Type::Vec2Bool),
                            ("Vec2",Type::U8) => Ok(Type::Vec2U8),
                            ("Vec2",Type::I8) => Ok(Type::Vec2I8),
                            ("Vec2",Type::U16) => Ok(Type::Vec2U16),
                            ("Vec2",Type::I16) => Ok(Type::Vec2I16),
                            ("Vec2",Type::U32) => Ok(Type::Vec2U32),
                            ("Vec2",Type::I32) => Ok(Type::Vec2I32),
                            ("Vec2",Type::U64) => Ok(Type::Vec2U64),
                            ("Vec2",Type::I64) => Ok(Type::Vec2I64),
                            ("Vec2",Type::F16) => Ok(Type::Vec2F16),
                            ("Vec2",Type::F32) => Ok(Type::Vec2F32),
                            ("Vec2",Type::F64) => Ok(Type::Vec2F64),
                            ("Vec3",Type::Bool) => Ok(Type::Vec3Bool),
                            ("Vec3",Type::U8) => Ok(Type::Vec3U8),
                            ("Vec3",Type::I8) => Ok(Type::Vec3I8),
                            ("Vec3",Type::U16) => Ok(Type::Vec3U16),
                            ("Vec3",Type::I16) => Ok(Type::Vec3I16),
                            ("Vec3",Type::U32) => Ok(Type::Vec3U32),
                            ("Vec3",Type::I32) => Ok(Type::Vec3I32),
                            ("Vec3",Type::U64) => Ok(Type::Vec3U64),
                            ("Vec3",Type::I64) => Ok(Type::Vec3I64),
                            ("Vec3",Type::F16) => Ok(Type::Vec3F16),
                            ("Vec3",Type::F32) => Ok(Type::Vec3F32),
                            ("Vec3",Type::F64) => Ok(Type::Vec3F64),
                            ("Vec4",Type::Bool) => Ok(Type::Vec4Bool),
                            ("Vec4",Type::U8) => Ok(Type::Vec4U8),
                            ("Vec4",Type::I8) => Ok(Type::Vec4I8),
                            ("Vec4",Type::U16) => Ok(Type::Vec4U16),
                            ("Vec4",Type::I16) => Ok(Type::Vec4I16),
                            ("Vec4",Type::U32) => Ok(Type::Vec4U32),
                            ("Vec4",Type::I32) => Ok(Type::Vec4I32),
                            ("Vec4",Type::U64) => Ok(Type::Vec4U64),
                            ("Vec4",Type::I64) => Ok(Type::Vec4I64),
                            ("Vec4",Type::F16) => Ok(Type::Vec4F16),
                            ("Vec4",Type::F32) => Ok(Type::Vec4F32),
                            ("Vec4",Type::F64) => Ok(Type::Vec4F64),
                            ("Mat2x2",Type::F32) => Ok(Type::Mat2x2F32),
                            ("Mat2x2",Type::F64) => Ok(Type::Mat2x2F64),
                            ("Mat2x3",Type::F32) => Ok(Type::Mat2x3F32),
                            ("Mat2x3",Type::F64) => Ok(Type::Mat2x3F64),
                            ("Mat2x4",Type::F32) => Ok(Type::Mat2x4F32),
                            ("Mat2x4",Type::F64) => Ok(Type::Mat2x4F64),
                            ("Mat3x2",Type::F32) => Ok(Type::Mat3x2F32),
                            ("Mat3x2",Type::F64) => Ok(Type::Mat3x2F64),
                            ("Mat3x3",Type::F32) => Ok(Type::Mat3x3F32),
                            ("Mat3x3",Type::F64) => Ok(Type::Mat3x3F64),
                            ("Mat3x4",Type::F32) => Ok(Type::Mat3x4F32),
                            ("Mat3x4",Type::F64) => Ok(Type::Mat3x4F64),
                            ("Mat4x2",Type::F32) => Ok(Type::Mat4x2F32),
                            ("Mat4x2",Type::F64) => Ok(Type::Mat4x2F64),
                            ("Mat4x3",Type::F32) => Ok(Type::Mat4x3F32),
                            ("Mat4x3",Type::F64) => Ok(Type::Mat4x3F64),
                            ("Mat4x4",Type::F32) => Ok(Type::Mat4x4F32),
                            ("Mat4x4",Type::F64) => Ok(Type::Mat4x4F64),
                            _ => self.err(&format!("Unknown type {}<{}>",ident,type_)),
                        }
                    }
                    else {
                        Ok(Type::Ident(ident))
                    }
                }
            }
        }

        else if let Some(types) = self.paren_types()? {
            Ok(Type::AnonTuple(types))
        }

        else if let Some(mut parser) = self.group('[') {
            let type_ = parser.type_()?;
            parser.punct(';');
            let count = parser.integer_literal();
            if count.is_none() {
                return self.err("integer array count expected");
            }
            let count = count.unwrap();
            Ok(Type::Array(Box::new(type_),count as usize))
        }

        else {
            self.err(&format!("Type expected"))
        }
    }
}
