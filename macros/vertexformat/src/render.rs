use {
    crate::*,
};

pub struct VertexAttribute {
    pub location: usize,
    pub binding: usize,
    pub field_type: String,
    pub offset: usize,
}

pub(crate) fn render_struct(s: &Struct) -> String {

    let mut stride: usize = 0;
    let mut attributes: Vec<VertexAttribute> = Vec::new();
    let mut i = 0usize;
    for field in &s.fields {
        let (field_type,size) = if let Some(gen_param) = &field.gen_param {
            match field.ty.as_str() {
                "Vec2" => match gen_param.as_str() {
                    "u8" => ("U8XY",2),
                    "i8" => ("I8XY",2),
                    "u16" => ("U16XY",4),
                    "i16" => ("I16XY",4),
                    "u32" => ("U32XY",8),
                    "i32" => ("I32XY",8),
                    "u64" => ("U64XY",16),
                    "i64" => ("I64XY",16),
                    "f32" => ("F32XY",8),
                    "f64" => ("F64XY",16),
                    _ => { panic!("Vec2<{}> not supported",gen_param); },
                },
                "Vec3" => match gen_param.as_str() {
                    "u8" => ("U8XYZ",3),
                    "i8" => ("I8XYZ",3),
                    "u16" => ("U16XYZ",6),
                    "i16" => ("I16XYZ",6),
                    "u32" => ("U32XYZ",12),
                    "i32" => ("I32XYZ",12),
                    "u64" => ("U64XYZ",24),
                    "i64" => ("I64XYZ",24),
                    "f32" => ("F32XYZ",12),
                    "f64" => ("F64XYZ",24),
                    _ => { panic!("Vec3<{}> not supported",gen_param); },
                },
                "Vec4" => match gen_param.as_str() {
                    "u8" => ("U8XYZW",4),
                    "i8" => ("I8XYZW",4),
                    "u16" => ("U16XYZW",8),
                    "i16" => ("I16XYZW",8),
                    "u32" => ("U32XYZW",16),
                    "i32" => ("I32XYZW",16),
                    "u64" => ("U64XYZW",32),
                    "i64" => ("I64XYZW",32),
                    "f32" => ("F32XYZW",16),
                    "f64" => ("F64XYZW",32),
                    _ => { panic!("Vec4<{}> not supported",gen_param); },
                },
                "Color" => match gen_param.as_str() {
                    "u8" => ("U8RGBA",4),
                    "u16" => ("U16RGBA",8),
                    "f32" => ("F32RGBA",16),
                    "f64" => ("F64RGBA",32),
                    _ => { panic!("Color<{}> not supported",gen_param); },
                }
                _ => { panic!("{} not supported",field.ty); },
            }
        }
        else {
            match field.ty.as_str() {
                "u8" => ("U8",1),
                "i8" => ("I8",1),
                "u16" => ("U16",2),
                "i16" => ("I16",2),
                "u32" => ("U32",4),
                "i32" => ("I32",4),
                "u64" => ("U64",8),
                "i64" => ("I64",8),
                "f32" => ("F32",4),
                "f64" => ("F64",8),
                "u8xy" => ("U8XY",2),
                "i8xy" => ("I8XY",2),
                "u16xy" => ("U16XY",4),
                "i16xy" => ("I16XY",4),
                "u32xy" => ("U32XY",8),
                "i32xy" => ("I32XY",8),
                "u64xy" => ("U64XY",16),
                "i64xy" => ("I64XY",16),
                "f32xy" => ("F32XY",8),
                "f64xy" => ("F64XY",16),
                "u8xyz" => ("U8XYZ",3),
                "i8xyz" => ("I8XYZ",3),
                "u16xyz" => ("U16XYZ",6),
                "i16xyz" => ("I16XYZ",6),
                "u32xyz" => ("U32XYZ",12),
                "i32xyz" => ("I32XYZ",12),
                "u64xyz" => ("U64XYZ",24),
                "i64xyz" => ("I64XYZ",24),
                "f32xyz" => ("F32XYZ",12),
                "f64xyz" => ("F64XYZ",24),
                "u8xyzw" => ("U8XYZW",4),
                "i8xyzw" => ("I8XYZW",4),
                "u16xyzw" => ("U16XYZW",8),
                "i16xyzw" => ("I16XYZW",8),
                "u32xyzw" => ("U32XYZW",16),
                "i32xyzw" => ("I32XYZW",16),
                "u64xyzw" => ("U64XYZW",32),
                "i64xyzw" => ("I64XYZW",32),
                "f32xyzw" => ("F32XYZW",16),
                "f64xyzw" => ("F64XYZW",32),
                "u8rgba" => ("U8NRGBA",4),
                "u16rgba" => ("U16NRGBA",8),
                "f32rgba" => ("F32RGBA",16),
                "f64rgba" => ("F64RGBA",32),
                _ => { panic!("{} not supported",field.ty); },
            }
        };
        attributes.push(VertexAttribute {
            location: i,
            binding: 0,
            field_type: field_type.to_string(),
            offset: stride,
        });
        stride += size;
        i += 1;
    }
    let mut r = format!("impl VertexFormat for {} {{ ",s.ident);
    r += &format!("fn stride() -> usize {{ {} }} ",stride);
    r += &format!("fn attributes() -> usize {{ {} }} ",attributes.len());
    r += "fn attribute(index: usize) -> VertexAttribute { ";
    r += "match index { ";
    for i in 0..attributes.len() {
        r += &format!("{} => VertexAttribute {{ location: {},binding: {},field_type: FieldType::{},offset: {}, }}, ",
            i,
            attributes[i].location,
            attributes[i].binding,
            attributes[i].field_type,
            attributes[i].offset,
        );
    }
    r += "_ => panic!(\"attribute index out of range\"), ";
    r += "} } }";
    r
}
