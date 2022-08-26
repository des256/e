use {
    crate::*,
    std::rc::Rc,
};

fn _gl_basetype_name(ty: sr::BaseType) -> &'static str {
    match ty {
        sr::BaseType::U8 => "uint",
        sr::BaseType::U16 => "uint",
        sr::BaseType::U32 => "uint",
        sr::BaseType::U64 => "uint",
        sr::BaseType::I8 => "int",
        sr::BaseType::I16 => "int",
        sr::BaseType::I32 => "int",
        sr::BaseType::I64 => "int",
        sr::BaseType::F16 => "float",
        sr::BaseType::F32 => "float",
        sr::BaseType::F64 => "double",
        sr::BaseType::Vec2U8 => "uvec2",
        sr::BaseType::Vec2U16 => "uvec2",
        sr::BaseType::Vec2U32 => "uvec2",
        sr::BaseType::Vec2U64 => "uvec2",
        sr::BaseType::Vec2I8 => "ivec2",
        sr::BaseType::Vec2I16 => "ivec2",
        sr::BaseType::Vec2I32 => "ivec2",
        sr::BaseType::Vec2I64 => "ivec2",
        sr::BaseType::Vec2F16 => "vec2",
        sr::BaseType::Vec2F32 => "vec2",
        sr::BaseType::Vec2F64 => "vec2",
        sr::BaseType::Vec3U8 => "uvec3",
        sr::BaseType::Vec3U16 => "uvec3",
        sr::BaseType::Vec3U32 => "uvec3",
        sr::BaseType::Vec3U64 => "uvec3",
        sr::BaseType::Vec3I8 => "ivec3",
        sr::BaseType::Vec3I16 => "ivec3",
        sr::BaseType::Vec3I32 => "ivec3",
        sr::BaseType::Vec3I64 => "ivec3",
        sr::BaseType::Vec3F16 => "vec3",
        sr::BaseType::Vec3F32 => "vec3",
        sr::BaseType::Vec3F64 => "vec3",
        sr::BaseType::Vec4U8 => "uvec4",
        sr::BaseType::Vec4U16 => "uvec4",
        sr::BaseType::Vec4U32 => "uvec4",
        sr::BaseType::Vec4U64 => "uvec4",
        sr::BaseType::Vec4I8 => "ivec4",
        sr::BaseType::Vec4I16 => "ivec4",
        sr::BaseType::Vec4I32 => "ivec4",
        sr::BaseType::Vec4I64 => "ivec4",
        sr::BaseType::Vec4F16 => "vec4",
        sr::BaseType::Vec4F32 => "vec4",
        sr::BaseType::Vec4F64 => "vec4",
        sr::BaseType::ColorU8 => "vec4",
        sr::BaseType::ColorU16 => "vec4",
        sr::BaseType::ColorF16 => "vec4",
        sr::BaseType::ColorF32 => "vec4",
        sr::BaseType::ColorF64 => "vec4",
    }    
}

pub fn compile_vertex_shader(items: Vec<sr::Item>,_vertex: &'static str,_vertex_types: Vec<sr::BaseType>) -> Option<String> {
    println!("compile_vertex_shader called with:");
    for item in items {
        println!("{}",item);
    }
    let r = "TODO".to_string();
    Some(r)
}

pub fn compile_fragment_shader(items: Vec<sr::Item>) -> Option<String> {
    println!("compile_fragment_shader called with:");
    for item in items {
        println!("{}",item);
    }
    let r = "TODO".to_string();
    Some(r)
}
