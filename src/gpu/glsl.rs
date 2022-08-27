fn _gl_basetype_name(ty: &sr::BaseType) -> &'static str {
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

pub fn compile_vertex_shader(items: Vec<sr::Item>,vertex: Vec<(String,sr::BaseType)>) -> Option<Vec<u8>> {
    println!("VERTEX SHADER:\ninput:");
    for item in items {
        println!("{}",item);
    }

    // begin rendering
    let mut r = "#version 450\n".to_string();

    // input locations
    for i in 0..vertex.len() {
        r += &format!("layout(location={}) in {} {};\n",i,_gl_basetype_name(&vertex[i].1),vertex[i].0);
    }

    // TODO: output locations
    r += "layout(location=0) out vec4 todo;\n";

    // main code
    r += "void main() {\n";

    // TODO: the actual code
    r += &format!("    gl_Position = vec4({}.x,{}.y,0.0,1.0);\n",vertex[0].0,vertex[0].0);
    r += &format!("    todo = {};\n",vertex[1].0);

    r += "}\n";

    // TODO: convert r to raw C string in Vec<u8>

    println!("output:\n{}",r);

    r += "\0";
    Some(r.into_bytes())
}

pub fn compile_fragment_shader(items: Vec<sr::Item>) -> Option<Vec<u8>> {
    println!("FRAGMENT SHADER:\ninput:");
    for item in items {
        println!("{}",item);
    }

    // begin rendering
    let mut r = "#version 450\n".to_string();

    // TODO: input locations
    r += "layout(location=0) in vec4 varying_;\n";

    // TODO: output locations
    r += "layout(location=0) out vec4 fragment;\n";

    // main code
    r += "void main() {\n";

    // TODO: the actual code
    r += "    fragment = varying_;\n";

    r += "}\n";

    println!("output:\n{}",r);

    // TODO: convert r to raw C string in Vec<u8>

    r += "\0";
    Some(r.into_bytes())
}
