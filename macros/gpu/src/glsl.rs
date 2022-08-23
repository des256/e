use crate::*;

impl BaseType {
    fn gl_name(&self) -> &'static str {
        match self {
            BaseType::U8 => "uint",
            BaseType::U16 => "uint",
            BaseType::U32 => "uint",
            BaseType::U64 => "uint",
            BaseType::I8 => "int",
            BaseType::I16 => "int",
            BaseType::I32 => "int",
            BaseType::I64 => "int",
            BaseType::F16 => "float",
            BaseType::F32 => "float",
            BaseType::F64 => "double",
            BaseType::Vec2U8 => "uvec2",
            BaseType::Vec2U16 => "uvec2",
            BaseType::Vec2U32 => "uvec2",
            BaseType::Vec2U64 => "uvec2",
            BaseType::Vec2I8 => "ivec2",
            BaseType::Vec2I16 => "ivec2",
            BaseType::Vec2I32 => "ivec2",
            BaseType::Vec2I64 => "ivec2",
            BaseType::Vec2F16 => "vec2",
            BaseType::Vec2F32 => "vec2",
            BaseType::Vec2F64 => "vec2",
            BaseType::Vec3U8 => "uvec3",
            BaseType::Vec3U16 => "uvec3",
            BaseType::Vec3U32 => "uvec3",
            BaseType::Vec3U64 => "uvec3",
            BaseType::Vec3I8 => "ivec3",
            BaseType::Vec3I16 => "ivec3",
            BaseType::Vec3I32 => "ivec3",
            BaseType::Vec3I64 => "ivec3",
            BaseType::Vec3F16 => "vec3",
            BaseType::Vec3F32 => "vec3",
            BaseType::Vec3F64 => "vec3",
            BaseType::Vec4U8 => "uvec4",
            BaseType::Vec4U16 => "uvec4",
            BaseType::Vec4U32 => "uvec4",
            BaseType::Vec4U64 => "uvec4",
            BaseType::Vec4I8 => "ivec4",
            BaseType::Vec4I16 => "ivec4",
            BaseType::Vec4I32 => "ivec4",
            BaseType::Vec4I64 => "ivec4",
            BaseType::Vec4F16 => "vec4",
            BaseType::Vec4F32 => "vec4",
            BaseType::Vec4F64 => "vec4",
            BaseType::ColorU8 => "vec4",
            BaseType::ColorU16 => "vec4",
            BaseType::ColorF16 => "vec4",
            BaseType::ColorF32 => "vec4",
            BaseType::ColorF64 => "vec4",
        }    
    }
}

pub fn compile_vertex_shader(item: sr::Item,vertex: String) -> String {

    // get the main function
    let items = if let sr::Item::Module(_,items) = item {
        items
    }
    else {
        panic!("vertex shader should be a module");
    };
    if items.len() > 1 {
        panic!("for now, vertex shader can only contain one main() function");
    }
    let (params,return_ty,stats) = if let sr::Item::Function(symbol,params,return_ty,stats) = &items[0] {
        if symbol != "main" {
            panic!("for now, vertex shader can only contain one main() function");
        }
        if let Some(return_ty) = return_ty {
            (params,return_ty,stats)
        }
        else {
            panic!("vertex shader main() function should return Vec4<f32> as position, as well as varyings");
        }
    }
    else {
        panic!("for now, vertex shader can only contain one main() function");
    };

    // start the shader
    let mut r = "#version 450\n".to_string();

    // make ins from main parameters
    let mut loc = 0usize;
    for (symbol,ty) in params {
        match &**ty {
            sr::Type::Symbol(type_name) => {
                if let Some(base_type) = BaseType::from_rust(&type_name) {
                    r += &format!("layout(location = {}) in {} {};\n",loc,base_type.gl_name(),symbol);
                    loc += 1;
                }
                else if *type_name == vertex {
                    // somehow magically figure out the fields of vertex...
                    r += &format!("layout(location = {}) in {} {};  // TODO\n",loc,"?",vertex);
                    loc += 1;
                }
            },
            _ => {
                panic!("for now, only base type parameters allowed in function");
            },
        }
    }

    // TODO: make outs from the return type

    // add function
    r += "void main() {\n}\n";
    r
}

pub fn compile_fragment_shader(item: sr::Item) -> String {

    // get the main function
    let items = if let sr::Item::Module(_,items) = item {
        items
    }
    else {
        panic!("fragment shader should be a module");
    };
    if items.len() > 1 {
        panic!("for now, fragment shader can only contain one main() function");
    }
    let (params,return_ty,stats) = if let sr::Item::Function(symbol,params,return_ty,stats) = &items[0] {
        if symbol != "main" {
            panic!("for now, fragment shader can only contain one main() function");
        }
        if let Some(return_ty) = return_ty {
            (params,return_ty,stats)
        }
        else {
            panic!("for now, fragment shader main() function can only return the sample color as Color<f32>");
        }
    }
    else {
        panic!("for now, fragment shader can only contain one main() function");
    };

    // start the shader
    let mut r = "#version 450\n".to_string();

    // make ins from main parameters
    let mut loc = 0usize;
    for (symbol,ty) in params {
        match &**ty {
            sr::Type::Symbol(type_name) => {
                if let Some(base_type) = BaseType::from_rust(&type_name) {
                    r += &format!("layout(location = {}) in {} {};\n",loc,base_type.gl_name(),symbol);
                    loc += 1;
                }
                else {
                    panic!("for now, only base type parameters allowed in function");
                }
            },
            _ => {
                panic!("for now, only base type parameters allowed in function");
            },
        }
    }

    // make outs from return type
    if let sr::Type::Symbol(type_name) = &**return_ty {
        if type_name != "Color<f32>" {
            panic!("for now, fragment shader main() function can only return the sample color as Color<f32>");
        }
    }
    let mut loc = 0usize;
    r += &format!("layout(location = {}) out vec4 __sample__;\n",loc);
    loc += 1;

    // add function
    r += "void main() {\n}\n";
    
    r
}
