use crate::*;

/*
fn gl_base_type_name(ty: &sr::BaseType) -> &'static str {
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

fn collect_layouts(symbol: &str,ty: &sr::Type,vertex_symbol: &str,vertex_fields: &Vec<(String,sr::BaseType)>) -> Vec<(String,String)> {
    match ty {
        sr::Type::Array(_,_) => panic!("unable to handle array parameters just yet"),
        sr::Type::Base(base_type) => vec![(gl_base_type_name(&base_type).to_string(),symbol.to_string())],
        sr::Type::Inferred => panic!("unable to handle inferred parameters just yet"),
        sr::Type::Symbol(type_symbol) => if type_symbol == vertex_symbol {
            let mut layouts: Vec<(String,String)> = Vec::new();
            for (symbol,ty) in vertex_fields {
                layouts.push((gl_base_type_name(ty).to_string(),symbol.to_string()));
            }
            layouts
        }
        else {
            panic!("unable to handle symbol parameter other than the vertex just yet");
        },
        sr::Type::Tuple(types) => {
            let mut layouts: Vec<(String,String)> = Vec::new();
            let mut i = 0usize;
            for ty in types {
                layouts.extend_from_slice(&collect_layouts(&format!("{}{}",symbol,i),ty,vertex_symbol,vertex_fields));
                i += 1;
            }
            layouts
        },
    }
}

pub fn compile_vertex_shader(items: Vec<sr::Item>,vertex_symbol: String,vertex_fields: Vec<(String,sr::BaseType)>) -> Option<Vec<u8>> {
    println!("VERTEX SHADER:\ninput rust functions:\n");
    for item in &items {
        println!("{}",item);
    }

    // find main function
    let mut main: Option<(&Vec<(sr::Pat,Box<sr::Type>)>,&Option<Box<sr::Type>>,&Vec<sr::Stat>)> = None;
    for item in &items { 
        if let sr::Item::Function(symbol,params,return_ty,stats) = item {
            if symbol == "main" {
                main = Some((params,return_ty,stats));
                break;
            }
        }
    }
    let (main_params,main_return_ty,_main_stats) = main.expect("main function missing in vertex shader");

    // collect input layouts
    let mut input_layouts: Vec<(String,String)> = Vec::new();
    for (pat,ty) in main_params {
        if let sr::Pat::Symbol(symbol) = pat {
            input_layouts.extend_from_slice(&collect_layouts(symbol,ty,&vertex_symbol,&vertex_fields));
        }
        else {
            panic!("only symbol allowed as parameter pattern (not {})",pat);
        }
    }

    // collect output layouts
    let main_return_ty = if let Some(main_return_ty) = &main_return_ty { main_return_ty } else { panic!("main should have a return type") };
    let output_layouts = collect_layouts("output",main_return_ty,&vertex_symbol,&vertex_fields);
    if let output_layouts[0].1 != "vec4" {
        panic!("output should be a vec4 for the position, optionally followed by other parameters");
    }

    // render start
    let mut r = "#version 450\n".to_string();

    // render input locations
    for i in 0..input_layouts.len() {
        r += &format!("layout(location={}) in {} {};\n",i,input_layouts[i].0,input_layouts[i].1);
    }

    // render output locations (0 is gl_Position)
    for i in 1..output_layouts.len() {
        r += &format!("layout(location={}) out {} {};\n",i,output_layouts[i].0,output_layouts[0].1);
    }

    // render main code
    r += "void main() {\n";

    // TODO: the actual code
    r += &format!("    gl_Position = vec4({}.x,{}.y,0.0,1.0);\n",vertex_fields[0].0,vertex_fields[0].0);
    r += &format!("    todo = {};\n",vertex_fields[1].0);

    r += "}\n";

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
*/

pub fn compile_vertex_shader(module: sr::Module,vertex_ident: String,vertex_fields: Vec<(String,sr::BaseType)>) -> Option<Vec<u8>> {
    println!("COMPILE VERTEX SHADER");

    // add external vertex definition to structs
    let mut fields: Vec<(String,sr::Type)> = Vec::new();
    for (ident,bt) in vertex_fields {
        fields.push((ident.clone(),sr::Type::Base(bt.clone())));
    }
    module.structs.insert(vertex_ident,fields);

    // 
    let module = resolve_idents(&module,Some(vertex_ident.clone()));
    let module = resolve_anon_tuples(&module);
    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (ident,(ty,expr)) in module.consts {
            println!("    {}: {} = {}",ident,ty,expr);
        }
    }
    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,fields) in module.structs {
            println!("    {} {{",ident);
            for (ident,ty) in fields {
                println!("        {}: {},",ident,ty);
            }
            println!("    }}");
        }
    }
    if module.anon_tuple_structs.len() > 0 {
        println!("Anonymous Tuple Structs:");
        for (ident,fields) in module.anon_tuple_structs {
            println!("    {} {{",ident);
            for (ident,ty) in fields {
                println!("        {}: {},",ident,ty);
            }
            println!("    }}");
        }
    }
    println!("Vertex struct:");
    println!("    {} {{",vertex_ident);
    for (ident,ty) in vertex_fields {
        println!("        {}: {}",ident,ty.to_rust());
    }
    println!("    }}");
    if module.enums.len() > 0 {
        println!("Enums:");
        for (ident,variants) in module.enums {
            println!("    {} {{",ident);
            for variant in variants {
                println!("        {},",variant);
            }
            println!("    }}");
        }
    }
    if module.functions.len() > 0 {
        println!("Functions:");
        for (ident,(params,return_type,block)) in module.functions {
            print!("    fn {}(",ident);
            for (ident,ty) in params {
                print!("{}: {},",ident,ty);
            }
            print!(")");
            if let sr::Type::Void = return_type { } else {
                print!(" -> {}",return_type);
            }
            println!("{}",block);
        }
    }
    println!("TODO: replace enums");
    println!("TODO: roll out patterns");
    println!("TODO: render GLSL");
    None
}

pub fn compile_fragment_shader(module: sr::Module) -> Option<Vec<u8>> {
    println!("COMPILE FRAGMENT SHADER");
    let module = resolve_idents(&module,None);
    let module = resolve_anon_tuples(&module);
    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (ident,(ty,expr)) in module.consts {
            println!("    {}: {} = {}",ident,ty,expr);
        }
    }
    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,fields) in module.structs {
            println!("    {} {{",ident);
            for (ident,ty) in fields {
                println!("        {}: {},",ident,ty);
            }
            println!("    }}");
        }
    }
    if module.anon_tuple_structs.len() > 0 {
        println!("Anonymous Tuple Structs:");
        for (ident,fields) in module.anon_tuple_structs {
            println!("    {} {{",ident);
            for (ident,ty) in fields {
                println!("        {}: {},",ident,ty);
            }
            println!("    }}");
        }
    }
    if module.enums.len() > 0 {
        println!("Enums:");
        for (ident,variants) in module.enums {
            println!("    {} {{",ident);
            for variant in variants {
                println!("        {},",variant);
            }
            println!("    }}");
        }
    }
    if module.functions.len() > 0 {
        println!("Functions:");
        for (ident,(params,return_type,block)) in module.functions {
            print!("    fn {}(",ident);
            for (ident,ty) in params {
                print!("{}: {},",ident,ty);
            }
            print!(")");
            if let sr::Type::Void = return_type { } else {
                print!(" -> {}",return_type);
            }
            println!("{}",block);
        }
    }

    println!("TODO: replace enums");
    println!("TODO: roll out patterns");
    println!("TODO: render GLSL");
    None
}
