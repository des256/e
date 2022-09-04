use crate::*;

const SPIRV_MAGIC_NUMBER: u32 = 0x07230203;
const SPIRV_VERSION_13: u32 = 0x00010300;
const SPIRV_GENERATOR_MAGIC_NUMBER: u32 = 0xB5009BB1;

/*
let r: Vec<u32> = vec![SPIRV_MAGIC_NUMBER,SPIRV_VERSION_15,SPIRV_GENERATOR_MAGIC_NUMBER,0];
// length+opcode (typeid) (resultid) (op1) ... (opN)
// 1. opcapability
// 2. opextension
// 3. opextinstimport
// 4. opmemorymodel
// 5. opentrypoint
// 6. all execution-mode declarations opexecutionmode/opexecutionmodeid
// 7. opstring/opsourceextension/opsource/opsourcecontinued, opname, opmembername, opmoduleprocessed
// 8. all annotated instructions
// 9. all type declarations. constant declarations, global variable declarations, opundef, opline/opnoline, opextinst
// 10. declarations: opfunction, opfunctionparameter, opfunctionend
// 11. definitions: opfunction, opfunctionparameter, block, ..., block, opfunctionend
*/

pub fn compile_vertex_shader(module: sr::Module,vertex_ident: String,vertex_fields: Vec<(String,sr::BaseType)>) -> Option<Vec<u8>> {
    println!("COMPILE VERTEX SHADER");
    println!("Rust input module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (ident,(ty,expr)) in module.consts {
            println!("    {}: {} = {}",ident,ty,expr);
        }
    }
    if module.tuples.len() > 0 {
        println!("Tuples:");
        for (ident,types) in module.tuples {
            print!("    {}(",ident);
            for ty in types {
                print!("{},",ty);
            }
            println!(")");
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
    println!("TODO: replace tuples with structs");
    println!("TODO: replace enums");
    println!("TODO: roll out patterns");
    println!("TODO: render SPIR-V");
    None
}

pub fn compile_fragment_shader(module: sr::Module) -> Option<Vec<u8>> {
    println!("COMPILE FRAGMENT SHADER");
    println!("Rust input module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (ident,(ty,expr)) in module.consts {
            println!("    {}: {} = {}",ident,ty,expr);
        }
    }
    if module.tuples.len() > 0 {
        println!("Tuples:");
        for (ident,types) in module.tuples {
            print!("    {}(",ident);
            for ty in types {
                print!("{},",ty);
            }
            println!(")");
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
    println!("TODO: replace tuples with structs");
    println!("TODO: replace enums");
    println!("TODO: roll out patterns");
    println!("TODO: render SPIR-V");
    None
}
