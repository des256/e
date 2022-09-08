use {
    crate::*,
    std::rc::Rc,
};

pub fn process_vertex_shader(module: sr::Module,vertex_ident: String,vertex_fields: Vec<sr::Field>) -> Option<Vec<u8>> {

    println!("PROCESS VERTEX SHADER");

    // add external vertex definition to structs
    let mut fields: Vec<sr::Field> = Vec::new();
    for field in vertex_fields.iter() {
        fields.push(
            sr::Field {
                ident: field.ident.clone(),
                type_: if let sr::Type::Base(bt) = field.type_ {
                    sr::Type::Base(bt.clone())
                }
                else {
                    panic!("vertex struct field should be of base type (instead of {})",field.type_);
                },
            }
        );
    }
    module.structs.insert(
        vertex_ident,
        Rc::new(
            sr::Struct {
                ident: vertex_ident,
                fields,
            }
        )
    );

    let module = resolve_idents(&module,Some(vertex_ident.clone()));
    let module = resolve_anon_tuples(&module);
    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (_,const_) in module.consts {
            println!("    {}: {} = {}",const_.ident,const_.type_,const_.value.as_ref().unwrap());
        }
    }
    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,struct_) in module.structs {
            println!("    {} {{",ident);
            for field in struct_.fields.iter() {
                println!("        {}: {},",field.ident,field.type_);
            }
            println!("    }}");
        }
    }
    println!("Vertex struct:");
    println!("    {} {{",vertex_ident);
    for field in vertex_fields.iter() {
        println!("        {}: {}",field.ident,field.type_);
    }
    println!("    }}");
    if module.enums.len() > 0 {
        println!("Enums:");
        for (_,enum_) in module.enums.iter() {
            println!("    {} {{",enum_.ident);
            for variant in enum_.variants.iter() {
                println!("        {},",variant);
            }
            println!("    }}");
        }
    }
    if module.functions.len() > 0 {
        println!("Functions:");
        for (_,function) in module.functions.iter() {
            print!("    fn {}(",function.ident);
            for param in function.params.iter() {
                print!("{}: {},",param.ident,param.type_);
            }
            print!(")");
            if let sr::Type::Void = function.return_type { } else {
                print!(" -> {}",function.return_type);
            }
            println!("{}",function.block);
        }
    }

    println!("TODO: replace enums");
    println!("TODO: roll out patterns");

    None
}

pub fn process_fragment_shader(module: sr::Module) -> Option<Vec<u8>> {

    println!("COMPILE FRAGMENT SHADER");

    let module = resolve_idents(&module,None);
    let module = resolve_anon_tuples(&module);
    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (_,const_) in module.consts {
            println!("    {}: {} = {}",const_.ident,const_.type_,const_.value.as_ref().unwrap());
        }
    }
    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,struct_) in module.structs {
            println!("    {} {{",ident);
            for field in struct_.fields.iter() {
                println!("        {}: {},",field.ident,field.type_);
            }
            println!("    }}");
        }
    }
    if module.enums.len() > 0 {
        println!("Enums:");
        for (_,enum_) in module.enums.iter() {
            println!("    {} {{",enum_.ident);
            for variant in enum_.variants.iter() {
                println!("        {},",variant);
            }
            println!("    }}");
        }
    }
    if module.functions.len() > 0 {
        println!("Functions:");
        for (_,function) in module.functions.iter() {
            print!("    fn {}(",function.ident);
            for param in function.params.iter() {
                print!("{}: {},",param.ident,param.type_);
            }
            print!(")");
            if let sr::Type::Void = function.return_type { } else {
                print!(" -> {}",function.return_type);
            }
            println!("{}",function.block);
        }
    }

    println!("TODO: replace enums");
    println!("TODO: roll out patterns");

    None
}
