use {
    crate::*,
    std::rc::Rc,
};

pub fn process_vertex_shader(module: &mut sr::Module,vertex_ident: String,vertex_fields: Vec<sr::Field>) -> Option<Vec<u8>> {

    println!("PROCESS VERTEX SHADER");

    // add external vertex definition to structs
    let mut fields: Vec<sr::Field> = Vec::new();
    for field in vertex_fields.iter() {
        fields.push(
            sr::Field {
                ident: field.ident.clone(),
                type_: if let sr::Type::Base(bt) = &field.type_ {
                    sr::Type::Base(bt.clone())
                }
                else {
                    panic!("vertex struct field should be of base type (instead of {})",field.type_);
                },
            }
        );
    }
    module.structs.insert(
        vertex_ident.clone(),
        Rc::new(
            sr::Struct {
                ident: vertex_ident,
                fields,
            }
        )
    );

    resolve_unknowns(module);
    resolve_anon_tuples(module);

    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (_,const_) in module.consts.iter() {
            println!("    {}: {} = {}",const_.ident,const_.type_,const_.value.as_ref().unwrap());
        }
    }

    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,struct_) in module.structs.iter() {
            println!("    {} {{",ident);
            for field in struct_.fields.iter() {
                println!("        {}: {},",field.ident,field.type_);
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

    None
}

pub fn process_fragment_shader(module: &mut sr::Module) -> Option<Vec<u8>> {

    println!("COMPILE FRAGMENT SHADER");

    resolve_unknowns(module);
    resolve_anon_tuples(module);
    
    println!("Module: {}",module.ident);
    if module.consts.len() > 0 {
        println!("Constants:");
        for (_,const_) in module.consts.iter() {
            println!("    {}: {} = {}",const_.ident,const_.type_,const_.value.as_ref().unwrap());
        }
    }

    if module.structs.len() > 0 {
        println!("Structs:");
        for (ident,struct_) in module.structs.iter() {
            println!("    {} {{",ident);
            for field in struct_.fields.iter() {
                println!("        {}: {},",field.ident,field.type_);
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

    None
}
