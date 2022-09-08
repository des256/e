use {
    crate::*,
    std::rc::Rc,
};

pub fn process_vertex_shader(mut module: sr::Module,vertex_ident: String,vertex_fields: Vec<sr::Field>) -> Option<Vec<u8>> {

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

    resolve_unknowns(&mut module);
    resolve_loose_types(&mut module);

    // for all expressions ask what their tightest type is
    // when this is still too loose, try to infer in other ways: function parameter types, function return values, let statements
    // once known, broadcast down accordingly
    // resolve anonymous tuples into Expr::Struct

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

    None
}

pub fn process_fragment_shader(mut module: sr::Module) -> Option<Vec<u8>> {

    println!("COMPILE FRAGMENT SHADER");

    resolve_unknowns(&mut module);
    resolve_loose_types(&mut module);
    
    // for all expressions ask what their tightest type is
    // when this is still too loose, try to infer in other ways: function parameter types, function return values, let statements
    // once known, broadcast down all types accordingly
    // resolve anonymous tuples

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

    None
}
