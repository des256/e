use {
    crate::*,
    std::collections::HashMap,
};

impl Parser {

    pub fn module(&mut self) -> Module {
        self.keyword("mod");
        let ident = self.ident().expect("identifier expected");
        let mut functions: HashMap<String,(Vec<(String,Type)>,Type,Block)> = HashMap::new();
        let mut structs: HashMap<String,Vec<(String,Type)>> = HashMap::new();
#[allow(unused_assignments)]
        let mut anon_tuple_structs: HashMap<String,Vec<(String,Type)>> = HashMap::new();
        let mut enum_structs: HashMap<String,Vec<(String,Type)>> = HashMap::new();
        let mut enums: HashMap<String,(Vec<Variant>,Vec<Vec<usize>>)> = HashMap::new();
        let mut consts: HashMap<String,(Type,Expr)> = HashMap::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {

                // function
                if parser.keyword("fn") {
                    let ident = parser.ident().expect("identifier expected");
                    let params = parser.paren_ident_types();
                    let return_type = if parser.punct2('-','>') {
                        parser.type_()
                    }
                    else {
                        Type::Void
                    };
                    let block = parser.block().expect("{ expected");
                    functions.insert(ident,(params,return_type,block));        
                }

                // struct or tuple
                else if parser.keyword("struct") {
                    let ident = parser.ident().expect("identifier expected");
                    if parser.peek_group('{') {
                        let ident_types = parser.brace_ident_types();
                        let mut fields: Vec<(String,Type)> = Vec::new();
                        for (ident,ty) in ident_types {
                            fields.push((ident.clone(),ty))
                        }
                        structs.insert(ident,fields);
                    }
                    else if parser.peek_group('(') {
                        let types = parser.paren_types().unwrap();
                        let mut fields: Vec<(String,Type)> = Vec::new();
                        let mut i = 0usize;
                        for ty in types {
                            fields.push((format!("_{}",i),ty));
                            i += 1;
                        }
                        structs.insert(ident,fields);
                    }
                    else {
                        panic!("{}","{ or ( expected");
                    }
                }

                // enum
                else if parser.keyword("enum") {
                    let ident = parser.ident().expect("identifier expected");
                    if let Some(mut parser) = parser.group('{') {
                        let mut variants: Vec<Variant> = Vec::new();
                        while !parser.done() {
                            let ident = parser.ident().expect("identifier expected");
                            if parser.peek_group('{') {
                                let fields = parser.brace_ident_types();
                                variants.push(Variant::Struct(ident,fields));
                            }
                            else if parser.peek_group('(') {
                                let types = parser.paren_types().unwrap();
                                variants.push(Variant::Tuple(ident,types));
                            }
                            else {
                                variants.push(Variant::Naked(ident));
                            }
                            parser.punct(',');
                        }

                        // build tight field type list for the resulting struct
                        let mut field_types: Vec<(Type,usize)> = Vec::new();
                        for variant in variants.iter() {
                            match variant {
                                Variant::Naked(_) => { },
                                Variant::Tuple(_,types) => {
                                    let mut variant_field_types: Vec<(Type,usize)> = Vec::new();
                                    for type_ in types.iter() {
                                        let mut found = false;
                                        for i in 0..variant_field_types.len() {
                                            if variant_field_types[i].0 == *type_ {
                                                variant_field_types[i].1 += 1;
                                                found = true;
                                            }
                                        }
                                        if !found {
                                            variant_field_types.push((type_.clone(),1));
                                        }
                                    }
                                    for (type_,count) in variant_field_types.iter() {
                                        let mut found = false;
                                        for i in 0..field_types.len() {
                                            if field_types[i].0 == *type_ {
                                                if field_types[i].1 < *count {
                                                    field_types[i].1 = *count;
                                                }
                                                found = true;
                                            }
                                        }
                                        if !found {
                                            field_types.push((type_.clone(),*count));
                                        }
                                    }
                                },
                                Variant::Struct(_,fields) => {
                                    let mut variant_field_types: Vec<(Type,usize)> = Vec::new();
                                    for (_,type_) in fields.iter() {
                                        let mut found = false;
                                        for i in 0..variant_field_types.len() {
                                            if variant_field_types[i].0 == *type_ {
                                                variant_field_types[i].1 += 1;
                                                found = true;
                                            }
                                        }
                                        if !found {
                                            variant_field_types.push((type_.clone(),1));
                                        }
                                    }
                                    for (type_,count) in variant_field_types.iter() {
                                        let mut found = false;
                                        for i in 0..field_types.len() {
                                            if field_types[i].0 == *type_ {
                                                if field_types[i].1 < *count {
                                                    field_types[i].1 = *count;
                                                }
                                                found = true;
                                            }
                                        }
                                        if !found {
                                            field_types.push((type_.clone(),*count));
                                        }
                                    }
                                },
                            }
                        }

                        // field_types now contains the types and how many of them are needed

                        // describe the actual fields of the final struct, excluding the ID
                        let mut fields: Vec<(String,Type)> = Vec::new();
                        let mut i = 0usize;
                        for (type_,count) in field_types.iter() {
                            for _ in 0..*count {
                                fields.push((format!("_{}",i),type_.clone()));
                                i += 1;
                            }
                        }

                        // for each variant, allocate the fields
                        let mut variant_field_indices: Vec<Vec<usize>> = Vec::new();
                        for id in 0..variants.len() {
                            match &variants[id] {
                                Variant::Naked(_) => variant_field_indices.push(Vec::new()),
                                Variant::Tuple(_,types) => {

                                    // keep track of which type is already allocated and at which index that was
                                    let mut allocated: Vec<(Type,usize)> = Vec::new();

                                    // build list of indices that refer to the fields in the final struct
                                    let mut indices: Vec<usize> = Vec::new();

                                    for type_ in types.iter() {
                                        // if this type was already allocated before, allocate the next one
                                        let mut found = false;
                                        for i in 0..allocated.len() {
                                            if allocated[i].0 == *type_ {
                                                allocated[i].1 += 1;
                                                indices.push(allocated[i].1);
                                                found = true;
                                            }
                                        }

                                        // otherwise find the type in the fields and start allocating there
                                        if !found {
                                            for i in 0..fields.len() {
                                                if fields[i].1 == *type_ {
                                                    allocated.push((type_.clone(),i));
                                                    indices.push(i);
                                                }
                                            }
                                        }
                                    }

                                    // and store the indices for this variant
                                    variant_field_indices.push(indices);
                                },

                                Variant::Struct(_,fields) => {

                                    // keep track of which type is already allocated and at which index that was
                                    let mut allocated: Vec<(Type,usize)> = Vec::new();

                                    // build list of indices that refer to the fields in the final struct
                                    let mut indices: Vec<usize> = Vec::new();

                                    for field in fields.iter() {
                                        // if this type was already allocated before, allocate the next one
                                        let mut found = false;
                                        for i in 0..allocated.len() {
                                            if allocated[i].0 == field.1 {
                                                allocated[i].1 += 1;
                                                indices.push(allocated[i].1);
                                                found = true;
                                            }
                                        }

                                        // otherwise find the type in the fields and start allocating there
                                        if !found {
                                            for i in 0..fields.len() {
                                                if fields[i].1 == field.1 {
                                                    allocated.push((field.1.clone(),i));
                                                    indices.push(i);
                                                }
                                            }
                                        }
                                    }

                                    variant_field_indices.push(indices);
                                },
                            }
                        }

                        // add the ID
                        let mut final_fields: Vec<(String,Type)> = Vec::new();
                        final_fields.push(("id".to_string(),Type::Base(sr::BaseType::U32)));
                        final_fields.append(&mut fields);

                        // and build the struct and enum objects
                        enum_structs.insert(ident.clone(),final_fields);
                        enums.insert(ident,(variants,variant_field_indices));
                    }
                    else {
                        panic!("{}","{ expected");
                    }
                }

                // const
                else if parser.keyword("const") {
                    let ident = parser.ident().expect("identifier expected");
                    parser.punct(':');
                    let ty = parser.type_();
                    parser.punct('=');
                    let expr = parser.expr();
                    consts.insert(ident,(ty,expr));
                }
            }

            anon_tuple_structs = parser.anon_tuple_structs;
        }
        else {
            panic!("{}","{ expected");
        }

        Module {
            ident,
            functions,
            structs,
            anon_tuple_structs,
            enum_structs,
            enums,
            consts,
        }
    }

    pub fn parse_struct(&mut self) -> (String,Vec<(String,Type)>) {
        if !self.keyword("struct") {
            panic!("struct expected");
        }
        let ident = self.ident().expect("identifier expected");
        let fields = self.brace_ident_types();
        (ident,fields)
    }
}
