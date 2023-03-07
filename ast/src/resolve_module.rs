use {
    super::*,
    std::collections::HashMap,
};

impl Resolver {

    pub fn resolve_module(&mut self) -> Module {

        let module = self.module.clone();

        // convert tuples into structs
        let tuples: HashMap<String,Tuple> = HashMap::new();
        let mut tuple_structs = module.tuple_structs;
        for (ident,tuple) in module.tuples.iter() {

            self.push_context(format!("tuple {}",ident));

            let mut fields: Vec<(String,Type)> = Vec::new();
            for i in 0..tuple.types.len() {
                let new_type = self.resolve_type(&tuple.types[i]);
                fields.push((format!("f{}",i),new_type));
            }
            tuple_structs.insert(ident.clone(),Struct {
                ident: ident.clone(),
                fields,
            });

            self.pop_context();
            self.log_change(format!("converted tuple {} to struct",ident));
        }

        // resolve types referred to by the struct fields
        let mut structs: HashMap<String,Struct> = HashMap::new();
        for (struct_ident,struct_) in module.structs.iter() {

            self.push_context(format!("struct {}",struct_ident));

            let mut fields: Vec<(String,Type)> = Vec::new();
            for (ident,type_) in struct_.fields.iter() {
                let new_type = self.resolve_type(type_);
                fields.push((ident.clone(),new_type));
            }
            structs.insert(struct_ident.clone(),Struct {
                ident: struct_ident.clone(),
                fields,
            });

            self.pop_context();
            // no need to log, we're only touching the field types
        }

        let anon_tuple_structs = module.anon_tuple_structs;
        let extern_structs = module.extern_structs;
        let enums = module.enums;
        let aliases = module.aliases;
        let consts = module.consts;
        let functions = module.functions;

        Module {
            ident: self.module.ident.clone(),
            tuples,
            structs,
            tuple_structs,
            anon_tuple_structs,
            extern_structs,
            enums,
            aliases,
            consts,
            functions,
        }
    }
}
