use {
    super::*,
    std::collections::HashMap,
};

impl Resolver {

    // TODO:
    // - anywhere where tuple reference shows up, turn it into a struct reference
    // - anywhere where an alias reference shows up, convert it into the ultimate type being referred

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

        // TODO: resolve enums into enum_structs

        // resolve function types and expressions
        let mut functions: HashMap<String,Function> = HashMap::new();
        for (function_ident,function) in module.functions.iter() {

            self.push_context(format!("function {}",function_ident));

            let return_type = self.resolve_type(&function.return_type);
            let mut params: Vec<(String,Type)> = Vec::new();
            for (ident,type_) in function.params.iter() {
                let new_type = self.resolve_type(type_);
                params.push((ident.clone(),new_type));
            }
            let block = self.resolve_should_block(&function.block,&return_type);
            functions.insert(function_ident.clone(),Function {
                ident: function_ident.clone(),
                params,
                return_type,
                block,
            });

            self.pop_context();
        }

        Module {
            ident: self.module.ident.clone(),
            tuples,
            structs,
            tuple_structs,
            anon_tuple_structs: module.anon_tuple_structs,
            extern_structs: module.extern_structs,
            enums: module.enums,
            aliases: module.aliases,
            consts: module.consts,
            functions,
        }
    }
}
