use super::*;

impl Parser {

    pub(crate) fn module(&mut self) -> Result<Module,String> {
        self.keyword("mod");
        let ident = self.ident();
        if ident.is_none() {
            return self.err("identifier expected");
        }
        let ident = ident.unwrap();
        let mut structs: HashMap<String,Struct> = HashMap::new();
        let mut tuple_structs: HashMap<String,Vec<Type>> = HashMap::new();
        let mut enums: HashMap<String,Enum> = HashMap::new();
        let mut aliases: HashMap<String,Alias> = HashMap::new();
        let mut consts: HashMap<String,Const> = HashMap::new();
        let mut functions: HashMap<String,Function> = HashMap::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {

                // function
                if parser.keyword("fn") {
                    let ident = parser.ident();
                    if ident.is_none() {
                        return self.err("identifier expected");
                    }
                    let ident = ident.unwrap();
                    let ident_types = parser.paren_ident_types()?;
                    let mut params: Vec<(String,Type)> = Vec::new();
                    for (ident,type_) in ident_types.iter() {
                        params.push((ident.clone(),type_.clone()));
                    }
                    let return_type = if parser.punct2('-','>') {
                        parser.type_()?
                    }
                    else {
                        Type::Void
                    };
                    let block = parser.block()?;
                    if block.is_none() {
                        return self.err("{{ expected (module)");
                    }
                    let block = block.unwrap();
                    functions.insert(ident.clone(),Function { ident,params,return_type,block, });
                }

                // struct or tuple
                else if parser.keyword("struct") {

                    let ident = parser.ident();
                    if ident.is_none() {
                        return self.err("identifier expected");
                    }
                    let ident = ident.unwrap();

                    // struct
                    if parser.peek_group('{') {
                        let ident_types = parser.brace_ident_types()?;
                        let mut fields: Vec<(String,Type)> = Vec::new();
                        for (ident,type_) in ident_types.iter() {
                            fields.push((ident.clone(),type_.clone()));
                        }
                        structs.insert(ident.clone(),Struct { ident: ident.clone(),fields, });
                    }

                    // tuple
                    else if parser.peek_group('(') {
                        let types = parser.paren_types()?.unwrap();
                        tuple_structs.insert(ident.clone(),types);
                    }
                    else {
                        panic!("{}","{ or ( expected");
                    }
                }

                // enum
                else if parser.keyword("enum") {
                    let ident = parser.ident();
                    if ident.is_none() {
                        return self.err("identifier expected");
                    }
                    let ident = ident.unwrap();
                    if let Some(mut parser) = parser.group('{') {
                        let mut variants: Vec<(String,Variant)> = Vec::new();
                        while !parser.done() {
                            let ident = parser.ident();
                            if ident.is_none() {
                                return self.err("identifier expected");
                            }
                            let ident = ident.unwrap();
                            if parser.peek_group('{') {
                                let ident_types = parser.brace_ident_types()?;
                                let mut fields: Vec<(String,Type)> = Vec::new();
                                for (ident,type_) in ident_types.iter() {
                                    fields.push((ident.clone(),type_.clone()));
                                }
                                variants.push((ident,Variant::Struct(fields)));
                            }
                            else if parser.peek_group('(') {
                                let types = parser.paren_types()?.unwrap();
                                variants.push((ident,Variant::Tuple(types)));
                            }
                            else {
                                variants.push((ident,Variant::Naked));
                            }
                            parser.punct(',');
                        }
                        enums.insert(ident.clone(),Enum { ident,variants, });
                    }
                    else {
                        return self.err("{{ expected (module, enum)");
                    }
                }

                // const
                else if parser.keyword("const") {
                    let ident = parser.ident();
                    if ident.is_none() {
                        return self.err("identifier expected");
                    }
                    let ident = ident.unwrap();
                    parser.punct(':');
                    let type_ = parser.type_()?;
                    parser.punct('=');
                    let expr = parser.expr()?;
                    consts.insert(ident.clone(),Const { ident,type_,expr, });
                }

                // alias
                else if parser.keyword("type") {
                    let ident = parser.ident();
                    if ident.is_none() {
                        return self.err("identifier expected");
                    }
                    let ident = ident.unwrap();
                    parser.punct('=');
                    let type_ = parser.type_()?;
                    aliases.insert(ident.clone(),Alias { ident,type_, });
                }

                // skip a semicolon if any
                parser.punct(';');
            }
        }
        else {
            return self.err("{{ expected (expr)");
        }

        Ok(Module {
            ident,
            structs,
            tuple_structs,
            extern_structs: HashMap::new(),
            enums,
            aliases,
            consts,
            functions,
        })
    }

    pub(crate) fn struct_(&mut self) -> Result<Struct,String> {
        if !self.keyword("struct") {
            panic!("struct expected");
        }
        let ident = self.ident().expect("identifier expected");
        let ident_types = self.brace_ident_types()?;
        let mut fields: Vec<(String,Type)> = Vec::new();
        for (ident,type_) in ident_types.iter() {
            fields.push((ident.clone(),type_.clone(), ));
        }
        Ok(Struct { ident,fields, })
    }
}
