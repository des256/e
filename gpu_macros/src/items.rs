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
        let mut enums: HashMap<String,Vec<Variant>> = HashMap::new();
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
                        enums.insert(ident,variants);
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
