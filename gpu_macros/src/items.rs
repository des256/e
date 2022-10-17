use super::*;

impl Parser {

    pub(crate) fn module(&mut self) -> Module {
        self.keyword("mod");
        let ident = self.ident().expect("identifier expected");
        let mut aliases: Vec<Alias> = Vec::new();
        let mut tuples: Vec<Tuple> = Vec::new();
        let mut structs: Vec<Struct> = Vec::new();
        let mut enums: Vec<Enum> = Vec::new();
        let mut consts: Vec<Const> = Vec::new();
        let mut functions: Vec<Function> = Vec::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {

                // function
                if parser.keyword("fn") {
                    let ident = parser.ident().expect("identifier expected");
                    let ident_types = parser.paren_ident_types();
                    let mut params: Vec<(String,Type)> = Vec::new();
                    for (ident,type_) in ident_types.iter() {
                        params.push((ident.clone(),type_.clone()));
                    }
                    let type_ = if parser.punct2('-','>') {
                        parser.type_()
                    }
                    else {
                        Type::Void
                    };
                    let block = parser.block().expect("{ expected");
                    functions.push(Function { ident,params,type_,block, });
                }

                // struct or tuple
                else if parser.keyword("struct") {

                    let ident = parser.ident().expect("identifier expected");

                    // struct
                    if parser.peek_group('{') {
                        let ident_types = parser.brace_ident_types();
                        let mut fields: Vec<(String,Type)> = Vec::new();
                        for (ident,type_) in ident_types.iter() {
                            fields.push((ident.clone(),type_.clone()));
                        }
                        structs.push(Struct { ident: ident.clone(),fields, });
                    }

                    // tuple
                    else if parser.peek_group('(') {
                        let types = parser.paren_types().unwrap();
                        tuples.push(Tuple { ident,types, });
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
                                let ident_types = parser.brace_ident_types();
                                let mut fields: Vec<(String,Type)> = Vec::new();
                                for (ident,type_) in ident_types.iter() {
                                    fields.push((ident.clone(),type_.clone()));
                                }
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
                        enums.push(Enum { ident,variants, });
                    }
                    else {
                        self.fatal("{ expected");
                    }
                }

                // const
                else if parser.keyword("const") {
                    let ident = parser.ident().expect("identifier expected");
                    parser.punct(':');
                    let type_ = parser.type_();
                    parser.punct('=');
                    let expr = parser.expr();
                    consts.push(Const { ident,type_,expr, });
                }

                // alias
                else if parser.keyword("type") {
                    let ident = parser.ident().expect("identifier expected");
                    parser.punct('=');
                    let type_ = parser.type_();
                    aliases.push(Alias { ident,type_, });
                }

                // skip a semicolon if any
                parser.punct(';');
            }
        }
        else {
            self.fatal("{ expected");
        }

        Module {
            ident,
            aliases,
            tuples,
            structs,
            enums,
            consts,
            functions,
        }
    }

    pub(crate) fn struct_(&mut self) -> Struct {
        if !self.keyword("struct") {
            panic!("struct expected");
        }
        let ident = self.ident().expect("identifier expected");
        let ident_types = self.brace_ident_types();
        let mut fields: Vec<(String,Type)> = Vec::new();
        for (ident,type_) in ident_types.iter() {
            fields.push((ident.clone(),type_.clone(), ));
        }
        Struct { ident,fields, }
    }
}
