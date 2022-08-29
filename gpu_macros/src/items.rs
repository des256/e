use {
    crate::*,
    std::collections::HashMap,
};

impl Parser {

    pub fn parse_module(&mut self) -> Module {
        self.keyword("mod");
        let ident = self.ident().expect("identifier expected");
        let mut functions: HashMap<String,(Vec<(String,Type)>,Option<Type>,Block)> = HashMap::new();
        let mut structs: HashMap<String,Vec<(String,Type)>> = HashMap::new();
        let mut tuples: HashMap<String,Vec<Type>> = HashMap::new();
        let mut enums: HashMap<String,Vec<Variant>> = HashMap::new();
        let mut consts: HashMap<String,(Type,Expr)> = HashMap::new();
        if let Some(mut parser) = self.group('{') {

            // function
            if self.keyword("fn") {
                let ident = self.ident().expect("identifier expected");
                let params = self.parse_paren_ident_types();
                let return_type = if self.punct2('-','>') {
                    Some(self.parse_type())
                }
                else {
                    None
                };
                let block = self.parse_block().expect("{{ expected");
                functions.insert(ident,(params,return_type,block));
            }

            // struct or tuple
            else if self.keyword("struct") {
                let ident = self.ident().expect("identifier expected");
                if self.peek_group('{') {
                    let fields = self.parse_brace_ident_types();
                    structs.insert(ident,fields);
                }
                else if self.peek_group('(') {
                    let types = self.parse_paren_types().unwrap();
                    tuples.insert(ident,types);
                }
                else {
                    panic!("{{ or ( expected");
                }
            }

            // enum
            else if self.keyword("enum") {
                let ident = self.ident().expect("identifier expected");
                if let Some(mut parser) = self.group('{') {
                    let mut variants: Vec<Variant> = Vec::new();
                    while !parser.done() {
                        let ident = parser.ident().expect("identifier expected");
                        if parser.peek_group('{') {
                            let fields = parser.parse_brace_ident_types();
                            variants.push(Variant::Struct(ident,fields));
                        }
                        else if parser.peek_group('(') {
                            let types = parser.parse_paren_types().unwrap();
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
                    panic!("{{ expected");
                }
            }

            // const
            else if self.keyword("const") {
                let ident = self.ident().expect("identifier expected");
                self.punct(':');
                let r#type = self.parse_type();
                self.punct('=');
                let expr = self.parse_expr();
                consts.insert(ident,(r#type,expr));
            }
        }
        else {
            panic!("{{ expected");
        }
        Module {
            ident,
            functions,
            structs,
            tuples,
            enums,
            consts,
        }
    }

    pub fn parse_struct(&mut self) -> (String,Vec<(String,Type)>) {
        if !self.keyword("struct") {
            panic!("struct expected");
        }
        let ident = self.ident().expect("identifier expected");
        let fields = self.parse_brace_ident_types();
        (ident,fields)
    }
}
