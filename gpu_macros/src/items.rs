use {
    crate::*,
    std::{
        cell::RefCell,
        collections::HashMap,
    },
};

impl Parser {

    pub fn parse_module(&mut self) -> Module {
        self.keyword("mod");
        let ident = self.ident().expect("identifier expected");
        let mut functions: HashMap<String,(Vec<(String,Type)>,RefCell<Type>,RefCell<Block>)> = HashMap::new();
        let mut structs: HashMap<String,Vec<(String,RefCell<Type>)>> = HashMap::new();
        let mut tuples: HashMap<String,Vec<RefCell<Type>>> = HashMap::new();
        let mut enums: HashMap<String,Vec<Variant>> = HashMap::new();
        let mut consts: HashMap<String,(RefCell<Type>,RefCell<Expr>)> = HashMap::new();
        if let Some(mut parser) = self.group('{') {

            // function
            if parser.keyword("fn") {
                let ident = parser.ident().expect("identifier expected");
                let params = parser.parse_paren_ident_types();
                let return_type = if parser.punct2('-','>') {
                    self.parse_type()
                }
                else {
                    Type::Void
                };
                let block = parser.parse_block().expect("{{ expected");
                functions.insert(ident,(params,RefCell::new(return_type),RefCell::new(block)));
            }

            // struct or tuple
            else if parser.keyword("struct") {
                let ident = parser.ident().expect("identifier expected");
                if parser.peek_group('{') {
                    let ident_types = parser.parse_brace_ident_types();
                    let mut fields: Vec<(String,RefCell<Type>)> = Vec::new();
                    for (ident,ty) in ident_types {
                        fields.push((ident.clone(),RefCell::new(ty)))
                    }
                    structs.insert(ident,fields);
                }
                else if parser.peek_group('(') {
                    let tys = parser.parse_paren_types().unwrap();
                    let mut types: Vec<RefCell<Type>> = Vec::new();
                    for ty in tys {
                        types.push(RefCell::new(ty));
                    }
                    tuples.insert(ident,types);
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
                    panic!("{}","{ expected");
                }
            }

            // const
            else if parser.keyword("const") {
                let ident = parser.ident().expect("identifier expected");
                parser.punct(':');
                let r#type = parser.parse_type();
                parser.punct('=');
                let expr = parser.parse_expr();
                consts.insert(ident,(RefCell::new(r#type),RefCell::new(expr)));
            }
        }
        else {
            panic!("{}","{ expected");
        }
        Module {
            ident,
            functions: RefCell::new(functions),
            structs: RefCell::new(structs),
            tuples: RefCell::new(tuples),
            enums: RefCell::new(enums),
            consts: RefCell::new(consts),
            anon_tuple_count: RefCell::new(0),
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
