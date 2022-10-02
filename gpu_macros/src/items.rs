use {
    crate::*,
    sr::*,
    std::{
        collections::HashMap,
        rc::Rc,
        cell::RefCell,
    },
};

impl Parser {

    pub(crate) fn module(&mut self) -> ast::Module {
        self.keyword("mod");
        let ident = self.ident().expect("identifier expected");
        let mut aliases: HashMap<String,Rc<RefCell<ast::Alias>>> = HashMap::new();
        let mut tuples: HashMap<String,Rc<RefCell<ast::Tuple>>> = HashMap::new();
        let mut structs: HashMap<String,Rc<RefCell<ast::Struct>>> = HashMap::new();
        let mut enums: HashMap<String,Rc<RefCell<ast::Enum>>> = HashMap::new();
        let mut consts: HashMap<String,Rc<RefCell<ast::Const>>> = HashMap::new();
        let mut functions: HashMap<String,Rc<RefCell<ast::Function>>> = HashMap::new();
        if let Some(mut parser) = self.group('{') {
            while !parser.done() {

                // function
                if parser.keyword("fn") {
                    let ident = parser.ident().expect("identifier expected");
                    let ident_types = parser.paren_ident_types();
                    let mut params: Vec<Rc<RefCell<ast::Symbol>>> = Vec::new();
                    for (ident,type_) in ident_types.iter() {
                        params.push(Rc::new(RefCell::new(ast::Symbol { ident: ident.clone(),type_: type_.clone(), })));
                    }
                    let type_ = if parser.punct2('-','>') {
                        parser.type_()
                    }
                    else {
                        ast::Type::Void
                    };
                    let block = parser.block().expect("{ expected");
                    functions.insert(ident.clone(),Rc::new(RefCell::new(ast::Function { ident,params,type_,block, })));
                }

                // struct or tuple
                else if parser.keyword("struct") {

                    let ident = parser.ident().expect("identifier expected");

                    // struct
                    if parser.peek_group('{') {
                        let ident_types = parser.brace_ident_types();
                        let mut fields: Vec<ast::Symbol> = Vec::new();
                        for (ident,type_) in ident_types.iter() {
                            fields.push(ast::Symbol { ident: ident.clone(),type_: type_.clone(), });
                        }
                        structs.insert(ident.clone(),Rc::new(RefCell::new(ast::Struct { ident,fields, })));
                    }

                    // tuple
                    else if parser.peek_group('(') {
                        let types = parser.paren_types().unwrap();
                        tuples.insert(ident.clone(),Rc::new(RefCell::new(ast::Tuple { ident,types, })));
                    }
                    else {
                        panic!("{}","{ or ( expected");
                    }
                }

                // enum
                else if parser.keyword("enum") {
                    let ident = parser.ident().expect("identifier expected");
                    if let Some(mut parser) = parser.group('{') {
                        let mut variants: Vec<ast::Variant> = Vec::new();
                        while !parser.done() {
                            let ident = parser.ident().expect("identifier expected");
                            if parser.peek_group('{') {
                                let ident_types = parser.brace_ident_types();
                                let mut fields: Vec<ast::Symbol> = Vec::new();
                                for (ident,type_) in ident_types.iter() {
                                    fields.push(ast::Symbol { ident: ident.clone(),type_: type_.clone(), });
                                }
                                variants.push(ast::Variant::Struct(ident,fields));
                            }
                            else if parser.peek_group('(') {
                                let types = parser.paren_types().unwrap();
                                variants.push(ast::Variant::Tuple(ident,types));
                            }
                            else {
                                variants.push(ast::Variant::Naked(ident));
                            }
                            parser.punct(',');
                        }
                        enums.insert(ident.clone(),Rc::new(RefCell::new(ast::Enum { ident,variants, })));
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
                    consts.insert(ident.clone(),Rc::new(RefCell::new(ast::Const { ident,type_,expr, })));
                }

                // alias
                else if parser.keyword("type") {
                    let ident = parser.ident().expect("identifier expected");
                    parser.punct('=');
                    let type_ = parser.type_();
                    aliases.insert(ident.clone(),Rc::new(RefCell::new(ast::Alias { ident,type_, })));
                }

                // skip a semicolon if any
                parser.punct(';');
            }
        }
        else {
            self.fatal("{ expected");
        }

        ast::Module {
            ident,
            aliases,
            tuples,
            structs,
            enums,
            consts,
            functions,
        }
    }

    pub(crate) fn struct_(&mut self) -> ast::Struct {
        if !self.keyword("struct") {
            panic!("struct expected");
        }
        let ident = self.ident().expect("identifier expected");
        let ident_types = self.brace_ident_types();
        let mut fields: Vec<ast::Symbol> = Vec::new();
        for (ident,type_) in ident_types.iter() {
            fields.push(ast::Symbol { ident: ident.clone(),type_: type_.clone(), });
        }
        ast::Struct { ident,fields, }
    }
}
