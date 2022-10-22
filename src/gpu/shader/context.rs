use {
    super::*,
    std::{
        collections::HashMap,
    }
};

use ast::*;

pub struct Context {
    pub stdlib: StandardLib,
    pub tuples: HashMap<String,Tuple>,
    pub structs: HashMap<String,Struct>,
    pub enums: HashMap<String,Enum>,
    pub consts: HashMap<String,Const>,
    pub functions: HashMap<String,Function>,
    pub aliases: HashMap<String,Alias>,
    pub locals: HashMap<String,Symbol>,
    pub params: HashMap<String,Symbol>,
}

impl Context {

    pub fn new(source: &Source) -> Context {

        let stdlib = StandardLib::new();

        let tuples: HashMap<String,Tuple> = HashMap::new();
        for tuple in source.tuples.iter() {
            tuples.insert(tuple.ident.clone(),tuple.clone());
        }

        let structs: HashMap<String,Struct> = HashMap::new();
        for struct_ in source.structs.iter() {
            structs.insert(struct_.ident.clone(),struct_.clone());
        }

        let enums: HashMap<String,Enum> = HashMap::new();
        for enum_ in source.enums.iter() {
            enums.insert(enum_.ident.clone(),enum_.clone());
        }

        let consts: HashMap<String,Const> = HashMap::new();
        for const_ in source.consts.iter() {
            consts.insert(const_.ident.clone(),const_.clone());
        }

        let aliases: HashMap<String,Alias> = HashMap::new();
        for alias in source.aliases.iter() {
            aliases.insert(alias.ident.clone(),alias.clone());
        }

        let functions: HashMap<String,Function> = HashMap::new();
        for function in source.functions.iter() {
            functions.insert(function.ident.clone(),function.clone());
        }
        
        Context {
            stdlib,
            tuples,
            structs,
            enums,
            consts,
            aliases,
            functions,
            locals: HashMap::new(),
            params: HashMap::new(),
        }
    }

    fn enter_function(&mut self,function: &Function) {
        self.params.clear();
        for param in function.params.iter() {
            self.params.insert(param.ident.clone(),param.clone());
        }
    }

    fn leave_function(&mut self) {
        self.params.clear();
    }

    fn enter_block(&mut self) { }

    fn leave_block(&mut self) {
        self.locals.clear();
    }

    fn register_local(&mut self,local: &Symbol) {
        self.locals.insert(local.ident.clone(),local.clone());
    }
}
