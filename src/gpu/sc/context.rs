use {
    super::*,
    super::ast::*,
    std::collections::HashMap,
};

pub struct Context {
    pub stdlib: StandardLib,
    pub alias_types: HashMap<&'static str,Type>,
    pub consts: HashMap<&'static str,Const>,
    pub structs: HashMap<&'static str,Struct>,
    pub tuple_types: HashMap<&'static str,Vec<Type>>,
    pub extern_structs: HashMap<&'static str,Struct>,
    pub anon_tuple_types: Vec<Vec<Type>>,
    pub enums: HashMap<String,Enum>,
    pub functions: HashMap<String,Function>,
}

impl Context {

    fn new() -> Context {
        Context {
            stdlib: StandardLib::new(),
            alias_types: HashMap::new(),
            consts: HashMap::new(),
            structs: HashMap::new(),
            tuple_types: HashMap::new(),
            extern_structs: HashMap::new(),
            anon_tuple_types: Vec::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
        }
    }
}