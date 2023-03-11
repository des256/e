use {
    super::*,
    super::ast::*,
    std::collections::HashMap,
};

pub struct Context {
    pub stdlib: StandardLib,
    pub alias_types: HashMap<&'static str,Type>,
    pub consts: HashMap<&'static str,Const>,
    pub structs: HashMap<String,Struct>,
    pub extern_structs: HashMap<String,Struct>,
    pub anon_tuple_structs: Vec<Vec<Type>>,
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
            extern_structs: HashMap::new(),
            enums: HashMap::new(),
            anon_tuple_structs: Vec::new(),
            functions: HashMap::new(),
        }
    }
}