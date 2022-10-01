use {
    crate::ast::*,
    std::{
        collections::HashMap,
        rc::Rc,
    },
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

pub struct StandardLib {
    pub consts: HashMap<String,Rc<Const>>,
    pub structs: HashMap<String,Rc<Struct>>,
    pub enums: HashMap<String,Rc<Enum>>,
    pub functions: HashMap<String,Vec<Rc<Function>>>,
    pub methods: HashMap<String,Vec<Rc<Method>>>,
}

impl StandardLib {

    //const INT_TYPES: [Type; 5] = [Type::I8,Type::I16,Type::I32,Type::I64,Type::ISize];
    //const UINT_TYPES: [Type; 5] = [Type::U8,Type::U16,Type::U32,Type::U64,Type::USize];
    const FLOAT_TYPES: [Type; 3] = [Type::F16,Type::F32,Type::F64];
    const INT_FLOAT_TYPES: [Type; 8] = [Type::I8,Type::I16,Type::I32,Type::I64,Type::ISize,Type::F16,Type::F32,Type::F64];
    //const INT_UINT_TYPES: [Type; 10] = [Type::U8,Type::I8,Type::U16,Type::I16,Type::U32,Type::I32,Type::U64,Type::I64,Type::USize,Type::ISize];
    const INT_UINT_FLOAT_TYPES: [Type; 13] = [Type::U8,Type::I8,Type::U16,Type::I16,Type::U32,Type::I32,Type::U64,Type::I64,Type::USize,Type::ISize,Type::F16,Type::F32,Type::F64];

    fn insert_vector(&mut self,r: usize,comp_type: Type) {
        let ident = format!("Vec{}<{}>",r,comp_type);
        let mut fields: Vec<Symbol> = Vec::new();
        fields.push(Symbol { ident: "x".to_string(),type_: comp_type.clone(), });
        fields.push(Symbol { ident: "y".to_string(),type_: comp_type.clone(), });
        if r > 2 {
            fields.push(Symbol { ident: "z".to_string(),type_: comp_type.clone(), });
        }
        if r > 3 {
            fields.push(Symbol { ident: "w".to_string(),type_: comp_type.clone(), });
        }
        let struct_ = Rc::new(Struct {
            ident: ident.clone(),
            fields,
        });
        self.structs.insert(ident,struct_);
    }

    fn insert_matrix(&mut self,c: usize,r: usize,comp_type: Type) {
        let ident = format!("Mat{}x{}<{}>",c,r,comp_type);
        let component = &self.structs[&format!("Vec{}<{}>",r,comp_type)];
        let mut fields: Vec<Symbol> = Vec::new();
        fields.push(Symbol { ident: "x".to_string(),type_: Type::Struct(Rc::clone(&component)), });
        fields.push(Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&component)), });
        if c > 2 {
            fields.push(Symbol { ident: "z".to_string(),type_: Type::Struct(Rc::clone(&component)), });
        }
        if c > 3 {
            fields.push(Symbol { ident: "w".to_string(),type_: Type::Struct(Rc::clone(&component)), });
        }
        let struct_ = Rc::new(Struct {
            ident: ident.clone(),
            fields: fields,
        });
        self.structs.insert(ident,struct_);
    }

    fn insert_method(&mut self,from_type: &Type,ident: &str,params: Vec<Symbol>,type_: &Type) {
        let method = Rc::new(Method {
            from_type: from_type.clone(),
            ident: ident.to_string(),
            params,
            type_: type_.clone(),
        });
        if self.methods.contains_key(ident) {
            self.methods.get_mut(ident).unwrap().push(method);
        }
        else {
            self.methods.insert(ident.to_string(),vec![method]);
        }
    }

    pub fn new() -> StandardLib {

        let mut stdlib = StandardLib {
            consts: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            functions: HashMap::new(),
            methods: HashMap::new(),
        };

        // vector structs
        for r in 2..5 {
            stdlib.insert_vector(r,Type::Bool);
            for t in Self::INT_UINT_FLOAT_TYPES {
                stdlib.insert_vector(r,t);
            }
        }

        // matrix structs
        for c in 2..5 {
            for r in 2..5 {
                for t in Self::FLOAT_TYPES {
                    stdlib.insert_matrix(c,r,t);
                }
            }
        }

        // int+float scalar methods
        for t in Self::INT_FLOAT_TYPES {
            stdlib.insert_method(&t,"abs",vec![],&t);
            stdlib.insert_method(&t,"signum",vec![],&t);
        }

        // int+uint+float scalar methods
        for t in Self::INT_UINT_FLOAT_TYPES {
            stdlib.insert_method(&t,"min",vec![
                Symbol { ident: "other".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"max",vec![
                Symbol { ident: "other".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"clamp",vec![
                Symbol { ident: "low".to_string(),type_: t.clone(), },
                Symbol { ident: "high".to_string(),type_: t.clone(), },
            ],&t);
        }

        // float scalar methods
        for t in Self::FLOAT_TYPES {
            stdlib.insert_method(&t,"to_radians",vec![],&t);
            stdlib.insert_method(&t,"to_degrees",vec![],&t);
            stdlib.insert_method(&t,"sin",vec![],&t);
            stdlib.insert_method(&t,"cos",vec![],&t);
            stdlib.insert_method(&t,"tan",vec![],&t);
            stdlib.insert_method(&t,"sinh",vec![],&t);
            stdlib.insert_method(&t,"cosh",vec![],&t);
            stdlib.insert_method(&t,"tanh",vec![],&t);
            stdlib.insert_method(&t,"asin",vec![],&t);
            stdlib.insert_method(&t,"acos",vec![],&t);
            stdlib.insert_method(&t,"atan",vec![],&t);
            stdlib.insert_method(&t,"atan2",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"asinh",vec![],&t);
            stdlib.insert_method(&t,"acosh",vec![],&t);
            stdlib.insert_method(&t,"atanh",vec![],&t);
            stdlib.insert_method(&t,"powf",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"exp",vec![],&t);
            stdlib.insert_method(&t,"ln",vec![],&t);
            stdlib.insert_method(&t,"exp2",vec![],&t);
            stdlib.insert_method(&t,"log2",vec![],&t);
            stdlib.insert_method(&t,"sqrt",vec![],&t);
            stdlib.insert_method(&t,"invsqrt",vec![],&t);
            stdlib.insert_method(&t,"floor",vec![],&t);
            stdlib.insert_method(&t,"trunc",vec![],&t);
            stdlib.insert_method(&t,"round",vec![],&t);
            stdlib.insert_method(&t,"ceil",vec![],&t);
            stdlib.insert_method(&t,"fract",vec![],&t);
            stdlib.insert_method(&t,"rem_euclid",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"modf",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"mix",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
                Symbol { ident: "a".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"step",vec![
                Symbol { ident: "edge".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"smoothstep",vec![
                Symbol { ident: "edge0".to_string(),type_: t.clone(), },
                Symbol { ident: "edge1".to_string(),type_: t.clone(), },
            ],&t);
            stdlib.insert_method(&t,"is_nan",vec![],&Type::Bool);
            stdlib.insert_method(&t,"is_infinite",vec![],&Type::Bool);
            stdlib.insert_method(&t,"fma",vec![
                Symbol { ident: "y".to_string(),type_: t.clone(), },
                Symbol { ident: "z".to_string(),type_: t.clone(), },
            ],&t);
        }

        // vector methods
        for r in 2..5 {

            // boolean vector methods
            let from_type = Rc::clone(&stdlib.structs[&format!("Vec{}<bool>",r)]);
            stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"all",vec![],&Type::Bool);
            stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"any",vec![],&Type::Bool);
            stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"not",vec![],&Type::Struct(Rc::clone(&from_type)));

            // int+float vector methods
            for t in Self::INT_FLOAT_TYPES {
                let from_type = stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"abs",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"signum",vec![],&Type::Struct(Rc::clone(&from_type)));    
            }

            // int+uint+float vector methods
            for t in Self::INT_UINT_FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                stdlib.insert_method(
                    &Type::Struct(Rc::clone(&from_type)),
                    "min",
                    vec![
                        Symbol { ident: "other".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    ],
                    &Type::Struct(Rc::clone(&from_type)),
                );
                stdlib.insert_method(
                    &Type::Struct(Rc::clone(&from_type)),
                    "max",
                    vec![
                        Symbol { ident: "other".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    ],
                    &Type::Struct(Rc::clone(&from_type)),
                );
                stdlib.insert_method(
                    &Type::Struct(Rc::clone(&from_type)),
                    "clamp",
                    vec![
                        Symbol { ident: "low".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                        Symbol { ident: "high".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    ],
                    &Type::Struct(Rc::clone(&from_type)),
                );
                stdlib.insert_method(
                    &Type::Struct(Rc::clone(&from_type)),
                    "sclamp",
                    vec![
                        Symbol { ident: "low".to_string(),type_: t.clone(), },
                        Symbol { ident: "high".to_string(),type_: t.clone(), },
                    ],
                    &Type::Struct(Rc::clone(&from_type)),
                );
            }

            // float vector methods
            for t in Self::FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                let from_type_bool = &stdlib.structs[&format!("Vec{}<bool>",r)].clone();
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"to_radians",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"to_degrees",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"sin",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"cos",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"tan",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"sinh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"cosh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"tanh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"asin",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"acos",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"atan",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"atan2",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"asinh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"acosh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"atanh",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"powf",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"spowf",vec![
                    Symbol { ident: "y".to_string(),type_: t.clone(), },
                ],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"exp",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"ln",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"exp2",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"log2",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"sqrt",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"invsqrt",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"floor",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"trunc",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"round",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"ceil",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"fract",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"rem_euclid",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"modf",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"mix",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "a".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"smix",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "a".to_string(),type_: t.clone(), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"step",vec![
                    Symbol { ident: "edge".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"sstep",vec![
                    Symbol { ident: "edge".to_string(),type_: t.clone(), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"smoothstep",vec![
                    Symbol { ident: "edge0".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "edge1".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"smoothstep",vec![
                    Symbol { ident: "edge0".to_string(),type_: t.clone(), },
                    Symbol { ident: "edge1".to_string(),type_: t.clone(), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"is_nan",vec![],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"is_infinite",vec![],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"fma",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "z".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"length",vec![],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"distance",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"dot",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&t);
                if r == 3 {
                    stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"cross",vec![
                        Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    ],&Type::Struct(Rc::clone(&from_type)));
                }
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"normalize",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"faceforward",vec![
                    Symbol { ident: "i".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "n".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"reflect",vec![
                    Symbol { ident: "n".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"refract",vec![
                    Symbol { ident: "n".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    Symbol { ident: "e".to_string(),type_: t.clone(), },
                ],&Type::Struct(Rc::clone(&from_type)));
                for c in 2..5 {
                    let return_type = &stdlib.structs[&format!("Mat{}x{}<{}>",c,r,t)].clone();
                    let other_type = &stdlib.structs[&format!("Vec{}<{}>",c,t)].clone();
                    stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),&format!("outer{}",c),vec![
                        Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&other_type)), },
                    ],&Type::Struct(Rc::clone(&return_type)));
                }
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"less_than",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"less_than_equal",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"greater_than",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"greater_than_equal",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"equal",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"not_equal",vec![
                    Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                ],&Type::Struct(Rc::clone(&from_type_bool)));              
            }
        }

        // matrix methods
        for c in 2..5 {
            for r in 2..5 {
                for t in Self::FLOAT_TYPES {
                    let from_type = &stdlib.structs[&format!("Mat{}x{}<{}>",c,r,t)].clone();
                    stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"compmul",vec![
                        Symbol { ident: "y".to_string(),type_: Type::Struct(Rc::clone(&from_type)), },
                    ],&Type::Struct(Rc::clone(&from_type)));
                }
            }
        }

        //square matrix methods
        for cr in 2..5 {
            for t in Self::FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Mat{}x{}<{}>",cr,cr,t)].clone();
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"transpose",vec![],&Type::Struct(Rc::clone(&from_type)));
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"determinant",vec![],&t);
                stdlib.insert_method(&Type::Struct(Rc::clone(&from_type)),"inverse",vec![],&Type::Struct(Rc::clone(&from_type)));
            }
        }

        stdlib
    }
}

impl Display for StandardLib {
    fn fmt(&self,f: &mut Formatter) -> Result {
        for ident in self.consts.keys() {
            write!(f,"{}\n",self.consts[ident])?;
        }
        for ident in self.structs.keys() {
            write!(f,"{}\n",self.structs[ident])?;
        }
        for ident in self.enums.keys() {
            write!(f,"{}\n",self.enums[ident])?;
        }
        for ident in self.functions.keys() {
            for function in &self.functions[ident] {
                write!(f,"{}\n",function)?;
            }
        }
        for ident in self.methods.keys() {
            for method in &self.methods[ident] {
                write!(f,"{}\n",method)?;
            }
        }
        write!(f,"")
    }
}
