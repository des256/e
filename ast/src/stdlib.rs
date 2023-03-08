use {
    super::*,
    std::{
        collections::HashMap,
        fmt::{
            Display,
            Formatter,
            Result,
        }
    }
};

pub struct StandardLib {
    pub tuples: HashMap<String,Tuple>,
    pub structs: HashMap<String,Struct>,
    pub enums: HashMap<String,Enum>,
    pub aliases: HashMap<String,Alias>,
    pub consts: HashMap<String,Const>,
    pub functions: HashMap<String,Vec<Function>>,
    pub methods: HashMap<String,Vec<Method>>,
}

impl StandardLib {

    const FLOAT_TYPES: [Type; 3] = [
        Type::F16,
        Type::F32,
        Type::F64,
    ];
    const INT_FLOAT_TYPES: [Type; 7] = [
        Type::I8,
        Type::I16,
        Type::I32,
        Type::I64,
        Type::F16,
        Type::F32,
        Type::F64,
    ];
    const UINT_INT_FLOAT_TYPES: [Type; 11] = [
        Type::U8,
        Type::U16,
        Type::U32,
        Type::U64,
        Type::I8,
        Type::I16,
        Type::I32,
        Type::I64,
        Type::F16,
        Type::F32,
        Type::F64,
    ];

    // add vector to the structs
    fn insert_vector(&mut self,r: usize,type_: Type) {
        let ident = format!("Vec{}<{}>",r,type_);
        let mut fields: Vec<(String,Type)> = Vec::new();
        fields.push(("x".to_string(),type_.clone()));
        fields.push(("y".to_string(),type_.clone()));
        if r > 2 {
            fields.push(("z".to_string(),type_.clone()));
        }
        if r > 3 {
            fields.push(("w".to_string(),type_.clone()));
        }
        let struct_ = Struct {
            ident: ident.clone(),
            fields,
        };
        self.structs.insert(ident,struct_);
    }

    // add matrix to the structs
    fn insert_matrix(&mut self,c: usize,r: usize,type_: Type) {
        let ident = format!("Mat{}x{}<{}>",c,r,type_);
        let component = &self.structs[&format!("Vec{}<{}>",r,type_)];
        let mut fields: Vec<(String,Type)> = Vec::new();
        fields.push(("x".to_string(),Type::Struct(component.ident.clone())));
        fields.push(("y".to_string(),Type::Struct(component.ident.clone())));
        if c > 2 {
            fields.push(("z".to_string(),Type::Struct(component.ident.clone())));
        }
        if c > 3 {
            fields.push(("w".to_string(),Type::Struct(component.ident.clone())));
        }
        let struct_ = Struct {
            ident: ident.clone(),
            fields: fields,
        };
        self.structs.insert(ident,struct_);
    }

    // add method to a specific type
    fn insert_method(&mut self,from_type: &Type,ident: &str,params: Vec<(String,Type)>,return_type: &Type) {
        let method = Method {
            from_type: from_type.clone(),
            ident: ident.to_string(),
            params,
            return_type: return_type.clone(),
        };
        if self.methods.contains_key(ident) {
            self.methods.get_mut(ident).unwrap().push(method);
        }
        else {
            self.methods.insert(ident.to_string(),vec![method]);
        }
    }

    // create standard lib
    pub fn new() -> StandardLib {

        let mut stdlib = StandardLib {
            tuples: HashMap::new(),
            structs: HashMap::new(),
            enums: HashMap::new(),
            aliases: HashMap::new(),
            consts: HashMap::new(),
            functions: HashMap::new(),
            methods: HashMap::new(),
        };

        // insert vector structs
        for r in 2..5 {
            stdlib.insert_vector(r,Type::Bool);
            for t in Self::UINT_INT_FLOAT_TYPES {
                stdlib.insert_vector(r,t);
            }
        }

        // insert matrix structs
        for c in 2..5 {
            for r in 2..5 {
                for t in Self::FLOAT_TYPES {
                    stdlib.insert_matrix(c,r,t);
                }
            }
        }

        // insert int+float scalar methods
        for t in Self::INT_FLOAT_TYPES {
            stdlib.insert_method(&t,"abs",vec![],&t);
            stdlib.insert_method(&t,"signum",vec![],&t);
        }

        // insert int+uint+float scalar methods
        for t in Self::UINT_INT_FLOAT_TYPES {
            stdlib.insert_method(&t,"min",vec![
                ("other".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"max",vec![
                ("other".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"clamp",vec![
                ("low".to_string(),t.clone()),
                ("high".to_string(),t.clone()),
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
                ("y".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"asinh",vec![],&t);
            stdlib.insert_method(&t,"acosh",vec![],&t);
            stdlib.insert_method(&t,"atanh",vec![],&t);
            stdlib.insert_method(&t,"powf",vec![
                ("y".to_string(),t.clone()),
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
                ("y".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"modf",vec![
                ("y".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"mix",vec![
                ("y".to_string(),t.clone()),
                ("a".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"step",vec![
                ("edge".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"smoothstep",vec![
                ("edge0".to_string(),t.clone()),
                ("edge1".to_string(),t.clone()),
            ],&t);
            stdlib.insert_method(&t,"is_nan",vec![],&Type::Bool);
            stdlib.insert_method(&t,"is_infinite",vec![],&Type::Bool);
            stdlib.insert_method(&t,"fma",vec![
                ("y".to_string(),t.clone()),
                ("z".to_string(),t.clone()),
            ],&t);
        }

        // vector methods
        for r in 2..5 {

            // boolean vector methods
            let from_type = stdlib.structs[&format!("Vec{}<bool>",r)].clone();
            stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"all",vec![],&Type::Bool);
            stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"any",vec![],&Type::Bool);
            stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"not",vec![],&Type::Struct(from_type.ident.clone()));

            // int+float vector methods
            for t in Self::INT_FLOAT_TYPES {
                let from_type = stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"abs",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"signum",vec![],&Type::Struct(from_type.ident.clone()));
            }

            // int+uint+float vector methods
            for t in Self::UINT_INT_FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                stdlib.insert_method(
                    &Type::Struct(from_type.ident.clone()),
                    "min",
                    vec![
                        ("other".to_string(),Type::Struct(from_type.ident.clone())),
                    ],
                    &Type::Struct(from_type.ident.clone()),
                );
                stdlib.insert_method(
                    &Type::Struct(from_type.ident.clone()),
                    "max",
                    vec![
                        ("other".to_string(),Type::Struct(from_type.ident.clone())),
                    ],
                    &Type::Struct(from_type.ident.clone()),
                );
                stdlib.insert_method(
                    &Type::Struct(from_type.ident.clone()),
                    "clamp",
                    vec![
                        ("low".to_string(),Type::Struct(from_type.ident.clone())),
                        ("high".to_string(),Type::Struct(from_type.ident.clone())),
                    ],
                    &Type::Struct(from_type.ident.clone()),
                );
                stdlib.insert_method(
                    &Type::Struct(from_type.ident.clone()),
                    "sclamp",
                    vec![
                        ("low".to_string(),t.clone()),
                        ("high".to_string(),t.clone()),
                    ],
                    &Type::Struct(from_type.ident.clone()),
                );
            }

            // float vector methods
            for t in Self::FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Vec{}<{}>",r,t)].clone();
                let from_type_bool = &stdlib.structs[&format!("Vec{}<bool>",r)].clone();
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"to_radians",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"to_degrees",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"sin",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"cos",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"tan",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"sinh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"cosh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"tanh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"asin",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"acos",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"atan",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"atan2",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"asinh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"acosh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"atanh",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"powf",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"spowf",vec![
                    ("y".to_string(),t.clone()),
                ],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"exp",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"ln",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"exp2",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"log2",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"sqrt",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"invsqrt",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"floor",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"trunc",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"round",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"ceil",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"fract",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"rem_euclid",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"modf",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"mix",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                    ("a".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"smix",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                    ("a".to_string(),t.clone()),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"step",vec![
                    ("edge".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"sstep",vec![
                    ("edge".to_string(),t.clone()),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"smoothstep",vec![
                    ("edge0".to_string(),Type::Struct(from_type.ident.clone())),
                    ("edge1".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"smoothstep",vec![
                    ("edge0".to_string(),t.clone()),
                    ("edge1".to_string(),t.clone()),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"is_nan",vec![],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"is_infinite",vec![],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"fma",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                    ("z".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"length",vec![],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"distance",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"dot",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&t);
                if r == 3 {
                    stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"cross",vec![
                        ("y".to_string(),Type::Struct(from_type.ident.clone())),
                    ],&Type::Struct(from_type.ident.clone()));
                }
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"normalize",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"faceforward",vec![
                    ("i".to_string(),Type::Struct(from_type.ident.clone())),
                    ("n".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"reflect",vec![
                    ("n".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"refract",vec![
                    ("n".to_string(),Type::Struct(from_type.ident.clone())),
                    ("e".to_string(),t.clone()),
                ],&Type::Struct(from_type.ident.clone()));
                for c in 2..5 {
                    let return_type = &stdlib.structs[&format!("Mat{}x{}<{}>",c,r,t)].clone();
                    let other_type = &stdlib.structs[&format!("Vec{}<{}>",c,t)].clone();
                    stdlib.insert_method(&Type::Struct(from_type.ident.clone()),&format!("outer{}",c),vec![
                        ("y".to_string(),Type::Struct(other_type.ident.clone())),
                    ],&Type::Struct(return_type.ident.clone()));
                }
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"less_than",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"less_than_equal",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"greater_than",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"greater_than_equal",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"equal",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"not_equal",vec![
                    ("y".to_string(),Type::Struct(from_type.ident.clone())),
                ],&Type::Struct(from_type_bool.ident.clone()));              
            }
        }

        // insert matrix methods
        for c in 2..5 {
            for r in 2..5 {
                for t in Self::FLOAT_TYPES {
                    let from_type = &stdlib.structs[&format!("Mat{}x{}<{}>",c,r,t)].clone();
                    stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"compmul",vec![
                        ("y".to_string(),Type::Struct(from_type.ident.clone())),
                    ],&Type::Struct(from_type.ident.clone()));
                }
            }
        }

        // insert square matrix methods
        for cr in 2..5 {
            for t in Self::FLOAT_TYPES {
                let from_type = &stdlib.structs[&format!("Mat{}x{}<{}>",cr,cr,t)].clone();
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"transpose",vec![],&Type::Struct(from_type.ident.clone()));
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"determinant",vec![],&t);
                stdlib.insert_method(&Type::Struct(from_type.ident.clone()),"inverse",vec![],&Type::Struct(from_type.ident.clone()));
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
