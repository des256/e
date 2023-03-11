use {
    super::ast::*,
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
    pub structs: HashMap<&'static str,Struct>,
    pub enums: HashMap<&'static str,Enum>,
    pub aliases: HashMap<&'static str,Alias>,
    pub consts: HashMap<&'static str,Const>,
    pub functions: HashMap<&'static str,Vec<Function>>,
    pub methods: HashMap<&'static str,Vec<Method>>,
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

    // add method to a specific type
    fn insert_method(&mut self,from_type: &Type,ident: &'static str,params: Vec<(&'static str,Type)>,return_type: &Type) {
        let method = Method {
            from_type: from_type.clone(),
            ident,
            params,
            return_type: return_type.clone(),
        };
        if self.methods.contains_key(ident) {
            self.methods.get_mut(ident).unwrap().push(method);
        }
        else {
            self.methods.insert(ident,vec![method]);
        }
    }

    // insert boolean vector methods
    fn insert_vector_bool(&mut self,from_type: Type) {
        self.insert_method(&from_type,"all",vec![],&Type::Bool);
        self.insert_method(&from_type,"any",vec![],&Type::Bool);
        self.insert_method(&from_type,"not",vec![],&from_type);
    }

    // insert int+float vector methods
    fn insert_vector_int_float(&mut self,from_type: Type) {
        self.insert_method(&from_type,"abs",vec![],&from_type);
        self.insert_method(&from_type,"signum",vec![],&from_type);
    }

    // insert uint+int+float vector methods
    fn insert_vector_uint_int_float(&mut self,from_type: Type) {
        self.insert_method(&from_type,"min",vec![("other",from_type.clone()),],&from_type);
        self.insert_method(&from_type,"max",vec![("other",from_type.clone()),],&from_type);
        self.insert_method(&from_type,"clamp",vec![("low",from_type.clone()),("high",from_type.clone()),],&from_type);
        self.insert_method(&from_type,"sclamp",vec![("low",from_type.clone()),("high",from_type.clone()),],&from_type);
    }

    // insert float vector methods
    fn insert_vector_float(&mut self,from_type: Type,single_type: Type,bool_type: Type) {
        self.insert_method(&from_type,"to_radians",vec![],&from_type);
        self.insert_method(&from_type,"to_degrees",vec![],&from_type);
        self.insert_method(&from_type,"sin",vec![],&from_type);
        self.insert_method(&from_type,"cos",vec![],&from_type);
        self.insert_method(&from_type,"tan",vec![],&from_type);
        self.insert_method(&from_type,"sinh",vec![],&from_type);
        self.insert_method(&from_type,"cosh",vec![],&from_type);
        self.insert_method(&from_type,"tanh",vec![],&from_type);
        self.insert_method(&from_type,"asin",vec![],&from_type);
        self.insert_method(&from_type,"acos",vec![],&from_type);
        self.insert_method(&from_type,"atan",vec![],&from_type);
        self.insert_method(&from_type,"atan2",vec![("y",from_type.clone())],&single_type);
        self.insert_method(&from_type,"asinh",vec![],&from_type);
        self.insert_method(&from_type,"acosh",vec![],&from_type);
        self.insert_method(&from_type,"atanh",vec![],&from_type);
        self.insert_method(&from_type,"powf",vec![("y",from_type.clone())],&single_type);
        self.insert_method(&from_type,"spowf",vec![("y",single_type.clone())],&single_type);
        self.insert_method(&from_type,"exp",vec![],&from_type);
        self.insert_method(&from_type,"ln",vec![],&from_type);
        self.insert_method(&from_type,"exp2",vec![],&from_type);
        self.insert_method(&from_type,"log2",vec![],&from_type);
        self.insert_method(&from_type,"sqrt",vec![],&from_type);
        self.insert_method(&from_type,"invsqrt",vec![],&from_type);
        self.insert_method(&from_type,"floor",vec![],&from_type);
        self.insert_method(&from_type,"trunc",vec![],&from_type);
        self.insert_method(&from_type,"round",vec![],&from_type);
        self.insert_method(&from_type,"ceil",vec![],&from_type);
        self.insert_method(&from_type,"fract",vec![],&from_type);
        self.insert_method(&from_type,"rem_euclid",vec![("y",from_type.clone())],&from_type);
        self.insert_method(&from_type,"modf",vec![("y",from_type.clone())],&from_type);
        self.insert_method(&from_type,"mix",vec![("y",from_type.clone()),("a",from_type.clone())],&from_type);
        self.insert_method(&from_type,"smix",vec![("y",from_type.clone()),("a",single_type.clone())],&from_type);
        self.insert_method(&from_type,"step",vec![("edge",from_type.clone())],&from_type);
        self.insert_method(&from_type,"sstep",vec![("edge",single_type.clone())],&from_type);
        self.insert_method(&from_type,"smoothstep",vec![("edge0",from_type.clone()),("edge1",from_type.clone())],&from_type);
        self.insert_method(&from_type,"smoothstep",vec![("edge0",single_type.clone()),("edge1",single_type.clone())],&from_type);
        self.insert_method(&from_type,"is_nan",vec![],&bool_type);
        self.insert_method(&from_type,"is_infinite",vec![],&bool_type);
        self.insert_method(&from_type,"fma",vec![("y",from_type.clone()),("z",from_type.clone())],&from_type);
        self.insert_method(&from_type,"length",vec![],&single_type);
        self.insert_method(&from_type,"distance",vec![("y",from_type.clone())],&single_type);
        self.insert_method(&from_type,"dot",vec![("y",from_type.clone())],&single_type);
        self.insert_method(&from_type,"normalize",vec![],&from_type);
        self.insert_method(&from_type,"faceforward",vec![("i",from_type.clone()),("n",from_type.clone())],&from_type);
        self.insert_method(&from_type,"reflect",vec![("n",from_type.clone())],&from_type);
        self.insert_method(&from_type,"refract",vec![("n",from_type.clone()),("e",single_type.clone())],&from_type);
        self.insert_method(&from_type,"less_than",vec![("y",from_type.clone())],&bool_type);
        self.insert_method(&from_type,"less_than_equal",vec![("y",from_type.clone())],&bool_type);
        self.insert_method(&from_type,"greater_than",vec![("y",from_type.clone())],&bool_type);
        self.insert_method(&from_type,"greater_than_equal",vec![("y",from_type.clone())],&bool_type);
        self.insert_method(&from_type,"equal",vec![("y",from_type.clone())],&bool_type);
        self.insert_method(&from_type,"not_equal",vec![("y",from_type.clone())],&bool_type);
    }

    fn insert_vector_cross(&mut self,from_type: Type) {
        self.insert_method(&from_type,"cross",vec![("y",from_type.clone())],&from_type);
    }

    fn insert_vector_outer(&mut self,from_type: Type,vec2_type: Type,vec3_type: Type,vec4_type: Type,mat2_type: Type,mat3_type: Type,mat4_type: Type) {
        self.insert_method(&from_type,"outer2",vec![("y",vec2_type.clone())],&mat2_type);
        self.insert_method(&from_type,"outer3",vec![("y",vec3_type.clone())],&mat3_type);
        self.insert_method(&from_type,"outer4",vec![("y",vec4_type.clone())],&mat4_type);
    }

    // create standard lib
    pub fn new() -> StandardLib {

        let mut stdlib = StandardLib {
            structs: HashMap::new(),
            enums: HashMap::new(),
            aliases: HashMap::new(),
            consts: HashMap::new(),
            functions: HashMap::new(),
            methods: HashMap::new(),
        };

        // insert int+float scalar methods
        for t in Self::INT_FLOAT_TYPES {
            stdlib.insert_method(&t,"abs",vec![],&t);
            stdlib.insert_method(&t,"signum",vec![],&t);
        }

        // insert int+uint+float scalar methods
        for t in Self::UINT_INT_FLOAT_TYPES {
            stdlib.insert_method(&t,"min",vec![("other",t.clone())],&t);
            stdlib.insert_method(&t,"max",vec![("other",t.clone())],&t);
            stdlib.insert_method(&t,"clamp",vec![("low",t.clone()),("high",t.clone())],&t);
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
            stdlib.insert_method(&t,"atan2",vec![("y",t.clone())],&t);
            stdlib.insert_method(&t,"asinh",vec![],&t);
            stdlib.insert_method(&t,"acosh",vec![],&t);
            stdlib.insert_method(&t,"atanh",vec![],&t);
            stdlib.insert_method(&t,"powf",vec![("y",t.clone())],&t);
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
            stdlib.insert_method(&t,"rem_euclid",vec![("y",t.clone())],&t);
            stdlib.insert_method(&t,"modf",vec![("y",t.clone())],&t);
            stdlib.insert_method(&t,"mix",vec![("y",t.clone()),("a",t.clone())],&t);
            stdlib.insert_method(&t,"step",vec![("edge",t.clone())],&t);
            stdlib.insert_method(&t,"smoothstep",vec![("edge0",t.clone()),("edge1",t.clone())],&t);
            stdlib.insert_method(&t,"is_nan",vec![],&Type::Bool);
            stdlib.insert_method(&t,"is_infinite",vec![],&Type::Bool);
            stdlib.insert_method(&t,"fma",vec![("y",t.clone()),("z",t.clone())],&t);
        }

        // vector methods

        // boolean vector methods
        stdlib.insert_vector_bool(Type::Vec2Bool);
        stdlib.insert_vector_bool(Type::Vec3Bool);
        stdlib.insert_vector_bool(Type::Vec4Bool);

        // int+float vector methods
        stdlib.insert_vector_int_float(Type::Vec2I8);
        stdlib.insert_vector_int_float(Type::Vec2I16);
        stdlib.insert_vector_int_float(Type::Vec2I32);
        stdlib.insert_vector_int_float(Type::Vec2I64);
        stdlib.insert_vector_int_float(Type::Vec2F16);
        stdlib.insert_vector_int_float(Type::Vec2F32);
        stdlib.insert_vector_int_float(Type::Vec2F64);
        stdlib.insert_vector_int_float(Type::Vec3I8);
        stdlib.insert_vector_int_float(Type::Vec3I16);
        stdlib.insert_vector_int_float(Type::Vec3I32);
        stdlib.insert_vector_int_float(Type::Vec3I64);
        stdlib.insert_vector_int_float(Type::Vec3F16);
        stdlib.insert_vector_int_float(Type::Vec3F32);
        stdlib.insert_vector_int_float(Type::Vec3F64);
        stdlib.insert_vector_int_float(Type::Vec4I8);
        stdlib.insert_vector_int_float(Type::Vec4I16);
        stdlib.insert_vector_int_float(Type::Vec4I32);
        stdlib.insert_vector_int_float(Type::Vec4I64);
        stdlib.insert_vector_int_float(Type::Vec4F16);
        stdlib.insert_vector_int_float(Type::Vec4F32);
        stdlib.insert_vector_int_float(Type::Vec4F64);

        // int+uint+float vector methods
        stdlib.insert_vector_uint_int_float(Type::Vec2U8);
        stdlib.insert_vector_uint_int_float(Type::Vec2I8);
        stdlib.insert_vector_uint_int_float(Type::Vec2U16);
        stdlib.insert_vector_uint_int_float(Type::Vec2I16);
        stdlib.insert_vector_uint_int_float(Type::Vec2U32);
        stdlib.insert_vector_uint_int_float(Type::Vec2I32);
        stdlib.insert_vector_uint_int_float(Type::Vec2U64);
        stdlib.insert_vector_uint_int_float(Type::Vec2I64);
        stdlib.insert_vector_uint_int_float(Type::Vec2F16);
        stdlib.insert_vector_uint_int_float(Type::Vec2F32);
        stdlib.insert_vector_uint_int_float(Type::Vec2F64);
        stdlib.insert_vector_uint_int_float(Type::Vec3U8);
        stdlib.insert_vector_uint_int_float(Type::Vec3I8);
        stdlib.insert_vector_uint_int_float(Type::Vec3U16);
        stdlib.insert_vector_uint_int_float(Type::Vec3I16);
        stdlib.insert_vector_uint_int_float(Type::Vec3U32);
        stdlib.insert_vector_uint_int_float(Type::Vec3I32);
        stdlib.insert_vector_uint_int_float(Type::Vec3U64);
        stdlib.insert_vector_uint_int_float(Type::Vec3I64);
        stdlib.insert_vector_uint_int_float(Type::Vec3F16);
        stdlib.insert_vector_uint_int_float(Type::Vec3F32);
        stdlib.insert_vector_uint_int_float(Type::Vec3F64);
        stdlib.insert_vector_uint_int_float(Type::Vec4U8);
        stdlib.insert_vector_uint_int_float(Type::Vec4I8);
        stdlib.insert_vector_uint_int_float(Type::Vec4U16);
        stdlib.insert_vector_uint_int_float(Type::Vec4I16);
        stdlib.insert_vector_uint_int_float(Type::Vec4U32);
        stdlib.insert_vector_uint_int_float(Type::Vec4I32);
        stdlib.insert_vector_uint_int_float(Type::Vec4U64);
        stdlib.insert_vector_uint_int_float(Type::Vec4I64);
        stdlib.insert_vector_uint_int_float(Type::Vec4F16);
        stdlib.insert_vector_uint_int_float(Type::Vec4F32);
        stdlib.insert_vector_uint_int_float(Type::Vec4F64);

        // float vector methods
        stdlib.insert_vector_float(Type::Vec2F16,Type::F16,Type::Vec2Bool);
        stdlib.insert_vector_float(Type::Vec2F32,Type::F32,Type::Vec2Bool);
        stdlib.insert_vector_float(Type::Vec2F64,Type::F64,Type::Vec2Bool);
        stdlib.insert_vector_float(Type::Vec3F16,Type::F16,Type::Vec3Bool);
        stdlib.insert_vector_float(Type::Vec3F32,Type::F32,Type::Vec3Bool);
        stdlib.insert_vector_float(Type::Vec3F64,Type::F64,Type::Vec3Bool);
        stdlib.insert_vector_float(Type::Vec4F16,Type::F16,Type::Vec4Bool);
        stdlib.insert_vector_float(Type::Vec4F32,Type::F32,Type::Vec4Bool);
        stdlib.insert_vector_float(Type::Vec4F64,Type::F64,Type::Vec4Bool);
        stdlib.insert_vector_cross(Type::Vec3F16);
        stdlib.insert_vector_cross(Type::Vec3F32);
        stdlib.insert_vector_cross(Type::Vec3F64);
        stdlib.insert_vector_outer(Type::Vec2F32,Type::Vec2F32,Type::Vec3F32,Type::Vec4F32,Type::Mat2x2F32,Type::Mat3x2F32,Type::Mat4x2F32);
        stdlib.insert_vector_outer(Type::Vec2F64,Type::Vec2F64,Type::Vec3F64,Type::Vec4F64,Type::Mat2x2F64,Type::Mat3x2F64,Type::Mat4x2F64);
        stdlib.insert_vector_outer(Type::Vec3F32,Type::Vec2F32,Type::Vec3F32,Type::Vec4F32,Type::Mat2x3F32,Type::Mat3x3F32,Type::Mat4x3F32);
        stdlib.insert_vector_outer(Type::Vec3F64,Type::Vec2F64,Type::Vec3F64,Type::Vec4F64,Type::Mat2x3F64,Type::Mat3x3F64,Type::Mat4x3F64);
        stdlib.insert_vector_outer(Type::Vec4F32,Type::Vec2F32,Type::Vec3F32,Type::Vec4F32,Type::Mat2x4F32,Type::Mat3x4F32,Type::Mat4x4F32);
        stdlib.insert_vector_outer(Type::Vec4F64,Type::Vec2F64,Type::Vec3F64,Type::Vec4F64,Type::Mat2x4F64,Type::Mat3x4F64,Type::Mat4x4F64);

        // insert matrix methods
        stdlib.insert_method(&Type::Mat2x2F32,"compmul",vec![("y",Type::Mat2x2F32)],&Type::Mat2x2F32);
        stdlib.insert_method(&Type::Mat2x2F64,"compmul",vec![("y",Type::Mat2x2F64)],&Type::Mat2x2F64);
        stdlib.insert_method(&Type::Mat2x3F32,"compmul",vec![("y",Type::Mat2x3F32)],&Type::Mat2x3F32);
        stdlib.insert_method(&Type::Mat2x3F64,"compmul",vec![("y",Type::Mat2x3F64)],&Type::Mat2x3F64);
        stdlib.insert_method(&Type::Mat2x4F32,"compmul",vec![("y",Type::Mat2x4F32)],&Type::Mat2x4F32);
        stdlib.insert_method(&Type::Mat2x4F64,"compmul",vec![("y",Type::Mat2x4F64)],&Type::Mat2x4F64);
        stdlib.insert_method(&Type::Mat3x2F32,"compmul",vec![("y",Type::Mat3x2F32)],&Type::Mat3x2F32);
        stdlib.insert_method(&Type::Mat3x2F64,"compmul",vec![("y",Type::Mat3x2F64)],&Type::Mat3x2F64);
        stdlib.insert_method(&Type::Mat3x3F32,"compmul",vec![("y",Type::Mat3x3F32)],&Type::Mat3x3F32);
        stdlib.insert_method(&Type::Mat3x3F64,"compmul",vec![("y",Type::Mat3x3F64)],&Type::Mat3x3F64);
        stdlib.insert_method(&Type::Mat3x4F32,"compmul",vec![("y",Type::Mat3x4F32)],&Type::Mat3x4F32);
        stdlib.insert_method(&Type::Mat3x4F64,"compmul",vec![("y",Type::Mat3x4F64)],&Type::Mat3x4F64);
        stdlib.insert_method(&Type::Mat4x2F32,"compmul",vec![("y",Type::Mat4x2F32)],&Type::Mat4x2F32);
        stdlib.insert_method(&Type::Mat4x2F64,"compmul",vec![("y",Type::Mat4x2F64)],&Type::Mat4x2F64);
        stdlib.insert_method(&Type::Mat4x3F32,"compmul",vec![("y",Type::Mat4x3F32)],&Type::Mat4x3F32);
        stdlib.insert_method(&Type::Mat4x3F64,"compmul",vec![("y",Type::Mat4x3F64)],&Type::Mat4x3F64);
        stdlib.insert_method(&Type::Mat4x4F32,"compmul",vec![("y",Type::Mat4x4F32)],&Type::Mat4x4F32);
        stdlib.insert_method(&Type::Mat4x4F64,"compmul",vec![("y",Type::Mat4x4F64)],&Type::Mat4x4F64);

        // insert square matrix methods
        stdlib.insert_method(&Type::Mat2x2F32,"transpose",vec![],&Type::Mat2x2F32);
        stdlib.insert_method(&Type::Mat2x2F64,"transpose",vec![],&Type::Mat2x2F64);
        stdlib.insert_method(&Type::Mat3x3F32,"transpose",vec![],&Type::Mat3x3F32);
        stdlib.insert_method(&Type::Mat3x3F64,"transpose",vec![],&Type::Mat3x3F64);
        stdlib.insert_method(&Type::Mat4x4F32,"transpose",vec![],&Type::Mat4x4F32);
        stdlib.insert_method(&Type::Mat4x4F64,"transpose",vec![],&Type::Mat4x4F64);
        stdlib.insert_method(&Type::Mat2x2F32,"determinant",vec![],&Type::Mat2x2F32);
        stdlib.insert_method(&Type::Mat2x2F64,"determinant",vec![],&Type::Mat2x2F64);
        stdlib.insert_method(&Type::Mat3x3F32,"determinant",vec![],&Type::Mat3x3F32);
        stdlib.insert_method(&Type::Mat3x3F64,"determinant",vec![],&Type::Mat3x3F64);
        stdlib.insert_method(&Type::Mat4x4F32,"determinant",vec![],&Type::Mat4x4F32);
        stdlib.insert_method(&Type::Mat4x4F64,"determinant",vec![],&Type::Mat4x4F64);
        stdlib.insert_method(&Type::Mat2x2F32,"inverse",vec![],&Type::Mat2x2F32);
        stdlib.insert_method(&Type::Mat2x2F64,"inverse",vec![],&Type::Mat2x2F64);
        stdlib.insert_method(&Type::Mat3x3F32,"inverse",vec![],&Type::Mat3x3F32);
        stdlib.insert_method(&Type::Mat3x3F64,"inverse",vec![],&Type::Mat3x3F64);
        stdlib.insert_method(&Type::Mat4x4F32,"inverse",vec![],&Type::Mat4x4F32);
        stdlib.insert_method(&Type::Mat4x4F64,"inverse",vec![],&Type::Mat4x4F64);

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
