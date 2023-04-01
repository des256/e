use {
    std::fmt,
    super::*,
};

struct Context {
    stdlib: StandardLib,
    module: ProcessedModule,
}

impl Context {

    fn emit_type(&self,type_: &Type) -> Result<String,String> {
        match type_ {
            Type::Inferred => Err("unable to infer type".to_string()),
            Type::Void => Err("() not allowed".to_string()),
            Type::Bool => Ok("bool".to_string()),
            Type::U8 | Type::U16 | Type::U32 => Ok("uint".to_string()),
            Type::I8 | Type::I16 | Type::I32 => Ok("int".to_string()),
            Type::U64 => Err("u64 not supported in GLSL".to_string()),
            Type::I64 => Err("i64 not supported in GLSL".to_string()),
            Type::F16 | Type::F32 => Ok("float".to_string()),
            Type::F64 => Ok("double".to_string()),
            Type::Vec2Bool => Ok("bvec2".to_string()),
            Type::Vec2U8 | Type::Vec2U16 | Type::Vec2U32 => Ok("uvec2".to_string()),
            Type::Vec2I8 | Type::Vec2I16 | Type::Vec2I32 => Ok("ivec2".to_string()),
            Type::Vec2U64 => Err("Vec2<u64> not supported in GLSL".to_string()),
            Type::Vec2I64 => Err("Vec2<i64> not supported in GLSL".to_string()),
            Type::Vec2F16 | Type::Vec2F32 => Ok("vec2".to_string()),
            Type::Vec2F64 => Ok("dvec2".to_string()),
            Type::Vec3Bool => Ok("bvec3".to_string()),
            Type::Vec3U8 | Type::Vec3U16 | Type::Vec3U32 => Ok("uvec3".to_string()),
            Type::Vec3I8 | Type::Vec3I16 | Type::Vec3I32 => Ok("ivec3".to_string()),
            Type::Vec3U64 => Err("Vec3<u64> not supported in GLSL".to_string()),
            Type::Vec3I64 => Err("Vec3<i64> not supported in GLSL".to_string()),
            Type::Vec3F16 | Type::Vec3F32 => Ok("vec3".to_string()),
            Type::Vec3F64 => Ok("dvec3".to_string()),
            Type::Vec4Bool => Ok("bvec4".to_string()),
            Type::Vec4U8 | Type::Vec4U16 | Type::Vec4U32 => Ok("uvec4".to_string()),
            Type::Vec4I8 | Type::Vec4I16 | Type::Vec4I32 => Ok("ivec4".to_string()),
            Type::Vec4U64 => Err("Vec4<u64> not supported in GLSL".to_string()),
            Type::Vec4I64 => Err("Vec4<i64> not supported in GLSL".to_string()),
            Type::Vec4F16 | Type::Vec4F32 => Ok("vec4".to_string()),
            Type::Vec4F64 => Ok("dvec4".to_string()),
            Type::Mat2x2F32 => Ok("mat2".to_string()),
            Type::Mat2x3F32 => Ok("mat2x3".to_string()),
            Type::Mat2x4F32 => Ok("mat2x4".to_string()),
            Type::Mat3x2F32 => Ok("mat3x2".to_string()),
            Type::Mat3x3F32 => Ok("mat3".to_string()),
            Type::Mat3x4F32 => Ok("mat3x4".to_string()),
            Type::Mat4x2F32 => Ok("mat4x2".to_string()),
            Type::Mat4x3F32 => Ok("mat4x3".to_string()),
            Type::Mat4x4F32 => Ok("mat4".to_string()),
            Type::Mat2x2F64 => Ok("dmat2".to_string()),
            Type::Mat2x3F64 => Ok("dmat2x3".to_string()),
            Type::Mat2x4F64 => Ok("dmat2x4".to_string()),
            Type::Mat3x2F64 => Ok("dmat3x2".to_string()),
            Type::Mat3x3F64 => Ok("dmat3".to_string()),
            Type::Mat3x4F64 => Ok("dmat3x4".to_string()),
            Type::Mat4x2F64 => Ok("dmat4x2".to_string()),
            Type::Mat4x3F64 => Ok("dmat4x3".to_string()),
            Type::Mat4x4F64 => Ok("dmat4".to_string()),
            Type::AnonTuple(_) => Err("anonymous tuple cannot occur here".to_string()),
            Type::Array(_,_) => Err("TODO: array".to_string()),
            Type::Ident(_) => Err("unknown identifier cannot occur here".to_string()),
            Type::AnonTupleRef(_) => Err("anonymous tuple cannot occur here".to_string()),
            Type::StructRef(_) => Err("struct cannot occur here".to_string()),
            Type::TupleRef(_) => Err("tuple cannot occur here".to_string()),
            Type::EnumRef(_) => Err("enum cannot occur here".to_string()),
        }
    }

    fn emit_expr(&self,expr: &Expr) -> Result<String,String> {
        match expr {
            Expr::Boolean(value) => if *value { Ok("true".to_string()) } else { Ok("false".to_string()) },
            Expr::Integer(value) => Ok(format!("{}",value)),
            Expr::Float(value) => Ok(format!("{}",value)),
            Expr::Array(exprs) => Err("TODO: array literal".to_string()),
            Expr::Cloned(expr,count) => Err("TODO: cloned array literal".to_string()),
            Expr::Index(expr,index) => Ok(format!("{}[{}]",self.emit_expr(expr)?,self.emit_expr(index)?)),
            Expr::Cast(expr,type_) => Ok(format!("({}){}",self.emit_type(type_)?,self.emit_expr(expr)?)),
            Expr::AnonTuple(_) => Err("anonymous tuple cannot occur here".to_string()),
            Expr::Unary(op,expr) => Ok(format!("({}({}))",op,self.emit_expr(expr)?)),
            Expr::Binary(expr1,op,expr2) => Ok(format!("(({}){}({}))",self.emit_expr(expr1)?,op,self.emit_expr(expr2)?)),
            Expr::Continue => Ok("continue".to_string()),
            Expr::Break(expr) => Ok("break".to_string()),
            Expr::Return(expr) => if let Some(expr) = expr { Ok(format!("return {}",self.emit_expr(expr)?)) } else { Ok("return".to_string()) },
            Expr::Block(block) => self.emit_block(block),
            Expr::If(cond_expr,block,else_expr) => Err("TODO: if-statement".to_string()),
            Expr::While(cond_expr,block) => Err("TODO: while-statement".to_string()),
            Expr::Loop(block) => Err("TODO: loop-statement".to_string()),
            Expr::IfLet(_,_,_,_) => Err("if-let statement cannot occur here".to_string()),
            Expr::For(_,_,_) => Err("for statement cannot occur here".to_string()),
            Expr::WhileLet(_,_,_) => Err("while-let statement cannot occur here".to_string()),
            Expr::Match(_,_) => Err("match statement cannot occur here".to_string()),
            Expr::Ident(ident) => Err(format!("unknown identifier {}",ident)),
            Expr::TupleLitOrFunctionCall(_,_) => Err("named tuple literal or function call cannot occur here".to_string()),
            Expr::StructLit(ident,fields) => Err("TODO: struct literal".to_string()),
            Expr::VariantLit(ident,variant_ident,expr) => Err("TODO: enum variant literal".to_string()),
            Expr::MethodCall(expr,ident,exprs) => Err("TODO: method call".to_string()),
            Expr::Field(expr,ident) => Err("TODO: struct field selector".to_string()),
            Expr::TupleIndex(expr,index) => Err("TODO: tuple index (struct field selector)".to_string()),
            Expr::AnonTupleLit(index,exprs) => Err("TODO: anon tuple literal (struct literal)".to_string()),
            Expr::LocalRefOrParamRef(ident) => Ok(ident.to_string()),
            Expr::ConstRef(ident) => Ok(ident.to_string()),
            Expr::FunctionCall(ident,exprs) => Err("TODO: function call".to_string()),
            Expr::TupleLit(ident,exprs) => Err("TODO: tuple literal".to_string()),
            Expr::EnumDiscr(expr,index) => Err("TODO: enum discriminant".to_string()),
            Expr::EnumArg(expr,variant_index,index) => Err("TODO: enum variant argument".to_string()),
            Expr::Constructor(type_,fields) => Err("TODO: constructor".to_string()),
        }
    }

    fn emit_stat(&self,stat: &Stat) -> Result<String,String> {
        match stat {
            Stat::Let(_,_,_) => Err("let-statement cannot occur here".to_string()),
            Stat::Expr(expr) => self.emit_expr(expr),
            Stat::Local(ident,type_,expr) => Ok(format!("{} {} = {};",self.emit_type(type_)?,ident,self.emit_expr(expr)?)),
        }
    }

    fn emit_block(&self,block: &Block) -> Result<String,String> {
        let mut result = String::new();
        result += "{\n";
        for stat in block.stats.iter() {
            result += &format!("{}\n",self.emit_stat(stat)?);
        }
        if let Some(expr) = &block.expr {
            result += &format!("return {};\n",self.emit_expr(&expr)?);
        }
        result += "}\n";
        Ok(result)
    }

    fn emit_function(&self,function: &Function) -> Result<String,String> {
        let mut result = String::new();
        if function.ident == "main" {
            result += "void main() {\n";
            for stat in function.block.stats.iter() {
                result += &format!("{}\n",self.emit_stat(stat)?);
            }
            if let Some(expr) = &function.block.expr {
                result += &format!("TODO: gl_Something = {};\n",self.emit_expr(&expr)?);
            }
            result += "}\n";    
        }
        else {
            result += &format!("{} {}(",self.emit_type(&function.return_type)?,function.ident);
            let mut first = true;
            for param in function.params.iter() {
                if !first {
                    result += ",";
                }
                result += &format!("{} {}",self.emit_type(&param.1)?,param.0);
                first = false;
            }
            result += ") ";
        }
        result += &self.emit_block(&function.block)?;
        Ok(result)
    }
}

pub fn emit_module(module: &ProcessedModule,style: ShaderStyle) -> Result<String,String> {

    let context = Context {
        stdlib: StandardLib::new(),
        module: module.clone(),
    };

    let main = if let Some(function) = module.functions.iter().find(|function| function.ident == "main") {
        function
    }
    else {
        return Err("main function not found".to_string());
    };

    let mut inputs: Vec<String> = Vec::new();
    for param in main.params.iter() {
        match param.1 {
            Type::Inferred => return Err(format!("cannot infer type for parameter {} of main function",param.0)),
            Type::Void => return Err(format!("void parameter {} invalid in main function",param.0)),
            Type::AnonTuple(_) => return Err(format!("anonymous tuple cannot occur as parameter {} in main function",param.0)),
            Type::Array(_,_) => return Err(format!("invalid array parameter {} in main function",param.0)),
            Type::Ident(ident) => return Err(format!("unknown type {} as parameter {} of main function",ident,param.0)),
            Type::AnonTupleRef(index) => {
                let types = &module.anon_tuple_types[index];
                for type_ in types.iter() {
                    inputs.push(context.emit_type(type_)?);
                }
            },
            Type::StructRef(ident) => {
                let struct_ = {
                    let found_struct = module.structs.iter().find(|struct_| struct_.ident == ident);
                    if let Some(struct_) = found_struct {
                        struct_
                    }
                    else {
                        let found_struct = module.extern_structs.iter().find(|struct_| struct_.ident == ident);
                        if let Some(struct_) = found_struct {
                            struct_
                        }
                        else {
                            let found_struct = context.stdlib.structs.iter().find(|struct_| struct_.ident == ident);
                            if let Some(struct_) = found_struct {
                                struct_
                            }
                            else {
                                return Err(format!("unknown struct {} for parameter in main function",ident));
                            }
                        }
                    }
                };
                for field in struct_.fields.iter() {
                    inputs.push(context.emit_type(&field.1)?);
                }
            },
            Type::TupleRef(ident) => {
                let tuple = {
                    let found_tuple = module.tuples.iter().find(|tuple| tuple.ident == ident);
                    if let Some(tuple) = found_tuple {
                        tuple
                    }
                    else {
                        let found_tuple = context.stdlib.tuples.iter().find(|tuple| tuple.ident == ident);
                        if let Some(tuple) = found_tuple {
                            tuple
                        }
                        else {
                            return Err(format!("unknown tuple {} for parameter in main function",ident));
                        }
                    }
                };
                for type_ in tuple.types.iter() {
                    inputs.push(context.emit_type(type_)?);
                }
            },
            Type::EnumRef(ident) => {
                let tuple = {
                    let found_tuple = module.enum_tuples.iter().find(|tuple| tuple.ident == ident);
                    if let Some(tuple) = found_tuple {
                        tuple
                    }
                    else {
                        return Err(format!("unknown enum {} for parameter in main function",ident));
                    }
                };
                for type_ in tuple.types.iter() {
                    inputs.push(context.emit_type(type_)?);
                }
            },
            _ => inputs.push(context.emit_type(&param.1)?),
        }
    }

    let mut outputs: Vec<String> = Vec::new();
    match main.return_type {
        Type::Inferred => return Err("cannot infer type for result main function".to_string()),
        Type::Void => { },
        Type::AnonTuple(_) => return Err("anonymous tuple cannot occur as result of main function".to_string()),
        Type::Array(_,_) => return Err("array cannot occur as result of main function".to_string()),
        Type::Ident(ident) => return Err(format!("unknown type {} as result of of main function",ident)),
        Type::AnonTupleRef(index) => {
            let types = &module.anon_tuple_types[index];
            for type_ in types.iter() {
                outputs.push(context.emit_type(type_)?);
            }
        },
        Type::StructRef(ident) => {
            let struct_ = {
                let found_struct = module.structs.iter().find(|struct_| struct_.ident == ident);
                if let Some(struct_) = found_struct {
                    struct_
                }
                else {
                    let found_struct = module.extern_structs.iter().find(|struct_| struct_.ident == ident);
                    if let Some(struct_) = found_struct {
                        struct_
                    }
                    else {
                        let found_struct = context.stdlib.structs.iter().find(|struct_| struct_.ident == ident);
                        if let Some(struct_) = found_struct {
                            struct_
                        }
                        else {
                            return Err(format!("unknown struct {} as result of main function",ident));
                        }
                    }
                }
            };
            for field in struct_.fields.iter() {
                outputs.push(context.emit_type(&field.1)?);
            }
        },
        Type::TupleRef(ident) => {
            let tuple = {
                let found_tuple = module.tuples.iter().find(|tuple| tuple.ident == ident);
                if let Some(tuple) = found_tuple {
                    tuple
                }
                else {
                    let found_tuple = context.stdlib.tuples.iter().find(|tuple| tuple.ident == ident);
                    if let Some(tuple) = found_tuple {
                        tuple
                    }
                    else {
                        return Err(format!("unknown tuple {} as result of main function",ident));
                    }
                }
            };
            for type_ in tuple.types.iter() {
                outputs.push(context.emit_type(type_)?);
            }
        },
        Type::EnumRef(ident) => {
            let tuple = {
                let found_tuple = module.enum_tuples.iter().find(|tuple| tuple.ident == ident);
                if let Some(tuple) = found_tuple {
                    tuple
                }
                else {
                    return Err(format!("unknown enum {} as result of main function",ident));
                }
            };
            for type_ in tuple.types.iter() {
                outputs.push(context.emit_type(type_)?);
            }
        },
        _ => outputs.push(context.emit_type(&main.return_type)?),
    }

    match style {
        ShaderStyle::Vertex => {
            let mut result = String::new();
            result += "#version 450\n\n";
            for i in 0..inputs.len() {
                result += &format!("layout(location = {}) in {} input{};\n",i,inputs[i],i);
            }
            for i in 1..outputs.len() {
                result += &format!("layout(location = {}) out {} output{};\n",i - 1,outputs[i],i - 1);
            }
            for function in module.functions.iter() {
                result += &context.emit_function(function)?;
            }
            Ok(result)
        },
        ShaderStyle::Fragment => {
            Err("TODO: fragment shader".to_string())
        },
    }
}
