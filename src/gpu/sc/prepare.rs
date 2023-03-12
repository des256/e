use {
    super::*,
    super::ast::*,
    std::{
        collections::HashMap,
        cmp::PartialEq,
    },
};

// - convert aliases to their types
// - convert anonymous tuples

impl PartialEq for Type {
    fn eq(&self,other: &Type) -> bool {
        match (self,other) {
            (Type::Inferred,_) => true,
            (_,Type::Inferred) => true,
            (Type::Void,Type::Void) => true,
            (Type::Bool,Type::Bool) => true,
            (Type::U8,Type::U8) => true,
            (Type::I8,Type::I8) => true,
            (Type::U16,Type::U16) => true,
            (Type::I16,Type::I16) => true,
            (Type::U32,Type::U32) => true,
            (Type::I32,Type::I32) => true,
            (Type::U64,Type::U64) => true,
            (Type::I64,Type::I64) => true,
            (Type::F16,Type::F16) => true,
            (Type::F32,Type::F32) => true,
            (Type::F64,Type::F64) => true,
            (Type::Vec2Bool,Type::Vec2Bool) => true,
            (Type::Vec2U8,Type::Vec2U8) => true,
            (Type::Vec2I8,Type::Vec2I8) => true,
            (Type::Vec2U16,Type::Vec2U16) => true,
            (Type::Vec2I16,Type::Vec2I16) => true,
            (Type::Vec2U32,Type::Vec2U32) => true,
            (Type::Vec2I32,Type::Vec2I32) => true,
            (Type::Vec2U64,Type::Vec2U64) => true,
            (Type::Vec2I64,Type::Vec2I64) => true,
            (Type::Vec2F16,Type::Vec2F16) => true,
            (Type::Vec2F32,Type::Vec2F32) => true,
            (Type::Vec2F64,Type::Vec2F64) => true,
            (Type::Vec3Bool,Type::Vec3Bool) => true,
            (Type::Vec3U8,Type::Vec3U8) => true,
            (Type::Vec3I8,Type::Vec3I8) => true,
            (Type::Vec3U16,Type::Vec3U16) => true,
            (Type::Vec3I16,Type::Vec3I16) => true,
            (Type::Vec3U32,Type::Vec3U32) => true,
            (Type::Vec3I32,Type::Vec3I32) => true,
            (Type::Vec3U64,Type::Vec3U64) => true,
            (Type::Vec3I64,Type::Vec3I64) => true,
            (Type::Vec3F16,Type::Vec3F16) => true,
            (Type::Vec3F32,Type::Vec3F32) => true,
            (Type::Vec3F64,Type::Vec3F64) => true,
            (Type::Vec4Bool,Type::Vec4Bool) => true,
            (Type::Vec4U8,Type::Vec4U8) => true,
            (Type::Vec4I8,Type::Vec4I8) => true,
            (Type::Vec4U16,Type::Vec4U16) => true,
            (Type::Vec4I16,Type::Vec4I16) => true,
            (Type::Vec4U32,Type::Vec4U32) => true,
            (Type::Vec4I32,Type::Vec4I32) => true,
            (Type::Vec4U64,Type::Vec4U64) => true,
            (Type::Vec4I64,Type::Vec4I64) => true,
            (Type::Vec4F16,Type::Vec4F16) => true,
            (Type::Vec4F32,Type::Vec4F32) => true,
            (Type::Vec4F64,Type::Vec4F64) => true,
            (Type::Mat2x2F32,Type::Mat2x2F32) => true,
            (Type::Mat2x2F64,Type::Mat2x2F64) => true,
            (Type::Mat2x3F32,Type::Mat2x3F32) => true,
            (Type::Mat2x3F64,Type::Mat2x3F64) => true,
            (Type::Mat2x4F32,Type::Mat2x4F32) => true,
            (Type::Mat2x4F64,Type::Mat2x4F64) => true,
            (Type::Mat3x2F32,Type::Mat3x2F32) => true,
            (Type::Mat3x2F64,Type::Mat3x2F64) => true,
            (Type::Mat3x3F32,Type::Mat3x3F32) => true,
            (Type::Mat3x3F64,Type::Mat3x3F64) => true,
            (Type::Mat3x4F32,Type::Mat3x4F32) => true,
            (Type::Mat3x4F64,Type::Mat3x4F64) => true,
            (Type::Mat4x2F32,Type::Mat4x2F32) => true,
            (Type::Mat4x2F64,Type::Mat4x2F64) => true,
            (Type::Mat4x3F32,Type::Mat4x3F32) => true,
            (Type::Mat4x3F64,Type::Mat4x3F64) => true,
            (Type::Mat4x4F32,Type::Mat4x4F32) => true,
            (Type::Mat4x4F64,Type::Mat4x4F64) => true,
            (Type::AnonTuple(types1),Type::AnonTuple(types2)) => {
                if types1.len() == types2.len() {
                    for i in 0..types1.len() {
                        if types1[i] != types2[i] {
                            return false;
                        }
                    }
                    true
                }
                else {
                    false
                }
            },
            (Type::Array(type1,count1),Type::Array(type2,count2)) => (type1 == type2) && (count1 == count2),
            (Type::Ident(ident1),Type::Ident(ident2)) => ident1 == ident2,
            _ => false,
        }
    }
}

impl Context {

    fn prepare_type(&self,type_: Type) -> Result<Type,String> {

        match type_ {
            
            // convert anonymous tuple type into a struct that can be referenced by unique index
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(self.prepare_type(*type_)?);
                }
                let found_index: Option<usize> = None;
                for i in 0..self.anon_tuple_types.len() {
                    if self.anon_tuple_types[i].len() == types.len() {
                        let mut all_types_match = true;
                        for k in 0..types.len() {
                            if self.anon_tuple_types[i][k] != new_types[k] {
                                all_types_match = false;
                                break;
                            }
                        }
                        if all_types_match {
                            found_index = Some(i);
                            break;
                        }
                    }
                }
                if let Some(index) = found_index {
                    Ok(Type::AnonTupleRef(index))
                }
                else {
                    self.anon_tuple_types.push(new_types);
                    Ok(Type::AnonTupleRef(self.anon_tuple_types.len() - 1))
                }
            },

            // recursively prepare type
            Type::Array(type_,count) => {
                let new_type = self.prepare_type(*type_)?;
                Ok(Type::Array(Box::new(new_type),count))
            },

            // if this refers to an alias, resolve to the final type
            Type::Ident(ident) => {
                if self.alias_types.contains_key(&ident) {
                    Ok(self.alias_types[&ident].clone())
                }
                else {
                    Ok(Type::Ident(ident))
                }
            },

            // everything else is fine as it is
            _ => Ok(type_),
        }
    }

    fn prepare_block(&self,block: Block,expected_type: Type) -> Result<Block,String> {

        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {

                // recurse into let statement, no need to fix anything further just yet
                Stat::Let(pat,type_,expr) => {
                    let new_type = self.prepare_type(*type_.clone())?;
                    let new_expr = self.prepare_expr(*expr.clone(),new_type.clone())?;
                    new_stats.push(Stat::Let(pat,Box::new(new_type),Box::new(new_expr)));
                },

                // recurse into expression, keep type inferred here and ignore the result
                Stat::Expr(expr) => {
                    let new_expr = self.prepare_expr(*expr.clone(),Type::Inferred)?;
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },

                // local variable cannot occur here
                Stat::Local(ident,type_,expr) => {
                    // parser does not generate Stat::Local nodes
                    return Err("illegal Stat::Local".to_string());
                },
            }
        }

        // recurse into return expression, expecting expected_type as result
        let new_expr = if let Some(expr) = block.expr {
            Some(Box::new(self.prepare_expr(*expr,expected_type)?))
        }
        else {
            None
        };

        Ok(Block {
            stats: new_stats,
            expr: new_expr,
        })
    }

    fn prepare_expr(&self,expr: Expr,expected_type: Type) -> Result<Expr,String> {
        match expr {

            // only allow boolean literal if boolean or inferred are expected
            Expr::Boolean(value) => match expected_type {
                Type::Inferred | Type::Bool => Ok(Expr::Boolean(value)),
                _ => Err(format!("{} expected instead of boolean literal",expected_type)),
            },

            // only allow integer literal if integer, float or inferred are expected
            Expr::Integer(value) => match expected_type {
                Type::Inferred | Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 => Ok(Expr::Integer(value)),
                _ => Err(format!("{} expected instead of integer literal",expected_type)),
            },

            // only allow float literal if float or inferred are expected
            Expr::Float(value) => match expected_type {
                Type::Inferred | Type::F16 | Type::F32 | Type::F64 => Ok(Expr::Float(value)),
                _ => Err(format!("{} expected instead of float literal",expected_type)),
            },

            Expr::Array(exprs) => match expected_type {

                // just recurse into array when expected_type is inferred
                Type::Inferred => {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for expr in exprs {
                        new_exprs.push(self.prepare_expr(expr,Type::Inferred)?);
                    }
                    Ok(Expr::Array(new_exprs))
                },

                // match exact types while recursing into array when expected_type is array
                Type::Array(expected_type,expected_count) => {
                    if exprs.len() == expected_count {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for i in 0..expected_count {
                            new_exprs.push(self.prepare_expr(expr,*expected_type)?);
                        }
                        Ok(Expr::Array(new_exprs))
                    }
                    else {
                        Err(format!("{} elements expected, {} found",expected_count,exprs.len()))
                    }
                },

                _ => Err(format!("{} expected instead of array literal",expected_type)),
            },

            Expr::Cloned(value_expr,count) => match expected_type {

                // just recurse into expression when expected_type is inferred
                Type::Inferred => {
                    let new_value_expr = self.prepare_expr(*value_expr,Type::Inferred)?;
                    Ok(Expr::Cloned(Box::new(new_value_expr),count))
                },

                // match exact types while recursing into array when expected_type is array
                Type::Array(expected_type,expected_count) => {
                    if count == expected_count {
                        let new_value_expr = self.prepare_expr(*value_expr,*expected_type)?;
                        Ok(Expr::Cloned(Box::new(new_value_expr),count))
                    }
                    else {
                        Err(format!("{} elements expected, {} found",expected_count,count))
                    }
                },

                _ => Err(format!("{} expected instead of array clone literal",expected_type)),
            },

            Expr::Index(array_expr,index_expr) => {
                // infer for now
                let new_array_expr = self.prepare_expr(*array_expr,Type::Inferred)?;
                let new_index_expr = self.prepare_expr(*index_expr,Type::Inferred)?;
                Ok(Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr)))
            },

            Expr::Cast(expr,type_) => {
                let new_type = self.prepare_type(*type_)?;
                if let Type::Inferred = expected_type { } else {
                    if new_type != expected_type {
                        return Err(format!("{} expected instead of {}",expected_type,new_type));
                    }
                }
                let new_expr = self.prepare_expr(*expr,new_type)?;
                Ok(Expr::Cast(Box::new(new_expr),Box::new(new_type)))
            },

            Expr::AnonTuple(exprs) => if let Type::AnonTupleRef(index) = expected_type {
                if self.anon_tuple_types[index].len() == exprs.len() {
                    let new_exprs: Vec<Expr> = Vec::new();
                    for i in 0..exprs.len() {
                        new_exprs.push(self.prepare_expr(exprs[i],self.anon_tuple_types[index][i])?);
                    }
                    Ok(Expr::AnonTupleRef(index,new_exprs))
                }
                else {
                    Err(format!("{} and {} have different dimensions",expr,expected_type))
                }
            }
            else {
                Err(format!("{} expected instead of {}",expected_type,expr))
            },

            Expr::Unary(op,expr) => {
                // infer for now
                let new_expr = self.prepare_expr(*expr,Type::Inferred)?;
                Ok(Expr::Unary(op,Box::new(new_expr)))
            },

            Expr::Binary(expr1,op,expr2) => {
                // infer for now
                let new_expr1 = self.prepare_expr(*expr1,Type::Inferred)?;
                let new_expr2 = self.prepare_expr(*expr2,Type::Inferred)?;
                Ok(Expr::Binary(Box::new(new_expr1),op,Box::new(new_expr2)))
            },

            Expr::Continue => Ok(Expr::Continue),

            Expr::Break(expr) => if let Some(expr) = expr {
                Ok(Expr::Break(Some(Box::new(self.prepare_expr(*expr,Type::Inferred)?))))
            }
            else {
                Ok(Expr::Break(None))
            },

            Expr::Return(expr) => if let Some(expr) = expr {
                Ok(Expr::Return(Some(Box::new(self.prepare_expr(*expr,Type::Inferred)?))))
            }
            else {
                Ok(Expr::Return(None))
            },

            Expr::Block(block) => Ok(Expr::Block(self.prepare_block(block,expected_type)?)),

            Expr::If(cond_expr,block,else_expr) => {
                // recurse, and make sure block and else_expr have the expected type
                let new_cond_expr = self.prepare_expr(*cond_expr,Type::Bool)?;
                let new_block = self.prepare_block(block,expected_type)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.prepare_expr(*else_expr,expected_type)?))
                }
                else {
                    None
                };
                Ok(Expr::If(Box::new(new_cond_expr),new_block,new_else_expr))
            },

            Expr::While(cond_expr,block) => {
                // recurse, make sure cond_expr is boolean
                let new_cond_expr = self.prepare_expr(*cond_expr,Type::Bool)?;
                let new_block = self.prepare_block(block,Type::Inferred)?;
                Ok(Expr::While(Box::new(new_cond_expr),new_block))
            },

            Expr::Loop(block) => Ok(Expr::Loop(self.prepare_block(block,Type::Inferred)?)),

            Expr::IfLet(pats,expr,block,else_expr) => {
                // recurse, and make sure block and else_expr have the expected type
                let new_expr = self.prepare_expr(expr,Type::Inferred)?;
                let new_block = self.prepare_block(block,expected_type)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(self.prepare_expr(else_expr,expected_type)?)
                }
                else {
                    None
                };
                Ok(Expr::IfLet(pats,Box::new(new_expr),new_block,new_else_expr))
            },
        
            Expr::For(pats,range,block) => {
                // recurse
                let new_range = self.prepare_range(range)?;
                let new_block = self.prepare_block(block,Type::Inferred)?;
                Ok(Expr::For(pats,new_range,new_block))
            },

            Expr::WhileLet(pats,expr,block) => {
                // recurse
                let new_expr = self.prepare_expr(expr,Type::Inferred)?;
                let new_block = self.prepare_block(block,Type::Inferred)?;
                Ok(Expr::WhileLet(pats,Box::new(new_expr),new_block))
            },

            Expr::Match(expr,arms) => {
                // recurse, make sure all arms have the expected type
                let new_expr = self.prepare_expr(expr,Type::Inferred)?;
                let mut new_arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
                for (pats,if_expr,expr) in arms.iter() {
                    let new_if_expr = if let Some(if_expr) = if_expr {
                        Some(self.prepare_expr(if_expr,Type::Bool)?)
                    }
                    else {
                        None
                    };
                    let new_expr = self.prepare_expr(expr,expected_type)?;
                    new_arms.push((pats,new_if_expr,new_expr))
                }
                Ok(Expr::Match(new_expr,new_arms))
            },

            Expr::Ident(ident) => {
                if self.consts.contains_key(ident) {
                    // it's a constant reference
                    if let Type::Inferred = expected_type {
                        Ok(Expr::ConstRef(ident))
                    }
                    else {
                        // make sure constant has the expected type
                        let const_ = &self.consts[ident];
                        if const_.type_ == expected_type {
                            Ok(Expr::ConstRef(ident))
                        }
                        else {
                            Err(format!("{} expected instead of {}",expected_type,expr))
                        }
                    }
                }
                else {
                    // it's a local or parameter reference, keep for after destructuring
                    Ok(Expr::LocalOrParamRef(ident))
                }
            },

            Expr::TupleOrFunction(ident,exprs) => {

                if self.functions.contains_key(ident) || self.stdlib.functions.contains_key(ident) {
                    let function = if self.functions.contains_key(ident) { &self.functions[ident] } else { &self.stdlib.functions[ident] };

                    // it's a function call, make sure the parameters and types line up
                    if exprs.len() != function.params.len() {
                        return Err(format!("{} parameters expected instead of {}",function.params.len(),exprs.len()));
                    }
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for i in 0..exprs.len() {
                        new_exprs.push(self.prepare_expr(exprs[i],function.params[i].1)?);
                    }
                    if let Type::Inferred = expected_type {
                        // just infer
                        Ok(Expr::Function(ident,new_exprs))
                    } else {
                        // make sure the return value is the expected type
                        if function.return_type != expected_type {
                            return Err(format!("{} expected instead of {}",expected_type,expr))
                        }
                        Ok(Expr::Function(ident,new_exprs))
                    }
                }

                else if self.tuple_types.contains_key(ident) {
                    let types = &self.tuple_types[ident];

                    // it's a tuple reference, make sure the components line up
                    if exprs.len() != types.len() {
                        return Err(format!("{} components expected instead of {}",types.len(),exprs.len()));
                    }
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for i in 0..exprs.len() {
                        new_exprs.push(self.prepare_expr(exprs[i],types[i])?);
                    }
                    if let Type::Inferred = expected_type {
                        // just infer
                        Ok(Expr::TupleRef(ident,new_exprs))
                    }
                    else {
                        if let Type::TupleRef(expected_ident) = expected_type {
                            if ident == expected_ident {
                                Ok(Expr::TupleRef(ident,new_exprs))
                            }
                            else {
                                Err(format!("{} expected instead of {}",expected_ident,ident))
                            }
                        }
                        else {
                            Err(format!("{} expected instead of {}",expected_type,ident))
                        }
                    }
                }
            },

            Expr::Struct(ident,fields) => if self.structs.contains_key(ident) || self.stdlib.structs.contains_key(ident) || self.extern_structs.contains_key(ident) {
                let struct_ = if self.structs.contains_key(ident) { &self.structs[ident] } else if self.stdlib.structs.contains_key(ident) { &self.stdlib.structs[ident] } else { &self.extern_structs[ident] };

                // it's a struct reference, make sure the fields line up
                if fields.len() != struct_.fields.len() {
                    return Err(format!("{} fields expected instead of {}",struct_.fields.len(),fields.len()));
                }
                let mut new_fields: Vec<(&'static str,Expr)> = Vec::new();
                for i in 0..fields.len() {
                    new_fields.push((fields[i].0,self.prepare_expr(fields[i].1,struct_.fields[i].1)));
                }
                if let Type::Inferred = expected_type {
                    // just infer
                    Ok(Expr::StructRef(ident,new_fields))
                }
                else {
                    if let Type::StructRef(expected_ident) = expected_type {
                        if ident == expected_ident {
                            Ok(Expr::StructRef(ident,new_fields))
                        }
                        else {
                            Err(format!("{} expected instead of {}",expected_ident,ident))
                        }
                    }
                    else {
                        Err(format!("{} expected instead of {}",expected_type,ident))
                    }
                }
            }
            else {
                Err(format!("unknown struct {}",ident))
            },

            Expr::Variant(enum_ident,variant_ident,variant_expr) => if self.enums.contains_key(enum_ident) {
                let enum_ = &self.enums[enum_ident];

                // it's an enum reference, find the corresponding variant
                let mut found_variant: Option<Variant> = None;
                for variant in enum_.variants.iter() {
                    if variant.0 == variant_ident {
                        found_variant = Some(variant.1);
                    }
                }
                if let None = found_variant {
                    return Err(format!("enum {} has no variant {}",enum_ident,variant_ident));
                }
                let variant = found_variant.unwrap();
                match (variant,variant_expr) {
                    (Variant::Naked,VariantExpr::Naked) => (),
                    (Variant::Tuple(types),VariantExpr::Tuple(exprs)) => {
                        if exprs.len() == types.len() {
                            let mut new_exprs: Vec<Expr> = Vec::new();
                            for i in 0..exprs.len() {
                                new_exprs.push(self.prepare_expr(exprs[i],types[i])?);
                            }
                            Ok(Expr::Variant(enum_ident,variant_ident,VariantExpr::Tuple(new_exprs)))
                        }
                        else {
                            Err(format!("{} components expected instead of {}",types.len(),exprs.len()))
                        }
                    },
                    (Variant::Struct(field_types),VariantExpr::Struct(field_exprs)) => {
                        if field_exprs.len() == field_types.len() {
                            let mut new_field_exprs: Vec<(&'static str,Expr)> = Vec::new();
                            for i in 0..field_exprs.len() {
                                if field_exprs[i].0 == field_types[i].0 {
                                    new_field_exprs.push((field_exprs[i].0,self.prepare_expr(field_exprs[i].1,field_types[i].1)?));
                                }
                                else {
                                    return Err(format!("{} field not found in {}::{}",field_exprs[i].0,enum_ident,variant_ident))
                                }
                            }
                            Ok(Expr::Variant(enum_ident,variant_ident,VariantExpr::Struct(new_field_exprs)))
                        }
                        else {
                            Err(format!("{} fields expected instead of {}",field_types.len(),field_exprs.len()))
                        }
                    },
                    (Variant::Naked,_) => Err(format!("{}::{} has no tuple components or struct fields",enum_ident,variant_ident)),
                    (Variant::Tuple(_),_) => Err(format!("{}::{}(...) requires tuple components",enum_ident,variant_ident)),
                    (Variant::Struct(_),_) => Err(format!("{}::{} {{ ... }} requires struct fields",enum_ident,variant_ident)),
                }
            }
            else {
                Err(format!("unknown enum {}",enum_ident))
            }

            Expr::Method(expr,ident,exprs) => if self.stdlib.methods.contains_key(ident) {
                // TODO: figure out type of expr

                let method = &self.stdlib.methods[ident];

                // it's a method call, make sure the parameters and types line up
                if exprs.len() != method.params.len() {
                    return Err(format!("{} parameters expected instead of {}",method.params.len(),exprs.len()));
                }
                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..exprs.len() {
                    new_exprs.push(self.prepare_expr(exprs[i],method.params[i].1)?);
                }
                if let Type::Inferred = expected_type {
                    // just infer
                    Ok(Expr::Method(ident,new_exprs))
                } else {
                    // make sure the return value is the expected type
                    if method.return_type != expected_type {
                        return Err(format!("{} expected instead of {}",expected_type,expr))
                    }
                    Ok(Expr::Method(ident,new_exprs))
                }
            }
            else {
                Err(format!("unknown method {}",ident))
            },

            Expr::Field(expr,ident) => {
                // TODO: figure out type of expr
                Ok(Expr::Field(self.prepare_expr(expr,Type::Inferred),ident))
            },

            Expr::TupleIndex(expr,index) => {
                // TODO: figure out type of expr
                Ok(Expr::TupleIndex(self.prepare_expr(expr,Type::Inferred),index))
            },

            Expr::TupleRef(ident,exprs) => Ok(Expr::TupleRef(ident,exprs)),

            Expr::FunctionRef(ident,exprs) => Ok(Expr::FunctionRef(ident,exprs)),

            Expr::StructRef(ident,exprs) => Ok(Expr::StructRef(ident,exprs)),

            Expr::AnonTupleRef(index,exprs) => Ok(Expr::AnonTupleRef(index,exprs)),

            Expr::ConstRef(ident) => Ok(Expr::ConstRef(ident)),

            Expr::LocalOrParamRef(ident) => Ok(Expr::LocalOrParamRef(ident)),
        }
    }
}

pub fn prepare_module(module: &Module) -> Result<Module,String> {

    // initialize the context   
    let mut context = Context {
        stdlib: StandardLib::new(),
        alias_types: HashMap::new(),
        consts: HashMap::new(),
        structs: HashMap::new(),
        enums: HashMap::new(),
        functions: HashMap::new(),
    };

    // simplify all aliases into their proper types
    for alias in module.aliases.values() {
        let mut type_ = alias.type_.clone();
        while let Type::Ident(ident) = type_ {
            if module.aliases.contains_key(ident) {
                type_ = module.aliases[ident].type_.clone();               
            }
            else {
                break;
            }
        }
        context.alias_types.insert(alias.ident,type_);
    }

    // process const
    for const_ in module.consts.values() {
        let new_type = context.process_type(const_.type_)?;
        let new_expr = context.process_expr(const_.expr)?;
        context.consts.insert(const_.ident,Const {
            ident: const_.ident,
            type_: new_type,
            expr: new_expr,
        });
    }

    Ok(Module {
        ident: module.ident,
        consts: context.consts,
        structs: context.structs,
        extern_structs: context.extern_structs,
        enums: context.enums,
        functions: context.functions,
        aliases: HashMap::new(),
    })
}