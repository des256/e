use super::*;

impl Resolver {

    fn get_anon_tuple_struct(&mut self,type_: &Type,new_types: &Vec<Type>) -> Type {

        // find the matching anon_tuple_struct
        let mut found_ident: Option<String> = None;
        for (ident,struct_) in self.module.anon_tuple_structs.iter() {

            // if the lengths match
            if new_types.len() == struct_.fields.len() {

                // check if all types are the same                
                let mut all_types_match = true;
                for i in 0..new_types.len() {
                    // TODO: better PartialEq implementation on ast::Type
                    if format!("{}",new_types[i]) != format!("{}",struct_.fields[i].1) {
                        all_types_match = false;
                        break;
                    }
                }

                // if so, we found a match
                if all_types_match {
                    found_ident = Some(ident.clone());
                    break;
                }
            }
        }

        // if a match was found, refer to it
        if let Some(ident) = found_ident {
            Type::Struct(ident)
        }

        // otherwise add a new anon_tuple_struct with these types
        else {
            let ident = format!("anon{:05}",self.module.anon_tuple_structs.len());
            let mut fields: Vec<(String,Type)> = Vec::new();
            for i in 0..new_types.len() {
                fields.push((format!("f{}",i),new_types[i].clone()));
            }
            self.module.anon_tuple_structs.insert(ident.clone(),Struct {
                ident: ident.clone(),
                fields,
            });

            self.log_change(format!("converted anonymous tuple {} to anon_tuple_struct {}",type_,ident));

            Type::Struct(ident)
        }
    }

    // resolve type when an expected type is already known
    pub fn resolve_should_type(&mut self,type_: &Type,should_type: &Type) -> Type {
        match type_ {
            
            Type::Inferred => {
                self.log_change(format!("type {} inferred",should_type));
                should_type.clone()
            },

            Type::Void => type_.clone(),

            Type::Integer => match type_ {
                Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 => type_.clone(),
                _ => panic!("cannot set {} to integer literal",type_),
            }

            Type::Float => match type_ {
                Type::F16 | Type::F32 | Type::F64 => type_.clone(),
                _ => panic!("cannot set {} to float literal",type_),
            }

            Type::Bool => if let Type::Bool = should_type { Type::Bool } else { panic!("{} expected instead of bool",should_type); },
            Type::U8 => if let Type::U8 = should_type { Type::U8 } else { panic!("{} expected instead of u8",should_type); },
            Type::I8 => if let Type::I8 = should_type { Type::I8 } else { panic!("{} expected instead of i8",should_type); },
            Type::U16 => if let Type::U16 = should_type { Type::U16 } else { panic!("{} expected instead of u16",should_type); },
            Type::I16 => if let Type::I16 = should_type { Type::I16 } else { panic!("{} expected instead of i16",should_type); },
            Type::U32 => if let Type::U32 = should_type { Type::U32 } else { panic!("{} expected instead of u32",should_type); },
            Type::I32 => if let Type::I32 = should_type { Type::I32 } else { panic!("{} expected instead of i32",should_type); },
            Type::U64 => if let Type::U64 = should_type { Type::U64 } else { panic!("{} expected instead of u64",should_type); },
            Type::I64 => if let Type::I64 = should_type { Type::I64 } else { panic!("{} expected instead of i64",should_type); },
            Type::F16 => if let Type::F16 = should_type { Type::F16 } else { panic!("{} expected instead of f16",should_type); },
            Type::F32 => if let Type::F32 = should_type { Type::F32 } else { panic!("{} expected instead of f32",should_type); },
            Type::F64 => if let Type::F64 = should_type { Type::F64 } else { panic!("{} expected instead of f64",should_type); },

            Type::AnonTuple(types) => if let Type::AnonTuple(should_types) = should_type.clone() {

                if types.len() != should_types.len() {
                    panic!("anonymous tuple {} expected instead of anonymous tuple {}",should_type,type_);
                }

                self.push_context(format!("anonymous tuple {}",type_));

                let mut new_types: Vec<Type> = Vec::new();
                for i in 0..types.len() {
                    new_types.push(self.resolve_should_type(&types[i],&should_types[i]));
                }

                self.pop_context();

                self.get_anon_tuple_struct(type_,&new_types)
            }
            else {
                panic!("{} expected instead of anonymous tuple {}",should_type,type_);
            },

            Type::Array(type_,expr) => if let Type::Array(should_type,should_expr) = should_type {

                self.push_context(format!("array [{}; {}]",type_,expr));

                let new_type = self.resolve_should_type(type_,should_type);
                let new_expr = self.resolve_should_expr(expr,should_type);

                self.pop_context();

                Type::Array(Box::new(new_type),Box::new(new_expr))
            }
            else {
                panic!("{} expected instead of {}",should_type,type_);
            },

            Type::UnknownStructTupleEnumAlias(ident) => match should_type {
                Type::Struct(struct_ident) => if ident == struct_ident {

                    self.log_change(format!("resolved {} to struct reference",ident));

                    Type::Struct(struct_ident.clone())
                }
                else {
                    panic!("struct {} expected instead of {}",struct_ident,ident);
                },
                Type::Tuple(tuple_ident) => if ident == tuple_ident {

                    self.log_change(format!("resolved {} to tuple struct reference",ident));

                    Type::Struct(ident.clone())
                }
                else {
                    panic!("tuple {} expected instead of {}",tuple_ident,ident)
                },
                Type::Enum(enum_ident) => if ident == enum_ident {

                    self.log_change(format!("resolved {} to enum reference",ident));

                    // TODO: resolve enum to struct

                    Type::Enum(enum_ident.clone())
                }
                else {
                    panic!("enum {} expected instead of {}",enum_ident,ident)
                },
                Type::Alias(alias_ident) => if ident == alias_ident {

                    // TODO: follow chain until final type

                    Type::Alias(alias_ident.clone())
                }
                else {
                    panic!("{} expected instead of {}",alias_ident,ident);
                },
                _ => {
                    panic!("{} expected instead of {}",should_type,type_);
                }
            },

            Type::Struct(ident) => if let Type::Struct(struct_ident) = should_type {
                Type::Struct(ident.clone())
            }
            else {
                panic!("{} expected instead of struct {}",should_type,ident);
            },

            Type::Tuple(ident) => if let Type::Struct(struct_ident) = should_type {

                self.log_change(format!("resolved tuple {} to struct reference",ident));

                Type::Struct(ident.clone())
            }
            else {
                panic!("{} expected instead of tuple {}",should_type,ident);
            },

            Type::Enum(ident) => if let Type::Enum(enum_ident) = should_type {

                // TODO: resolve enum to struct

                Type::Enum(ident.clone())
            }
            else {
                panic!("{} expected instead of enum {}",should_type,ident);
            },

            Type::Alias(ident) => if let Type::Alias(alias_ident) = should_type {

                // TODO: follow chain until final type

                Type::Alias(ident.clone())
            }
            else {
                panic!("{} expected instead of alias {}",should_type,ident);
            },
        }
    }

    // resolve type without knowledge of what's expected
    pub fn resolve_type(&mut self,type_: &Type) -> Type {
        match type_ {
             
            Type::Inferred | Type::Void | Type::Integer | Type::Float => type_.clone(),
            Type::Bool | Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 => type_.clone(),

            Type::AnonTuple(types) => {

                self.push_context(format!("anonymous tuple {}",type_));

                let mut new_types: Vec<Type> = Vec::new();
                for i in 0..types.len() {
                    new_types.push(self.resolve_type(&types[i]));
                }

                self.pop_context();

                self.get_anon_tuple_struct(&type_,&new_types)
            },

            Type::Array(type_,expr) => {

                self.push_context(format!("array [{}; {}]",type_,expr));

                let new_type = self.resolve_type(&*type_);
                let new_expr = self.resolve_expr(&*expr);

                self.pop_context();

                Type::Array(Box::new(new_type),Box::new(new_expr))
            },

            Type::UnknownStructTupleEnumAlias(ident) => {

                if self.module.structs.contains_key(ident) {

                    self.log_change(format!("resolved {} to struct reference",ident));

                    Type::Struct(ident.clone())
                }

                else if self.module.extern_structs.contains_key(ident) {

                    self.log_change(format!("resolved {} to external struct reference",ident));

                    Type::Struct(ident.clone())
                }

                else if self.module.tuple_structs.contains_key(ident) {

                    self.log_change(format!("resolved {} to tuple struct reference",ident));

                    Type::Struct(ident.clone())
                }

                else if self.module.enums.contains_key(ident) {

                    self.log_change(format!("resolved {} to enum reference",ident));

                    Type::Enum(ident.clone())
                }

                else if self.module.aliases.contains_key(ident) {

                    self.log_change(format!("resolved {} to alias reference",ident));

                    Type::Alias(ident.clone())
                }

                else {
                    // maybe fix later
                    type_.clone()
                }
            },

            Type::Struct(ident) => Type::Struct(ident.clone()),

            Type::Tuple(ident) => {

                self.log_change(format!("resolved {} to tuple struct reference",ident));

                Type::Struct(ident.clone())
            },

            Type::Enum(ident) => {

                // TODO: resolve enum to struct
                type_.clone()
            },

            Type::Alias(ident) => {

                // TODO: find matching alias and replace
                type_.clone()
            },
        }
    }
}
