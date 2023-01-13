use {
    super::*,
    std::collections::HashMap,
};

use ast::*;

struct Context {
    structs: HashMap<String,Struct>,
    enums: HashMap<String,Enum>,
    enum_variants: HashMap<String,HashMap<String,usize>>,
    enum_indices: HashMap<String,Vec<Vec<usize>>>,
    consts: HashMap<String,Const>,
    functions: HashMap<String,Function>,
    stdlib: StandardLib,
    locals: HashMap<String,Symbol>,
    params: HashMap<String,Symbol>,
    anon_tuple_structs: HashMap<String,Struct>,
}

impl Context {

    fn get_expr_type(&self,expr: &Expr) -> Type {
        match expr {
            Expr::Boolean(_) => Type::Bool,
            Expr::Integer(_) => Type::Integer,
            Expr::Float(_) => Type::Float,
            Expr::Array(exprs) => Type::Array(Box::new(self.get_expr_type(&exprs[0])),Box::new(Expr::Integer(exprs.len() as i64))),
            Expr::Cloned(expr,expr2) => Type::Array(Box::new(self.get_expr_type(&expr)),expr2.clone()),
            Expr::Index(expr,expr2) => if let Type::Array(type_,_) = self.get_expr_type(&expr) { (*type_).clone() } else { panic!("array index on non-array {}",expr2); },
            Expr::Cast(_,type_) => (**type_).clone(),
            Expr::AnonTuple(exprs) => {
                let mut new_types: Vec<Type> = Vec::new();
                for expr in exprs.iter() {
                    new_types.push(self.get_expr_type(expr));
                }
                Type::AnonTuple(new_types)
            },
            Expr::Unary(_,expr) => self.get_expr_type(expr),
            Expr::Binary(expr,op,_) => {
                match op {
                    BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Greater | BinaryOp::GreaterEq | BinaryOp::Less | BinaryOp::LessEq => Type::Bool,
                    _ => self.get_expr_type(expr),
                }
            },
            Expr::Continue => Type::Void,
            Expr::Break(expr) => if let Some(expr) = expr { self.get_expr_type(expr) } else { Type::Void },
            Expr::Return(expr) => if let Some(expr) = expr { self.get_expr_type(expr) } else { Type::Void },
            Expr::Block(block) => if let Some(expr) = &block.expr { self.get_expr_type(&*expr) } else { Type::Void },
            Expr::If(_,block,else_expr) => if let Some(else_expr) = else_expr {
                if let Some(expr) = &block.expr {
                    Self::tightest(&self.get_expr_type(&expr),&self.get_expr_type(&else_expr)).expect("if-block and else-block should have the same type")
                }
                else {
                    if Type::Void != self.get_expr_type(&else_expr) {
                        panic!("else-block should not return anything");
                    }
                    Type::Void
                }
            }
            else {
                if let Some(expr) = &block.expr {
                    self.get_expr_type(&expr)
                }
                else {
                    Type::Void
                }
            },
            Expr::While(_,_) => Type::Void,
            Expr::Loop(_) => Type::Void,
            Expr::Struct(ident,_) => Type::Struct(ident.clone()),
            Expr::Method(expr,ident,_) => {
                if self.stdlib.methods.contains_key(ident) {
                    let methods = &self.stdlib.methods[ident];
                    let from_type = self.get_expr_type(expr);
                    let mut found: Option<&Method> = None;
                    for method in methods.iter() {
                        if let Some(_) = Self::tightest(&method.from_type,&from_type) {
                            found = Some(method);
                            break; 
                        }
                    }
                    if let Some(method) = found {
                        method.type_.clone()
                    }
                    else {
                        panic!("unknown method {} for {}",ident,expr);
                    }
                }
                else {
                    panic!("unknown method {}",ident);
                }
            },
            Expr::Field(expr,ident) => {
                if let Type::Struct(struct_ident) = self.get_expr_type(expr) {
                    let struct_ = if self.structs.contains_key(&struct_ident) {
                        &self.structs[&struct_ident]
                    }
                    else if self.stdlib.structs.contains_key(&struct_ident) {
                        &self.stdlib.structs[&struct_ident]
                    }
                    else {
                        panic!("unknown struct {}",struct_ident);
                    };
                    let mut found: Option<Type> = None;
                    for field in struct_.fields.iter() {
                        if &field.ident == ident {
                            found = Some(field.type_.clone());
                            break;
                        }
                    }
                    found.expect(&format!("unknown field {} on {}",ident,expr))
                }
                else {
                    panic!("{} is not a struct",expr);
                }
            },
            Expr::Param(ident) => if self.params.contains_key(ident) {
                self.params[ident].type_.clone()
            }
            else {
                panic!("unknown parameter {}",ident);
            },
            Expr::Local(ident) => if self.locals.contains_key(ident) {
                self.locals[ident].type_.clone()
            }
            else {
                panic!("unknown local {}",ident);
            },
            Expr::Const(ident) => if self.consts.contains_key(ident) {
                self.consts[ident].type_.clone()
            }
            else if self.stdlib.consts.contains_key(ident) {
                self.stdlib.consts[ident].type_.clone()
            }
            else {
                panic!("unknown constant {}",ident);
            },
            Expr::Call(ident,exprs) => {
                if self.functions.contains_key(ident) {
                    self.functions[ident].type_.clone()
                }
                else if self.stdlib.functions.contains_key(ident) {
                    let functions = &self.stdlib.functions[ident];
                    let mut found: Option<&Function> = None;
                    for function in functions.iter() {
                        if exprs.len() == function.params.len() {
                            let mut all_params_match = true;
                            for i in 0..exprs.len() {
                                if function.params[i].type_ != self.get_expr_type(&exprs[i]) {
                                    all_params_match = false;
                                    break;
                                }
                            }
                            if all_params_match {
                                found = Some(function);
                                break;
                            }
                        }
                        if let Some(_) = found {
                            break;
                        }
                    }
                    if let Some(function) = found {
                        function.type_.clone()
                    }
                    else {
                        panic!("function {} not found for these parameters",ident);
                    }
                }
                else {
                    panic!("unknown function {}",ident);
                }
            },
            Expr::Discriminant(_,_) => panic!("attempting to get type of Expr::Discriminant"),
            Expr::DestructTuple(_,_,_) => panic!("attempting to get type from Expr::DestructTuple"),
            Expr::DestructStruct(_,_,_) => panic!("attempting to get type from Expr::DestructStruct"),
            _ => panic!("cannot get type of {} at this stage",expr),
        }
    }

    fn tightest(type1: &Type,type2: &Type) -> Option<Type> {
        match (type1,type2) {
            (Type::Alias(_),_) | (_,Type::Alias(_)) => None,
            (Type::UnknownIdent(_),_) | (_,Type::UnknownIdent(_)) => None,
            (Type::Tuple(_),_) | (_,Type::Tuple(_)) => None,
            (Type::Enum(_),_) | (_,Type::Enum(_)) => None,
            (Type::Inferred,_) => Some(type2.clone()),
            (_,Type::Inferred) => Some(type1.clone()),
            (Type::Integer,_) => match type2 {
                Type::Float => Some(Type::Float),
                Type::U8 => Some(Type::U8),
                Type::I8 => Some(Type::I8),
                Type::U16 => Some(Type::U16),
                Type::I16 => Some(Type::I16),
                Type::U32 => Some(Type::U32),
                Type::I32 => Some(Type::I32),
                Type::U64 => Some(Type::U64),
                Type::I64 => Some(Type::I64),
                Type::USize => Some(Type::USize),
                Type::ISize => Some(Type::ISize),
                Type::F16 => Some(Type::F16),
                Type::F32 => Some(Type::F32),
                Type::F64 => Some(Type::F64),
                _ => None,
            },
            (_,Type::Integer) => match type1 {
                Type::Float => Some(Type::Float),
                Type::U8 => Some(Type::U8),
                Type::I8 => Some(Type::I8),
                Type::U16 => Some(Type::U16),
                Type::I16 => Some(Type::I16),
                Type::U32 => Some(Type::U32),
                Type::I32 => Some(Type::I32),
                Type::U64 => Some(Type::U64),
                Type::I64 => Some(Type::I64),
                Type::USize => Some(Type::USize),
                Type::ISize => Some(Type::ISize),
                Type::F16 => Some(Type::F16),
                Type::F32 => Some(Type::F32),
                Type::F64 => Some(Type::F64),
                _ => None,
            },
            (Type::Float,_) => match type2 {
                Type::F16 => Some(Type::F16),
                Type::F32 => Some(Type::F32),
                Type::F64 => Some(Type::F64),
                _ => None,
            },
            (_,Type::Float) => match type1 {
                Type::F16 => Some(Type::F16),
                Type::F32 => Some(Type::F32),
                Type::F64 => Some(Type::F64),
                _ => None,
            },
            _ => if type1 == type2 {
                Some(type1.clone())
            }
            else {
                None
            },
        }
    }

    fn get_anon_tuple_struct(&mut self,types: &Vec<Type>) -> String {
        for struct_ in self.anon_tuple_structs.values() {
            if struct_.fields.len() == types.len() {
                let mut found: Option<String> = Some(struct_.ident.clone());
                for i in 0..types.len() {
                    if struct_.fields[i].type_ != types[i] {
                        found = None;
                        break;
                    }
                }
                if let Some(ident) = found {
                    return ident;
                }
            }
        }
        let mut new_fields: Vec<Symbol> = Vec::new();
        for i in 0..types.len() {
            new_fields.push(Symbol { ident: format!("_{}",i), type_: types[i].clone(), });
        }
        let ident = format!("AnonTuple{}",self.anon_tuple_structs.len());
        self.anon_tuple_structs.insert(ident.clone(),Struct { ident: ident.clone(),fields: new_fields, });
        ident
    }
        
    fn process_type(&mut self,type_: &Type,should_type: &Type) -> Option<Type> {
        match type_ {
            Type::AnonTuple(types) => {
                if let Type::AnonTuple(should_types) = should_type {
                    let mut new_types: Vec<Type> = Vec::new();
                    if types.len() == should_types.len() {
                        for i in 0..types.len() {
                            if let Some(type_) = Self::tightest(&types[i],&should_types[i]) {
                                new_types.push(type_);
                            }
                            else {
                                return None;
                            }
                        }
                        Some(Type::Struct(self.get_anon_tuple_struct(&new_types)))
                    }
                    else {
                        None
                    }
                }
                else {
                    let mut new_types: Vec<Type> = Vec::new();
                    for type_ in types.iter() {
                        new_types.push(type_.clone());
                    }
                    Some(Type::Struct(self.get_anon_tuple_struct(&new_types)))
                }
            },
            Type::Array(type_,expr) => if let Type::Array(should_type,_) = should_type {
                let new_type = if let Some(type_) = self.process_type(type_,should_type) { type_ } else { return None; };
                let new_expr = self.process_expr(expr,&Type::Integer);
                Some(Type::Array(Box::new(new_type),Box::new(new_expr)))                
            }
            else if let Type::Inferred = should_type {
                let new_expr = self.process_expr(expr,&Type::Integer);
                Some(Type::Array(Box::new((**type_).clone()),Box::new(new_expr)))
            }
            else {
                None
            },
            Type::Struct(ident) => if let Type::Struct(_) = should_type {
                Some(Type::Struct(ident.clone()))
            }
            else if let Type::Inferred = should_type {
                Some(Type::Struct(ident.clone()))
            }
            else {
                None
            },
            _ => if let Some(type_) = Self::tightest(type_,should_type) {
                Some(type_)
            }
            else {
                panic!("illegal type at this stage: {}",type_);
            },
        }        
    }

    fn process_expr(&mut self,expr: &Expr,should_type: &Type) -> Expr {
        // returns type with processed anonymous tuples, unless it doesn't fit should_type
        match expr {
            Expr::Boolean(value) => Expr::Boolean(*value),
            Expr::Integer(value) => Expr::Integer(*value),
            Expr::Float(value) => Expr::Float(*value),
            Expr::Array(exprs) => {
                if let Type::Array(should_type,_) = should_type {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for expr in exprs {
                        new_exprs.push(self.process_expr(expr,should_type));
                    }
                    Expr::Array(new_exprs)
                }
                else if let Type::Inferred = should_type {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for expr in exprs {
                        new_exprs.push(self.process_expr(expr,&Type::Inferred));
                    }
                    Expr::Array(new_exprs)
                }
                else {
                    panic!("type mismatch (array found, {} expected)",should_type);
                }
            },
            Expr::Cloned(expr,expr2) => {
                if let Type::Array(should_type,_) = should_type {
                    let new_expr = self.process_expr(expr,should_type);
                    let new_expr2 = self.process_expr(expr2,&Type::Integer);
                    Expr::Cloned(Box::new(new_expr),Box::new(new_expr2))
                }
                else if let Type::Inferred = should_type {
                    let new_expr = self.process_expr(expr,&Type::Inferred);
                    let new_expr2 = self.process_expr(expr2,&Type::Integer);
                    Expr::Cloned(Box::new(new_expr),Box::new(new_expr2))
                }
                else {
                    panic!("type mismatch (array found, {} expected)",should_type);
                }
            },
            Expr::Index(expr,expr2) => {
                let new_expr = self.process_expr(expr,&Type::Inferred);  // TODO: figure out how to extract array size
                let new_expr2 = self.process_expr(expr2,&Type::Integer);
                Expr::Index(Box::new(new_expr),Box::new(new_expr2))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.process_expr(expr,&Type::Inferred);
                let new_type = self.process_type(type_,should_type).expect(&format!("incompatible cast types ({} should be {})",type_,should_type));
                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },
            Expr::AnonTuple(exprs) => {
                if let Type::AnonTuple(should_types) = should_type {
                    let mut new_fields: Vec<(String,Expr)> = Vec::new();
                    for i in 0..exprs.len() {
                        new_fields.push((format!("_{}",i),self.process_expr(&exprs[i],&should_types[i])));
                    }
                    Expr::Struct(self.get_anon_tuple_struct(should_types),new_fields)
                }
                else {
                    panic!("type mismatch (anonymous tuple found, {} expected)",should_type);
                }
            },
            Expr::Unary(op,expr) => {
                let new_expr = self.process_expr(expr,should_type);
                Expr::Unary(
                    op.clone(),
                    Box::new(new_expr)
                )
            },
            Expr::Binary(expr,op,expr2) => {
                match op {
                    BinaryOp::Mul |
                    BinaryOp::Div |
                    BinaryOp::Mod |
                    BinaryOp::Add |
                    BinaryOp::Sub |
                    BinaryOp::Shl |
                    BinaryOp::Shr |
                    BinaryOp::And |
                    BinaryOp::Or |
                    BinaryOp::Xor |
                    BinaryOp::Assign |
                    BinaryOp::AddAssign |
                    BinaryOp::SubAssign |
                    BinaryOp::MulAssign |
                    BinaryOp::DivAssign |
                    BinaryOp::ModAssign |
                    BinaryOp::AndAssign |
                    BinaryOp::OrAssign |
                    BinaryOp::XorAssign |
                    BinaryOp::ShlAssign |
                    BinaryOp::ShrAssign => {
                        let new_expr = self.process_expr(expr,should_type);
                        let new_expr2 = self.process_expr(expr2,should_type);
                        Expr::Binary(Box::new(new_expr),op.clone(),Box::new(new_expr2))
                    },
                    BinaryOp::Eq |
                    BinaryOp::NotEq |
                    BinaryOp::Greater |
                    BinaryOp::Less |
                    BinaryOp::GreaterEq |
                    BinaryOp::LessEq |
                    BinaryOp::LogAnd |
                    BinaryOp::LogOr => {
                        let new_expr = self.process_expr(expr,&Type::Inferred);
                        let new_expr2 = self.process_expr(expr2,&Type::Inferred);
                        Expr::Binary(Box::new(new_expr),op.clone(),Box::new(new_expr2))
                    },
                }
            },
            Expr::Continue => Expr::Continue,
            Expr::Break(expr) => if let Some(expr) = expr {
                Expr::Break(Some(Box::new(self.process_expr(expr,should_type))))
            }
            else {
                Expr::Break(None)
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                Expr::Return(Some(Box::new(self.process_expr(expr,should_type))))
            }
            else {
                Expr::Return(None)
            },
            Expr::Block(block) => Expr::Block(self.process_block(block,should_type)),
            Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr {
                Expr::If(
                    Box::new(self.process_expr(expr,&Type::Bool)),
                    self.process_block(block,should_type),
                    Some(Box::new(self.process_expr(else_expr,should_type)))
                )
            }
            else {
                Expr::If(
                    Box::new(self.process_expr(expr,&Type::Bool)),
                    self.process_block(block,should_type),
                    None
                )
            },
            Expr::While(expr,block) => Expr::While(
                Box::new(self.process_expr(expr,&Type::Bool)),
                self.process_block(block,&Type::Inferred),
            ),
            Expr::Loop(block) => Expr::Loop(self.process_block(block,&Type::Inferred)),
            Expr::Struct(ident,fields) => {
                let struct_ = if self.structs.contains_key(ident) {
                    self.structs[ident].clone()
                }
                else if self.anon_tuple_structs.contains_key(ident) {
                    self.anon_tuple_structs[ident].clone()
                }
                else if self.stdlib.structs.contains_key(ident) {
                    self.stdlib.structs[ident].clone()
                }
                else {
                    panic!("unknown struct {}",ident);
                };
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for i in 0..fields.len() {
                    let new_expr = self.process_expr(&fields[i].1,&struct_.fields[i].type_);
                    new_fields.push((fields[i].0.clone(),new_expr));
                }
                Expr::Struct(ident.clone(),new_fields)    
            },
            Expr::Method(expr,ident,exprs) => {
                if self.stdlib.methods.contains_key(ident) {
                    let mut found: Option<Method> = None;
                    for method in self.stdlib.methods[ident].iter() {
                        if let None = Self::tightest(&method.type_,should_type) {
                            continue;
                        }
                        if let None = Self::tightest(&self.get_expr_type(expr),&method.from_type) {
                            continue;
                        }
                        if method.params.len() == exprs.len() {
                            let mut all_params_match = true;
                            for i in 0..exprs.len() {
                                if let None = Self::tightest(&self.get_expr_type(&exprs[i]),&method.params[i].type_) {
                                    all_params_match = false;
                                    break;
                                }
                            }
                            if all_params_match {
                                found = Some(method.clone());
                            }
                        }
                        if let Some(_) = found {
                            break;
                        }
                    }
                    if let Some(method) = found {
                        let new_expr = self.process_expr(expr,&method.from_type);
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for i in 0..exprs.len() {
                            new_exprs.push(self.process_expr(&exprs[i],&method.params[i].type_));
                        }
                        Expr::Method(Box::new(new_expr),method.ident,new_exprs)
                    }
                    else {
                        panic!("method {} not found for {}",ident,expr);
                    }
                }
                else {
                    panic!("method {} not found for {}",ident,expr);
                }
            },
            Expr::Field(expr,ident) => {
                if let Type::Struct(struct_ident) = self.get_expr_type(expr) {
                    let struct_ = if self.structs.contains_key(&struct_ident) {
                        self.structs[&struct_ident].clone()
                    }
                    else if self.stdlib.structs.contains_key(&struct_ident) {
                        self.stdlib.structs[&struct_ident].clone()
                    }
                    else {
                        panic!("unknown struct {}",struct_ident);
                    };
                    let mut found: Option<Expr> = None;
                    for field in struct_.fields.iter() {
                        if field.ident == *ident {
                            let new_expr = self.process_expr(expr,&field.type_);
                            found = Some(Expr::Field(Box::new(new_expr),ident.clone()));
                            break;
                        }
                    }
                    found.expect(&format!("unknown field {} on struct {}",ident,struct_.ident))
                }
                else {
                    panic!("{} not a struct",expr);
                }
            },
            Expr::Param(ident) => Expr::Param(ident.clone()),
            Expr::Local(ident) => Expr::Local(ident.clone()),
            Expr::Const(ident) => Expr::Const(ident.clone()),
            Expr::Call(ident,exprs) => {
                let function = if self.functions.contains_key(ident) {
                    self.functions[ident].clone()
                }
                else if self.stdlib.functions.contains_key(ident) {
                    let mut found: Option<&Function> = None;
                    for function in self.stdlib.functions[ident].iter() {
                        if exprs.len() == function.params.len() {
                            let mut all_params_match = true;
                            for i in 0..exprs.len() {
                                if let None = Self::tightest(&self.get_expr_type(&exprs[i]),&function.params[i].type_) {
                                    all_params_match = false;
                                    break;
                                }
                            }
                            if all_params_match {
                                found = Some(function);
                                break;
                            }
                        }
                    }
                    if let Some(function) = found {
                        function.clone()
                    }
                    else {
                        panic!("no matching function {} found",ident);
                    }
                }
                else {
                    panic!("unknown function {}",ident);
                };
                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..exprs.len() {
                    new_exprs.push(self.process_expr(&exprs[i],&function.params[i].type_));
                }
                Expr::Call(ident.clone(),new_exprs)
            },
            Expr::Discriminant(expr,variant_ident) => {
                if let Type::Struct(enum_ident) = self.get_expr_type(expr) {
                    let new_expr = self.process_expr(expr,&Type::Struct(enum_ident.clone()));
                    let variants = if self.enum_variants.contains_key(&enum_ident) {
                        self.enum_variants[&enum_ident].clone()
                    }
                    else if self.stdlib.enum_variants.contains_key(&enum_ident) {
                        self.stdlib.enum_variants[&enum_ident].clone()
                    }
                    else {
                        panic!("unknown enum {}",enum_ident);
                    };
                    let variant_index = variants[variant_ident];
                    Expr::Binary(Box::new(Expr::Field(Box::new(new_expr),"discr".to_string())),BinaryOp::Eq,Box::new(Expr::Integer(variant_index as i64)))
                }
                else {
                    panic!("{} is not an enum struct",expr);
                }
            },
            Expr::DestructTuple(expr,variant_ident,index) => {
                if let Type::Struct(enum_ident) = self.get_expr_type(expr) {
                    let new_expr = self.process_expr(expr,&Type::Struct(enum_ident.clone()));
                    let variants = if self.enum_variants.contains_key(&enum_ident) {
                        self.enum_variants[&enum_ident].clone()
                    }
                    else if self.stdlib.enum_variants.contains_key(&enum_ident) {
                        self.stdlib.enum_variants[&enum_ident].clone()
                    }
                    else {
                        panic!("unknown enum {}",enum_ident);
                    };
                    let indices = if self.enum_indices.contains_key(&enum_ident) {
                        self.enum_indices[&enum_ident].clone()
                    }
                    else if self.stdlib.enum_indices.contains_key(&enum_ident) {
                        self.stdlib.enum_indices[&enum_ident].clone()
                    }
                    else {
                        panic!("unknown variant {} of enum {}",variant_ident,enum_ident);
                    };
                    let variant_index = variants[variant_ident];
                    Expr::Field(Box::new(new_expr),format!("_{}",indices[variant_index][*index]))
                }
                else {
                    panic!("{} is not an enum struct",expr);
                }
            },
            Expr::DestructStruct(expr,variant_ident,ident) => {
                if let Type::Struct(enum_ident) = self.get_expr_type(expr) {
                    let new_expr = self.process_expr(expr,&Type::Struct(enum_ident.clone()));
                    let variants = if self.enum_variants.contains_key(&enum_ident) {
                        self.enum_variants[&enum_ident].clone()
                    }
                    else if self.stdlib.enum_variants.contains_key(&enum_ident) {
                        self.stdlib.enum_variants[&enum_ident].clone()
                    }
                    else {
                        panic!("unknown enum {}",enum_ident);
                    };
                    let indices = if self.enum_indices.contains_key(&enum_ident) {
                        self.enum_indices[&enum_ident].clone()
                    }
                    else if self.stdlib.enum_indices.contains_key(&enum_ident) {
                        self.stdlib.enum_indices[&enum_ident].clone()
                    }
                    else {
                        panic!("unknown variant {} of enum {}",variant_ident,enum_ident);
                    };
                    let variant_index = variants[variant_ident];
                    let enum_ = self.enums[&enum_ident].clone();
                    if let Variant::Struct(_,fields) = &enum_.variants[variant_index] {
                        let mut found: Option<Expr> = None;
                        for i in 0..fields.len() {
                            if fields[i].ident == *ident {
                                found = Some(Expr::Field(Box::new(new_expr),format!("_{}",indices[variant_index][i])));
                                break;
                            }
                        }
                        if let Some(expr) = found {
                            expr
                        }
                        else {
                            panic!("unknown field {} in variant {} of enum {}",ident,variant_ident,enum_ident);
                        }
                    }
                    else {
                        panic!("unknown variant {} of enum {}",variant_ident,enum_ident);
                    }
                }
                else {
                    panic!("{} is not an enum struct",expr);
                }
            },
            _ => panic!("illegal expression at this stage: {}",expr),
        }
    }

    fn process_block(&mut self,block: &Block,should_type: &Type) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            // TODO: build new_stats
            new_stats.push(match stat {
                Stat::Expr(expr) => {
                    let new_expr = self.process_expr(expr,&Type::Inferred);
                    Stat::Expr(Box::new(new_expr))
                },
                Stat::Local(ident,type_,expr) => {
                    let new_type = self.process_type(type_,&Type::Inferred).expect(&format!("Type not allowed {}",type_));
                    let new_expr = self.process_expr(expr,&new_type);
                    Stat::Local(ident.clone(),Box::new(new_type),Box::new(new_expr))
                },
                _ => panic!("Stat::Let cannot exist at this stage"),
            });
        }
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.process_expr(&expr,should_type)))
        }
        else {
            None
        };
        Block {
            stats: new_stats,
            expr: new_expr,
        }
    }

    fn process_module(module: ConvertedModule) -> Module {

        let mut context = Context {
            structs: module.structs.clone(),
            enums: module.enums.clone(),
            enum_variants: module.enum_variants.clone(),
            enum_indices: module.enum_indices.clone(),
            consts: module.consts.clone(),
            functions: module.functions.clone(),
            stdlib: StandardLib::new(),
            locals: HashMap::new(),
            params: HashMap::new(),
            anon_tuple_structs: HashMap::new(),
        };

        let mut new_structs: HashMap<String,Struct> = HashMap::new();
        for struct_ in module.structs.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(&field.type_,&Type::Inferred).expect(&format!("unknown type {}",field.type_)),
                });
            }
            new_structs.insert(
                struct_.ident.clone(),
                Struct {
                    ident: struct_.ident.clone(),
                    fields: new_fields,
                }
            );
        }
        
        let mut new_consts: HashMap<String,Const> = HashMap::new();
        for const_ in module.consts.values() {
            let new_type = context.process_type(&const_.type_,&Type::Inferred).expect(&format!("unknown type {}",const_.type_));
            let new_expr = context.process_expr(&const_.expr,&const_.type_);
            new_consts.insert(const_.ident.clone(),Const { ident: const_.ident.clone(),type_: new_type,expr: new_expr, });
        }

        let mut new_functions: HashMap<String,Function> = HashMap::new();
        for function in module.functions.values() {
            let mut new_params: Vec<Symbol> = Vec::new();
            context.params.clear();
            for param in function.params.iter() {
                let new_param = Symbol { ident: param.ident.clone(),type_: context.process_type(&param.type_,&Type::Inferred).expect(&format!("unknown type {}",param.type_)), };
                new_params.push(new_param.clone());
                context.params.insert(param.ident.clone(),new_param);
            }
            let new_type = context.process_type(&function.type_,&Type::Inferred).expect(&format!("unknown type {}",function.type_));
            let new_block = context.process_block(&function.block,&new_type);
            new_functions.insert(function.ident.clone(),Function { ident: function.ident.clone(),params: new_params,type_: new_type,block: new_block, });
        }

        Module {
            ident: module.ident.clone(),
            structs: new_structs,
            consts: new_consts,
            functions: new_functions,
        }
    }
}

pub fn resolve_module(module: ConvertedModule) -> Module {
    Context::process_module(module)
}