use {
    super::*,
    std::collections::HashMap,
};

use std::f32::consts::E;

use ast::*;

struct Context {
    structs: HashMap<String,Struct>,
    enums: HashMap<String,Enum>,
    enum_indices: HashMap<String,Vec<Vec<usize>>>,
    consts: HashMap<String,Const>,
    functions: HashMap<String,Function>,
    stdlib: StandardLib,
    locals: HashMap<String,Symbol>,
    params: HashMap<String,Symbol>,
    anon_tuple_structs: HashMap<String,Struct>,
}

impl Context {

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

    fn get_anon_tuple_struct(&self,types: &Vec<Type>) -> String {
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
        // returns type with processed anonymous tuples, unless it doesn't fit should_type
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
                    None
                }
            },
            Type::Array(type_,expr) => if let Type::Array(should_type,should_expr) = should_type {
                let new_type = if let Some(type_) = self.process_type(type_,should_type) { type_ } else { return None; };
                let new_expr = self.process_expr(expr,&Type::Integer);
                Some(Type::Array(Box::new(new_type),Box::new(new_expr)))                
            }
            else {
                None
            },
            Type::Struct(ident) => if let Type::Struct(ident) = should_type { Some(Type::Struct(ident.clone())) } else { None },
            _ => if let Some(type_) = Self::tightest(type_,should_type) { Some(type_) } else { panic!("illegal type at this stage: {}",type_); },
        }        
    }

    fn process_expr(&self,expr: &Expr,should_type: &Type) -> Expr {
        // returns type with processed anonymous tuples, unless it doesn't fit should_type
        match expr {
            Expr::Boolean(value) => Expr::Boolean(*value),
            Expr::Integer(value) => Expr::Integer(*value),
            Expr::Float(value) => Expr::Float(*value),
            Expr::Array(exprs) => {
                if let Type::Array(should_type,should_expr) = should_type {
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
                if let Type::Array(should_type,should_expr) = should_type {
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
                    self.structs[ident]
                }
                else if self.anon_tuple_structs.contains_key(ident) {
                    self.structs[ident]
                }
                else {
                    panic!("unknown struct {}",ident);
                };
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for i in 0..fields.len() {
                    let new_expr = self.process_expr(&fields[i].1,&struct_.fields[i].type_);
                    new_fields.push((fields[i].0,new_expr));
                }
                Expr::Struct(ident.clone(),new_fields)    
            },
            Expr::Method(expr,ident,exprs) => {
                if self.stdlib.methods.contains_key(ident) {
                    let mut found: Option<&Method> = None;
                    for method in self.stdlib.methods[ident].iter() {
                        // TODO: match expr's type with method.from_type
                        // TODO: match method.type_ with should_type
                        if method.params.len() == exprs.len() {
                            let mut all_params_fit = true;
                            for i in 0..exprs.len() {
                                // TODO: match exprs[i]'s type with method.params[i].type_
                            }
                            if all_params_fit {
                                found = Some(method);
                            }
                        }
                        if let Some(_) = found {
                            break;
                        }
                    }
                    if let Some(method) = found {
                        let mut new_expr = self.process_expr(expr,&method.from_type);
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
                // TODO: get expr's type, tighten with should_type
                let new_expr = self.process_expr(expr);
                Expr::Field(Box::new(new_expr),ident.clone())
            },
            Expr::Param(ident) => Expr::Param(ident.clone()),
            Expr::Local(ident) => Expr::Local(ident.clone()),
            Expr::Const(ident) => Expr::Const(ident.clone()),
            Expr::Call(ident,exprs) => {
                // TODO: find function in module or stdlib
                // TODO: match function.type_ and should_type
                // TODO: convert exprs matching the function params
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Call(ident.clone(),new_exprs)
            },
            Expr::Discriminant(expr,variant_ident) => {
                // TODO: convert to Expr::Field(expr,"discr") == Expr::Integer
                let new_expr = self.process_expr(expr);
                Expr::Discriminant(Box::new(new_expr),variant_ident.clone())
            },
            Expr::DestructTuple(expr,variant_ident,index) => {
                // TODO: convert to Expr::Field
                let new_expr = self.process_expr(expr);
                Expr::DestructTuple(Box::new(new_expr),variant_ident.clone(),*index)
            },
            Expr::DestructStruct(expr,variant_ident,ident) => {
                // TODO: convert to Expr::Field
                let new_expr = self.process_expr(expr);
                Expr::DestructStruct(Box::new(new_expr),variant_ident.clone(),ident.clone())
            },
            _ => panic!("illegal expression at this stage: {}",expr),
        }
    }

    fn process_block(&self,block: &Block,should_type: &Type) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {
                Stat::Expr(expr) => { },
                Stat::Local(ident,type_,expr) => { },
                _ => panic!("Stat::Let cannot exist at this stage"),
            }
        }
        let expr = if let Some(expr) = block.expr {
            Some()
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
                    type_: context.process_type(&field.type_),
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
            let new_type = context.process_type(&const_.type_);
            let new_expr = context.process_expr(&const_.expr,&const_.type_);
            new_consts.insert(const_.ident.clone(),Const { ident: const_.ident.clone(),type_: new_type,expr: new_expr, });
        }

        let mut new_functions: HashMap<String,Function> = HashMap::new();
        for function in module.functions.values() {
            let mut new_params: Vec<Symbol> = Vec::new();
            context.params.clear();
            for param in function.params.iter() {
                let new_param = Symbol { ident: param.ident.clone(),type_: context.process_type(&param.type_), };
                new_params.push(new_param.clone());
                context.params.insert(param.ident.clone(),new_param);
            }
            let mut new_type = context.process_type(&function.type_);
            let mut new_block = context.process_block(&function.block,&new_type);
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