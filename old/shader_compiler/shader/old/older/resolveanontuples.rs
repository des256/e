// for each block statement/expression, figure out the type of the expression

use {
    crate::*,
    std::{
        rc::Rc,
        collections::HashMap,
    },
};

struct Resolver { }

impl Resolver {

    pub fn tightest(&self,type1: &sr::Type,type2: &sr::Type) -> Option<sr::Type> {

        // if both types are identical, take one of them
        if type1 == type2 {
            Some(type1.clone())
        }
        
        // if one is sr::Type::Inferred, take the other one
        else if let sr::Type::Inferred = type1 {
            Some(type2.clone())
        }
        else if let sr::Type::Inferred = type2 {
            Some(type1.clone())
        }

        // if one is sr::Type::Integer, upcast to sr::Type::Float or sr::Type::Base
        else if let sr::Type::Integer = type1 {
            match type2 {
                sr::Type::Float => Some(sr::Type::Float),
                sr::Type::Base(bt) => match bt {
                    sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                    sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                    _ => None,
                },
                _ => None,
            }
        }
        else if let sr::Type::Integer = type2 {
            match type1 {
                sr::Type::Float => Some(sr::Type::Float),
                sr::Type::Base(bt) => match bt {
                    sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                    sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                    _ => None,
                },
                _ => None,
            }
        }

        // if one is sr::Type::Float, upcast to sr::Type::Base
        else if let sr::Type::Float = type1 {
            if let sr::Type::Base(bt) = type2 {
                match bt {
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                    _ => None,
                }
            }
            else {
                None
            }
        }
        else if let sr::Type::Float = type2 {
            if let sr::Type::Base(bt) = type1 {
                match bt {
                    sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(sr::Type::Base(bt.clone())),
                    _ => None,
                }
            }
            else {
                None
            }
        }

        // anything else is not compatible
        else {
            None
        }
    }

    pub fn block(&self,block: &mut sr::Block,expected_type: &sr::Type) -> sr::Type {
        if let Some(expr) = block.expr.as_mut() {
            self.expr(expr,expected_type)
        }
        else if let sr::Type::Inferred = expected_type {
            sr::Type::Void
        }
        else {
            panic!("block should return {}",expected_type);
        }
    }

    pub fn expr(&self,expr: &mut sr::Expr,expected_type: &sr::Type) -> sr::Type {
        match expr {
            sr::Expr::UnknownIdent(ident) => panic!("ASSERT ERROR: unknown identifier {}",ident),
            sr::Expr::UnknownStruct(ident,_) => panic!("ASSERT ERROR: unknown struct {}",ident),
            sr::Expr::UnknownCallOrTuple(ident,_) => panic!("ASSERT ERROR: unknown function call or tuple {}",ident),

            sr::Expr::Boolean(value) => self.tightest(&sr::Type::Boolean,expected_type).expect(&format!("{} expected instead of boolean {}",expected_type,if *value { "true" } else { "false" })),
            sr::Expr::Integer(value) => self.tightest(&sr::Type::Integer,expected_type).expect(&format!("{} expected instead of integer {}",expected_type,*value)),
            sr::Expr::Float(value) => self.tightest(&sr::Type::Float,expected_type).expect(&format!("{} expected instead of float {}",expected_type,*value)),
            sr::Expr::Base(bt,_) => self.tightest(&sr::Type::Base(bt.clone()),expected_type).expect(&format!("{} expected instead of {}",expected_type,bt.to_rust())),
            sr::Expr::Const(const_) => self.tightest(&const_.type_,expected_type).expect(&format!("{} expected instead of constant {}",expected_type,const_.type_)),
            sr::Expr::Local(local) => self.tightest(&local.type_,expected_type).expect(&format!("{} expected instead of local {}",expected_type,local.type_)),
            sr::Expr::Param(param) => self.tightest(&param.type_,expected_type).expect(&format!("{} expected instead of parameter {}",expected_type,param.type_)),
            sr::Expr::Array(exprs) => {
                // find tightest element type and check if all elements have compatible types
                let mut type_ = sr::Type::Inferred;
                let count = exprs.len();
                for expr in exprs.iter_mut() {
                    type_ = self.tightest(&type_,&self.expr(expr,expected_type)).expect(&format!("array element types incompatible at {}",expr));
                }

                // if this should be an array
                if let sr::Type::Array(element_expected_type,_) = expected_type {

                    // check if the element type is correct
                    if type_ == **element_expected_type {
                        expected_type.clone()
                    }
                    else {
                        panic!("array element should be {} instead of {}",element_expected_type,type_);
                    }
                }

                // if don't care, attempt to infer the type
                else if let sr::Type::Inferred = expected_type {
                    sr::Type::Array(Box::new(type_),Box::new(sr::Expr::Integer(count as i64)))
                }

                // otherwise this is incorrect
                else {
                    panic!("{} expected instead of [{}; {}]",expected_type,type_,count);
                }
            },
            sr::Expr::Cloned(expr,count) => {
                // infer element type
                let type_ = self.expr(expr,expected_type);

                // if this should be an array
                if let sr::Type::Array(element_expected_type,_) = expected_type {

                    // check if element type is correct
                    if type_ == **element_expected_type {
                        expected_type.clone()
                    }
                    else {
                        panic!("array element should be {} instead of {}",expected_type,type_);
                    }
                }

                // if don't care, attempt to infer the type
                else if let sr::Type::Inferred = expected_type {
                    sr::Type::Array(Box::new(type_),count.clone())
                }

                // otherwise this is incorrect
                else {
                    panic!("{} expected instead of [{}; {}]",expected_type,type_,count);
                }
            },
            sr::Expr::Struct(struct_,fields) => {
                // check if the fields match the struct fields
                for i in 0..struct_.fields.len() {
                    self.expr(&mut fields[i].1,&struct_.fields[i].type_);
                }

                // if this should be a struct
                if let sr::Type::Struct(expected_struct) = expected_type {

                    // check if it's the right one
                    if struct_ == expected_struct {
                        expected_type.clone()
                    }
                    else {
                        panic!("{} expected instead of {}",expected_struct,struct_);
                    }
                }

                // if don't care, return the referenced struct
                else if let sr::Type::Inferred = expected_type {
                    sr::Type::Struct(Rc::clone(&struct_))
                }
                
                // otherwise this is incorrect
                else {
                    panic!("{} expected instead of struct {}",expected_type,struct_);
                }
            },
            sr::Expr::Call(function,exprs) => {
                // check if the params match the function params
                for i in 0..function.params.len() {
                    self.expr(&mut exprs[i],&function.params[i].type_);
                }

                // and return the tightest of the two
                self.tightest(&function.return_type,expected_type).expect(&format!("function should return {} instead of {}",expected_type,function.return_type))
            },
            sr::Expr::Field(expr,ident) => {
                // check if expr is a struct and extract the field
                let type_ = if let sr::Type::Struct(struct_) = self.expr(expr,&sr::Type::Inferred) {
                    let mut found_type: Option<sr::Type> = None;
                    for field in struct_.fields.iter() {
                        if field.ident == *ident {
                            found_type = Some(field.type_.clone());
                            break;
                        }
                    }
                    found_type.expect(&format!("{} not a field of {}",ident,struct_))
                }
                else {
                    panic!("{} is not a struct",expr);
                };

                // and return the tightest of the two
                self.tightest(&type_,expected_type).expect(&format!("{} expected instead of {}",expected_type,type_))
            },
            sr::Expr::Method(expr,ident,exprs) => {
                /*
                // check if the params match the method params
                for i in 0..methods.params.len() {
                    self.expr(&mut exprs[i],&methods.params[i].type_);
                }

                // and return the tightest of the two
                self.tightest(&function.return_type,expected_type).expect(&format!("function should return {} instead of {}",expected_type,function.return_type))
                */
                sr::Type::Inferred
            },
            sr::Expr::Index(expr,_) => {
                // check if expr is an array
                let type_ = if let sr::Type::Array(type_,_) = self.expr(expr,&sr::Type::Inferred) {
                    type_
                }
                else {
                    panic!("{} is not an array",expr);
                };

                // and return the tightest of the two
                self.tightest(&type_,expected_type).expect(&format!("{} expected instead of {}",expected_type,type_))
            },
            sr::Expr::Cast(_,type_) => self.tightest(type_,expected_type).expect(&format!("{} expected instead of {}",expected_type,type_)),
            sr::Expr::AnonTuple(exprs) => {
                if let sr::Type::Struct(struct_) = expected_type {
                    // check each expression against the suggested type
                    for i in 0..exprs.len() {
                        self.expr(&mut exprs[i],&struct_.fields[i].type_);
                    }

                    // turn into Expr::Struct
                    let mut fields: Vec<(String,sr::Expr)> = Vec::new();
                    for i in 0..exprs.len() {
                        fields.push((struct_.fields[i].ident.clone(),exprs[i].clone()));
                    }
                    *expr = sr::Expr::Struct(Rc::clone(&struct_),fields);

                    expected_type.clone()
                }
                else {
                    panic!("attempting to match anonymous tuple with {}",expected_type);
                }
            },
            sr::Expr::Neg(expr) => self.expr(expr,expected_type),
            sr::Expr::Not(expr) => self.expr(expr,expected_type),
            sr::Expr::Mul(expr,expr2) |
            sr::Expr::Div(expr,expr2) |
            sr::Expr::Mod(expr,expr2) |
            sr::Expr::Add(expr,expr2) |
            sr::Expr::Sub(expr,expr2) |
            sr::Expr::Shl(expr,expr2) |
            sr::Expr::Shr(expr,expr2) |
            sr::Expr::And(expr,expr2) |
            sr::Expr::Or(expr,expr2) |
            sr::Expr::Xor(expr,expr2) |
            sr::Expr::Assign(expr,expr2) |
            sr::Expr::AddAssign(expr,expr2) |
            sr::Expr::SubAssign(expr,expr2) |
            sr::Expr::MulAssign(expr,expr2) |
            sr::Expr::DivAssign(expr,expr2) |
            sr::Expr::ModAssign(expr,expr2) |
            sr::Expr::AndAssign(expr,expr2) |
            sr::Expr::OrAssign(expr,expr2) |
            sr::Expr::XorAssign(expr,expr2) |
            sr::Expr::ShlAssign(expr,expr2) |
            sr::Expr::ShrAssign(expr,expr2) => self.tightest(&self.expr(expr,expected_type),&self.expr(expr2,expected_type)).expect(&format!("types of {} and {} incompatible",expr,expr2)),
            sr::Expr::Eq(expr,expr2) |
            sr::Expr::NotEq(expr,expr2) |
            sr::Expr::Greater(expr,expr2) |
            sr::Expr::Less(expr,expr2) |
            sr::Expr::GreaterEq(expr,expr2) |
            sr::Expr::LessEq(expr,expr2) |
            sr::Expr::LogAnd(expr,expr2) |
            sr::Expr::LogOr(expr,expr2) => {
                if let sr::Type::Boolean = expected_type {
                    self.tightest(&self.expr(expr,&sr::Type::Inferred),&self.expr(expr2,&sr::Type::Inferred)).expect(&format!("types of {} and {} incompatible",expr,expr2));
                }
                else {
                    panic!("{} expected instead of boolean from comparison result or logical operation",expected_type);
                }
                sr::Type::Boolean
            },
            sr::Expr::Continue => self.tightest(&sr::Type::Void,expected_type).expect(&format!("{} expected instead of void",expected_type)),
            sr::Expr::Break(expr) |
            sr::Expr::Return(expr) => if let Some(expr) = expr {
                self.expr(expr,expected_type)
            }
            else {
                self.tightest(&sr::Type::Void,expected_type).expect(&format!("{} expected instead of void",expected_type))
            },
            sr::Expr::Block(block) => self.block(block,expected_type),
            sr::Expr::If(expr,block,else_expr) => {
                self.expr(expr,&sr::Type::Boolean);
                let mut type_ = self.block(block,expected_type);
                if let Some(else_expr) = else_expr {
                    type_ = self.tightest(&type_,&self.expr(else_expr,expected_type)).expect(&format!("types of {} and {} incompatible",expr,else_expr));
                }
                if let sr::Type::Inferred = expected_type {
                    type_
                }
                else {
                    panic!("{} expected instead if {}",expected_type,type_);
                }
            },
            sr::Expr::Loop(block) => {
                self.block(block,&sr::Type::Inferred);
                expected_type.clone()
            },                
            sr::Expr::For(_,_,block) => {
                self.block(block,&sr::Type::Inferred);
                expected_type.clone()
            },
            sr::Expr::While(expr,block) => {
                if let sr::Type::Boolean = self.expr(expr,&sr::Type::Boolean) {
                    self.block(block,&sr::Type::Inferred);
                    expected_type.clone()
                }
                else {
                    panic!("while-expression {} should be boolean",expr);
                }
            },
        }
    }
}

pub fn resolve_anon_tuples(module: &mut sr::Module) {

    // create resolver context
    let resolver = Resolver { };

    // resolve function block
    let mut new_functions: HashMap<String,Rc<sr::Function>> = HashMap::new();
    for (_,function) in module.functions.iter() {
        let mut new_block = function.block.clone();
        resolver.block(&mut new_block,&function.return_type);
        let new_function = Rc::new(sr::Function { ident: function.ident.clone(),params: Vec::new(),return_type: function.return_type.clone(), block: new_block, });
        new_functions.insert(function.ident.clone(),new_function);
    }

    // resolve constant expressions
    let mut new_consts: HashMap<String,Rc<sr::Variable>> = HashMap::new();
    for (_,const_) in module.consts.iter() {
        let mut new_value = const_.value.as_ref().unwrap().clone();
        resolver.expr(&mut new_value,&const_.type_);
        new_consts.insert(const_.ident.clone(),Rc::new(sr::Variable { ident: const_.ident.clone(),type_: const_.type_.clone(),value: Some(new_value), }));
    }

    // and rebuild module
    module.consts = new_consts;
    module.functions = new_functions;
}
