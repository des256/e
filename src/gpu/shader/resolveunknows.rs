// resolve all unknown references

use {
    crate::*,
    std::{
        rc::Rc,
        collections::HashMap,
    },
};

struct Resolver {
    pub consts: HashMap<String,Rc<sr::Variable>>,
    pub structs: HashMap<String,Rc<sr::Struct>>,
    pub functions: HashMap<String,Rc<sr::Function>>,
    pub params: HashMap<String,Rc<sr::Variable>>,
    pub locals: HashMap<String,Rc<sr::Variable>>,
}

impl Resolver {

    fn block(&mut self,block: &mut sr::Block) {
        let local_frame = self.locals.clone();
        for stat in block.stats.iter_mut() {
            self.stat(stat);
        }
        if let Some(mut expr) = block.expr.as_mut() {
            self.expr(&mut expr);
        }
        self.locals = local_frame;
    }

    fn expr(&mut self,expr: &mut sr::Expr) {

        match expr {

            // no changes to literals
            sr::Expr::Boolean(_) => { },
            sr::Expr::Integer(_) => { },
            sr::Expr::Float(_) => { },

            // process base type literals
            sr::Expr::Base(_,fields) => {
                for field in fields.iter_mut() {
                    self.expr(&mut field.1);
                }
            },

            // convert unknown identifier to Const, Local or Param reference
            sr::Expr::UnknownIdent(ident) => {
                if self.consts.contains_key(ident) {
                    *expr = sr::Expr::Const(Rc::clone(&self.consts[ident]))
                }
                else if self.locals.contains_key(ident) {
                    *expr = sr::Expr::Local(Rc::clone(&self.locals[ident]))
                }
                else if self.params.contains_key(ident) {
                    *expr = sr::Expr::Param(Rc::clone(&self.params[ident]))
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },

            // these remain unchanged, if they even appear here
            sr::Expr::Const(_) => { },
            sr::Expr::Local(_) => { },
            sr::Expr::Param(_) => { },

            // process each element, as it might refer to other identifiers
            sr::Expr::Array(exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },

            // process both expressions
            sr::Expr::Cloned(expr,expr2) => {
                self.expr(expr);
                self.expr(expr2);
            },

            // find struct this refers to
            sr::Expr::UnknownStruct(ident,fields) => {
                if self.structs.contains_key(ident) {
                    for field in fields.iter_mut() {
                        self.expr(&mut field.1);
                    }
                    *expr = sr::Expr::Struct(Rc::clone(&self.structs[ident]),fields.clone())
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },

            // passthrough
            sr::Expr::Struct(_,fields) => {
                for field in fields.iter_mut() {
                    self.expr(&mut field.1);
                }
            },

            // find function or tuple this refers to
            sr::Expr::UnknownCallOrTuple(ident,exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
                if self.functions.contains_key(ident) {
                    *expr = sr::Expr::Call(Rc::clone(&self.functions[ident]),exprs.clone())
                }
                else if self.structs.contains_key(ident) {
                    let mut fields: Vec<(String,sr::Expr)> = Vec::new();
                    let mut i = 0usize;
                    for expr in exprs {
                        fields.push((format!("_{}",i),expr.clone()));
                        i += 1;
                    }
                    *expr = sr::Expr::Struct(Rc::clone(&self.structs[ident]),fields)
                }
                else {
                    panic!("{} is not a function or a tuple",ident);
                }
            },

            // passthrough
            sr::Expr::Call(_,exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },

            // process exprs and type
            sr::Expr::Field(expr,_) => self.expr(expr),
            sr::Expr::Index(expr,expr2) => {
                self.expr(expr);
                self.expr(expr2);
            },
            sr::Expr::Cast(expr,type_) => {
                self.expr(expr);
                self.type_(type_);
            },

            // process each element of the anonymous tuple (the tuple will be converted later when we can sufficiently determine the type of each element)
            sr::Expr::AnonTuple(exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },

            // check expressions for references
            sr::Expr::Neg(expr) |
            sr::Expr::Not(expr) => self.expr(expr),
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
            sr::Expr::Eq(expr,expr2) |
            sr::Expr::NotEq(expr,expr2) |
            sr::Expr::Greater(expr,expr2) |
            sr::Expr::Less(expr,expr2) |
            sr::Expr::GreaterEq(expr,expr2) |
            sr::Expr::LessEq(expr,expr2) |
            sr::Expr::LogAnd(expr,expr2) |
            sr::Expr::LogOr(expr,expr2) |
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
            sr::Expr::ShrAssign(expr,expr2) => { self.expr(expr); self.expr(expr2); },

            // remains the same
            sr::Expr::Continue => { },

            // check expressions
            sr::Expr::Break(expr) |
            sr::Expr::Return(expr) => if let Some(expr) = expr {
                self.expr(expr);
            },

            // process block
            sr::Expr::Block(block) => self.block(block),

            // process expressions and blocks for references
            sr::Expr::If(expr,block,else_expr) => {
                self.expr(expr);
                self.block(block);
                if let Some(else_expr) = else_expr {
                    self.expr(else_expr);
                }
            },
            sr::Expr::Loop(block) => self.block(block),
            sr::Expr::For(ident,range,block) => {
                match range {
                    sr::Range::Only(expr) => self.expr(expr),
                    sr::Range::FromTo(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    sr::Range::FromToIncl(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    sr::Range::From(expr) => self.expr(expr),
                    sr::Range::To(expr) => self.expr(expr),
                    sr::Range::ToIncl(expr) => self.expr(expr),
                    sr::Range::All => { },
                };
                self.block(block);
            },
            sr::Expr::While(expr,block) => {
                self.expr(expr);
                self.block(block);
            },
        }
    }

    fn stat(&mut self,stat: &mut sr::Stat) {
        match stat {
            sr::Stat::Let(ident,type_,expr) => {
                self.type_(type_);
                if let Some(expr) = expr {
                    self.expr(expr);
                }
            },
            sr::Stat::Expr(expr) => self.expr(expr),
        }
    }

    fn type_(&mut self,type_: &mut sr::Type) {
        match type_ {
            sr::Type::Inferred => { },
            sr::Type::Boolean => { },
            sr::Type::Integer => { },
            sr::Type::Float => { },
            sr::Type::Void => { },
            sr::Type::Base(_) => { },
            sr::Type::UnknownIdent(ident) => {
                if self.structs.contains_key(ident) {
                    *type_ = sr::Type::Struct(Rc::clone(&self.structs[ident]));
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },
            sr::Type::Struct(_) => { },
            sr::Type::Array(type_,expr) => {
                self.type_(type_);
                self.expr(expr);
            },
        }
    }
}

pub fn resolve_unknowns(module: &mut sr::Module) {

    // create resolver context
    let mut resolver = Resolver {
        consts: module.consts.clone(),
        structs: module.structs.clone(),
        functions: module.functions.clone(),
        locals: HashMap::new(),
        params: HashMap::new(),
    };

    // resolve function parameter types, return type and block
    let mut new_functions: HashMap<String,Rc<sr::Function>> = HashMap::new();
    for (_,function) in module.functions.iter() {
        let mut new_params: Vec<Rc<sr::Variable>> = Vec::new();
        for param in function.params.iter() {
            let mut new_type = param.type_.clone();
            resolver.type_(&mut new_type);
            let new_param = Rc::new(sr::Variable { ident: param.ident.clone(),type_: new_type,value: None, });
            resolver.params.insert(param.ident.clone(),Rc::clone(&new_param));
            new_params.push(new_param);
        }
        let mut new_return_type = function.return_type.clone();
        resolver.type_(&mut new_return_type);
        let mut new_block = function.block.clone();
        resolver.block(&mut new_block);
        resolver.params.clear();
        let new_function = Rc::new(sr::Function { ident: function.ident.clone(),params: new_params,return_type: new_return_type, block: new_block, });
        new_functions.insert(function.ident.clone(),new_function);
    }

    // resolve struct field types
    let mut new_structs: HashMap<String,Rc<sr::Struct>> = HashMap::new();
    for (_,struct_) in module.structs.iter() {
        let mut new_fields: Vec<sr::Field> = Vec::new();
        for field in struct_.fields.iter() {
            let mut new_type = field.type_.clone();
            resolver.type_(&mut new_type);
            new_fields.push(sr::Field { ident: field.ident.clone(),type_: new_type, });
        }
        let new_struct = Rc::new(sr::Struct { ident: struct_.ident.clone(),fields: new_fields, });
        new_structs.insert(struct_.ident.clone(),new_struct);
    }

    // resolve constant types and expressions
    let mut new_consts: HashMap<String,Rc<sr::Variable>> = HashMap::new();
    for (_,const_) in module.consts.iter() {
        let mut new_type = const_.type_.clone();
        resolver.type_(&mut new_type);
        let mut new_value = const_.value.as_ref().unwrap().clone();
        resolver.expr(&mut new_value);
        new_consts.insert(const_.ident.clone(),Rc::new(sr::Variable { ident: const_.ident.clone(),type_: new_type,value: Some(new_value), }));
    }

    // and rebuild module
    module.consts = new_consts;
    module.structs = new_structs;
    module.functions = new_functions;
}
