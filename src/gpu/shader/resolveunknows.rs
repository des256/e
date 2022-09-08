// resolve Type::Ident, Expr::Ident and Pat::Ident

use {
    crate::*,
    std::{
        rc::Rc,
        collections::HashMap,
    },
};

struct Resolver<'module> {
    pub module: &'module sr::Module,
    pub extern_vertex: Option<String>,
    pub locals: HashMap<String,Rc<sr::Variable>>,
    pub params: HashMap<String,Rc<sr::Variable>>,
    pub anon_tuple_structs: HashMap<String,Vec<(String,sr::Type)>>,
}

impl<'module> Resolver<'module> {

    fn block(&mut self,block: &sr::Block) -> sr::Block {

        // save local variable context
        let local_frame = self.locals.clone();

        // process the statements and expression
        let mut new_stats: Vec<sr::Stat> = Vec::new();
        for stat in block.stats.iter() {
            new_stats.push(self.stat(stat));
        }
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.expr(expr)))
        }
        else {
            None
        };

        // restore local variables to previous context
        self.locals = local_frame;

        sr::Block { stats: new_stats,expr: new_expr, }
    }

    fn expr(&mut self,expr: &sr::Expr) -> sr::Expr {

        match expr {

            // no changes to literals
            sr::Expr::Boolean(value) => sr::Expr::Boolean(*value),
            sr::Expr::Integer(value) => sr::Expr::Integer(*value),
            sr::Expr::Float(value) => sr::Expr::Float(*value),

            // process base type literals
            sr::Expr::Base(ident,fields) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.expr(expr)));
                }
                sr::Expr::Base(ident.clone(),new_fields)
            },

            // convert unknown identifier to Const, Local or Param reference
            sr::Expr::UnknownIdent(ident) => {
                if self.module.consts.contains_key(ident) {
                    sr::Expr::Const(Rc::clone(&self.module.consts[ident]))
                }
                else if self.locals.contains_key(ident) {
                    sr::Expr::Local(Rc::clone(&self.locals[ident]))
                }
                else if self.params.contains_key(ident) {
                    sr::Expr::Param(Rc::clone(&self.params[ident]))
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },

            // these remain unchanged, if they even appear here
            sr::Expr::Const(ident) => sr::Expr::Const(ident.clone()),
            sr::Expr::Local(ident) => sr::Expr::Local(ident.clone()),
            sr::Expr::Param(ident) => sr::Expr::Param(ident.clone()),

            // process each element, as it might refer to other identifiers
            sr::Expr::Array(exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::Array(new_exprs)
            },

            // process both expressions
            sr::Expr::Cloned(expr,expr2) => sr::Expr::Cloned(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),

            // find struct this refers to
            sr::Expr::UnknownStruct(ident,fields) => {
                if self.module.structs.contains_key(ident) {
                    let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                    for (ident,expr) in fields {
                        new_fields.push((ident.clone(),self.expr(expr)));
                    }
                    sr::Expr::Struct(Rc::clone(&self.module.structs[ident]),new_fields)    
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },

            // passthrough
            sr::Expr::Struct(struct_,fields) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.expr(expr)));
                }
                sr::Expr::Struct(Rc::clone(&struct_),new_fields)    
            },

            // find function or tuple this refers to
            sr::Expr::UnknownCall(ident,exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                if self.module.functions.contains_key(ident) {
                    sr::Expr::Call(Rc::clone(&self.module.functions[ident]),new_exprs)
                }
                else if self.module.structs.contains_key(ident) {
                    let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                    let mut i = 0usize;
                    for expr in new_exprs {
                        new_fields.push((format!("_{}",i),expr));
                        i += 1;
                    }
                    sr::Expr::Struct(Rc::clone(&self.module.structs[ident]),new_fields)
                }
                else {
                    panic!("{} is not a function or a tuple",ident);
                }
            },

            // passthrough
            sr::Expr::Call(function,exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::Call(Rc::clone(&function),new_exprs)
            },

            // find variant this refers to
            sr::Expr::UnknownVariant(ident,variantexpr) => {
                if self.module.enums.contains_key(ident) {
                    let new_variantexpr = match variantexpr {
                        sr::VariantExpr::Naked(ident) => sr::VariantExpr::Naked(ident.clone()),
                        sr::VariantExpr::Tuple(ident,exprs) => {
                            let mut new_exprs: Vec<sr::Expr> = Vec::new();
                            for expr in exprs {
                                new_exprs.push(self.expr(expr));
                            }
                            sr::VariantExpr::Tuple(ident.clone(),new_exprs)
                        },
                        sr::VariantExpr::Struct(ident,fields) => {
                            let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                            for (ident,expr) in fields {
                                new_fields.push((ident.clone(),self.expr(expr)));
                            }
                            sr::VariantExpr::Struct(ident.clone(),new_fields)
                        },
                    };
                    sr::Expr::Variant(Rc::clone(&self.module.enums[ident]),new_variantexpr)    
                }
                else {
                    panic!("unknown enum {}",ident);
                }
            },

            // find variant this refers to
            sr::Expr::Variant(enum_,variantexpr) => {
                let new_variantexpr = match variantexpr {
                    sr::VariantExpr::Naked(ident) => sr::VariantExpr::Naked(ident.clone()),
                    sr::VariantExpr::Tuple(ident,exprs) => {
                        let mut new_exprs: Vec<sr::Expr> = Vec::new();
                        for expr in exprs {
                            new_exprs.push(self.expr(expr));
                        }
                        sr::VariantExpr::Tuple(ident.clone(),new_exprs)
                    },
                    sr::VariantExpr::Struct(ident,fields) => {
                        let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                        for (ident,expr) in fields {
                            new_fields.push((ident.clone(),self.expr(expr)));
                        }
                        sr::VariantExpr::Struct(ident.clone(),new_fields)
                    },
                };
                sr::Expr::Variant(Rc::clone(&enum_),new_variantexpr)    
            },

            // process exprs and type
            sr::Expr::Field(expr,ident) => sr::Expr::Field(Box::new(self.expr(expr)),ident.clone()),
            sr::Expr::Index(expr,expr2) => sr::Expr::Index(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Cast(expr,type_) => sr::Expr::Cast(Box::new(self.expr(expr)),Box::new(self.type_(type_))),

            // process each element of the anonymous tuple (the tuple will be converted later when we can sufficiently determine the type of each element)
            sr::Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::AnonTuple(new_exprs)
            },

            // check expressions for references
            sr::Expr::Neg(expr) => sr::Expr::Neg(Box::new(self.expr(expr))),
            sr::Expr::Not(expr) => sr::Expr::Not(Box::new(self.expr(expr))),
            sr::Expr::Mul(expr,expr2) => sr::Expr::Mul(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Div(expr,expr2) => sr::Expr::Div(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Mod(expr,expr2) => sr::Expr::Mod(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Add(expr,expr2) => sr::Expr::Add(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Sub(expr,expr2) => sr::Expr::Sub(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Shl(expr,expr2) => sr::Expr::Shl(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Shr(expr,expr2) => sr::Expr::Shr(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::And(expr,expr2) => sr::Expr::And(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Or(expr,expr2) => sr::Expr::Or(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Xor(expr,expr2) => sr::Expr::Xor(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Eq(expr,expr2) => sr::Expr::Eq(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::NotEq(expr,expr2) => sr::Expr::NotEq(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Greater(expr,expr2) => sr::Expr::Greater(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Less(expr,expr2) => sr::Expr::Less(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::GreaterEq(expr,expr2) => sr::Expr::GreaterEq(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::LessEq(expr,expr2) => sr::Expr::LessEq(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::LogAnd(expr,expr2) => sr::Expr::LogAnd(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::LogOr(expr,expr2) => sr::Expr::LogOr(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Assign(expr,expr2) => sr::Expr::Assign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::AddAssign(expr,expr2) => sr::Expr::AddAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::SubAssign(expr,expr2) => sr::Expr::SubAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::MulAssign(expr,expr2) => sr::Expr::MulAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::DivAssign(expr,expr2) => sr::Expr::DivAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::ModAssign(expr,expr2) => sr::Expr::ModAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::AndAssign(expr,expr2) => sr::Expr::AndAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::OrAssign(expr,expr2) => sr::Expr::OrAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::XorAssign(expr,expr2) => sr::Expr::XorAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::ShlAssign(expr,expr2) => sr::Expr::ShlAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::ShrAssign(expr,expr2) => sr::Expr::ShrAssign(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),

            // remains the same
            sr::Expr::Continue => sr::Expr::Continue,

            // check expressions
            sr::Expr::Break(expr) => if let Some(expr) = expr {
                sr::Expr::Break(Some(Box::new(self.expr(expr))))
            }
            else {
                sr::Expr::Break(None)
            },
            sr::Expr::Return(expr) => if let Some(expr) = expr {
                sr::Expr::Return(Some(Box::new(self.expr(expr))))
            }
            else {
                sr::Expr::Return(None)
            },

            // process block
            sr::Expr::Block(block) => sr::Expr::Block(self.block(block)),

            // process expressions and blocks for references
            sr::Expr::If(expr,block,else_expr) => {
                let new_expr = self.expr(expr);
                let new_block = self.block(block);
                if let Some(else_expr) = else_expr {
                    sr::Expr::If(Box::new(new_expr),new_block,Some(Box::new(self.expr(else_expr))))
                }
                else {
                    sr::Expr::If(Box::new(new_expr),new_block,None)
                }
            },
            sr::Expr::IfLet(pats,expr,block,else_expr) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                let new_expr = self.expr(expr);
                let new_block = self.block(block);
                if let Some(else_expr) = else_expr {
                    sr::Expr::IfLet(new_pats,Box::new(new_expr),new_block,Some(Box::new(self.expr(else_expr))))
                }
                else {
                    sr::Expr::IfLet(new_pats,Box::new(new_expr),new_block,None)
                }
            },
            sr::Expr::Loop(block) => sr::Expr::Loop(self.block(block)),
            sr::Expr::For(pats,range,block) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                let new_range = match range {
                    sr::Range::Only(expr) => sr::Range::Only(Box::new(self.expr(expr))),
                    sr::Range::FromTo(expr,expr2) => sr::Range::FromTo(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
                    sr::Range::FromToIncl(expr,expr2) => sr::Range::FromToIncl(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
                    sr::Range::From(expr) => sr::Range::From(Box::new(self.expr(expr))),
                    sr::Range::To(expr) => sr::Range::To(Box::new(self.expr(expr))),
                    sr::Range::ToIncl(expr) => sr::Range::ToIncl(Box::new(self.expr(expr))),
                    sr::Range::All => sr::Range::All,
                };
                let new_block = self.block(block);
                sr::Expr::For(new_pats,new_range,new_block)
            },
            sr::Expr::While(expr,block) => sr::Expr::While(Box::new(self.expr(expr)),self.block(block)),
            sr::Expr::WhileLet(pats,expr,block) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                let new_expr = self.expr(expr);
                let new_block = self.block(block);
                sr::Expr::WhileLet(new_pats,Box::new(new_expr),new_block)
            },
            sr::Expr::Match(expr,arms) => {
                let new_expr = self.expr(expr);
                let mut new_arms: Vec<(Vec<sr::Pat>,Option<Box<sr::Expr>>,Box<sr::Expr>)> = Vec::new();
                for (pats,if_expr,expr) in arms {
                    let mut new_pats: Vec<sr::Pat> = Vec::new();
                    for pat in pats {
                        new_pats.push(self.pat(pat));
                    }
                    let new_if_expr = if let Some(if_expr) = if_expr {
                        Some(Box::new(self.expr(if_expr)))
                    }
                    else {
                        None
                    };
                    let new_expr = self.expr(expr);
                    new_arms.push((new_pats,new_if_expr,Box::new(new_expr)))
                }
                sr::Expr::Match(Box::new(new_expr),new_arms)
            },
        }
    }

    fn stat(&mut self,stat: &sr::Stat) -> sr::Stat {
        match stat {
            // check expression and type
            sr::Stat::Let(variable) => {
                let new_type = self.type_(&variable.type_);
                let new_expr = self.expr(&variable.value.as_ref().unwrap());
                let new_variable = Rc::new(sr::Variable { ident: variable.ident.clone(),type_: new_type,value: Some(new_expr), });
                self.locals.insert(variable.ident,Rc::clone(&new_variable));
                sr::Stat::Let(new_variable)
            },
            sr::Stat::Expr(expr) => sr::Stat::Expr(Box::new(self.expr(expr))),
        }
    }

    fn type_(&mut self,type_: &sr::Type) -> sr::Type {
        match type_ {
            sr::Type::Inferred => sr::Type::Inferred,
            sr::Type::Boolean => sr::Type::Boolean,
            sr::Type::Integer => sr::Type::Integer,
            sr::Type::Float => sr::Type::Float,
            sr::Type::Void => sr::Type::Void,
            sr::Type::Base(base_type) => sr::Type::Base(base_type.clone()),
            sr::Type::UnknownIdent(ident) => {
                if self.module.structs.contains_key(ident) {
                    sr::Type::Struct(Rc::clone(&self.module.structs[ident]))
                }
                else if self.module.enums.contains_key(ident) {
                    sr::Type::Enum(Rc::clone(&self.module.enums[ident]))
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },
            sr::Type::Struct(ident) => sr::Type::Struct(ident.clone()),
            sr::Type::Enum(ident) => sr::Type::Enum(ident.clone()),
            sr::Type::Array(type_,expr) => sr::Type::Array(Box::new(self.type_(type_)),Box::new(self.expr(expr))),
        }
    }

    fn pat(&mut self,pat: &sr::Pat) -> sr::Pat {
        match pat {
            sr::Pat::Wildcard => sr::Pat::Wildcard,
            sr::Pat::Rest => sr::Pat::Rest,
            sr::Pat::Boolean(value) => sr::Pat::Boolean(*value),
            sr::Pat::Integer(value) => sr::Pat::Integer(*value),
            sr::Pat::Float(value) => sr::Pat::Float(*value),
            sr::Pat::Ident(ident) => {
                if self.module.consts.contains_key(ident) {
                    sr::Pat::Const(ident.clone())
                }
                else {
                    sr::Pat::Ident(ident.clone())
                }
            },
            sr::Pat::Const(ident) => sr::Pat::Const(ident.clone()),
            sr::Pat::UnknownStruct(ident,identpats) => {
                if self.module.structs.contains_key(ident) {
                    let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                    for identpat in identpats {
                        new_identpats.push(match identpat {
                            sr::IdentPat::Wildcard => sr::IdentPat::Wildcard,
                            sr::IdentPat::Rest => sr::IdentPat::Rest,
                            sr::IdentPat::Ident(ident) => sr::IdentPat::Ident(ident.clone()),
                            sr::IdentPat::IdentPat(ident,pat) => sr::IdentPat::IdentPat(ident.clone(),self.pat(pat)),
                        });
                    }
                    sr::Pat::Struct(Rc::clone(&self.module.structs[ident]),new_identpats)
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },
            sr::Pat::Struct(struct_,identpats) => {
                let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                for identpat in identpats {
                    new_identpats.push(match identpat {
                        sr::IdentPat::Wildcard => sr::IdentPat::Wildcard,
                        sr::IdentPat::Rest => sr::IdentPat::Rest,
                        sr::IdentPat::Ident(ident) => sr::IdentPat::Ident(ident.clone()),
                        sr::IdentPat::IdentPat(ident,pat) => sr::IdentPat::IdentPat(ident.clone(),self.pat(pat)),
                    });
                }
                sr::Pat::Struct(Rc::clone(&struct_),new_identpats)
            },
            sr::Pat::Array(pats) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                sr::Pat::Array(new_pats)
            },
            sr::Pat::UnknownVariant(ident,variantpat) => {
                if self.module.enums.contains_key(ident) {
                    let new_variantpat = match variantpat {
                        sr::VariantPat::Naked(ident) => sr::VariantPat::Naked(ident.clone()),
                        sr::VariantPat::Tuple(ident,pats) => {
                            let mut new_pats: Vec<sr::Pat> = Vec::new();
                            for pat in pats {
                                new_pats.push(self.pat(pat));
                            }
                            sr::VariantPat::Tuple(ident.clone(),new_pats)
                        },
                        sr::VariantPat::Struct(ident,identpats) => {
                            let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                            for identpat in identpats {
                                new_identpats.push(match identpat {
                                    sr::IdentPat::Wildcard => sr::IdentPat::Wildcard,
                                    sr::IdentPat::Rest => sr::IdentPat::Rest,
                                    sr::IdentPat::Ident(ident) => sr::IdentPat::Ident(ident.clone()),
                                    sr::IdentPat::IdentPat(ident,pat) => sr::IdentPat::IdentPat(ident.clone(),self.pat(pat)),
                                });
                            }
                            sr::VariantPat::Struct(ident.clone(),new_identpats)
                        },
                    };
                    sr::Pat::Variant(Rc::clone(&self.module.enums[ident]),new_variantpat)
                }
                else {
                    panic!("unknown enum {}",ident);
                }
            },
            sr::Pat::Variant(enum_,variantpat) => {
                let new_variantpat = match variantpat {
                    sr::VariantPat::Naked(ident) => sr::VariantPat::Naked(ident.clone()),
                    sr::VariantPat::Tuple(ident,pats) => {
                        let mut new_pats: Vec<sr::Pat> = Vec::new();
                        for pat in pats {
                            new_pats.push(self.pat(pat));
                        }
                        sr::VariantPat::Tuple(ident.clone(),new_pats)
                    },
                    sr::VariantPat::Struct(ident,identpats) => {
                        let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                        for identpat in identpats {
                            new_identpats.push(match identpat {
                                sr::IdentPat::Wildcard => sr::IdentPat::Wildcard,
                                sr::IdentPat::Rest => sr::IdentPat::Rest,
                                sr::IdentPat::Ident(ident) => sr::IdentPat::Ident(ident.clone()),
                                sr::IdentPat::IdentPat(ident,pat) => sr::IdentPat::IdentPat(ident.clone(),self.pat(pat)),
                            });
                        }
                        sr::VariantPat::Struct(ident.clone(),new_identpats)
                    },
                };
                sr::Pat::Variant(Rc::clone(&enum_),new_variantpat)
            },
            sr::Pat::Range(pat,pat2) => sr::Pat::Range(Box::new(self.pat(pat)),Box::new(self.pat(pat2))),
        }
    }
}

pub fn resolve_idents(module: &sr::Module,extern_vertex: Option<String>) -> sr::Module {
    let mut resolver = Resolver {
        module,
        extern_vertex,
        locals: HashMap::new(),
        params: HashMap::new(),
        anon_tuple_structs: HashMap::new(),
    };

    let mut new_functions: HashMap<String,Rc<sr::Function>> = HashMap::new();
    for (_,function) in module.functions.iter() {
        let mut new_params: Vec<Rc<sr::Variable>> = Vec::new();
        for param in function.params.iter() {
            let new_type = resolver.type_(&param.type_);
            let variable = Rc::new(sr::Variable { ident: param.ident.clone(),type_: new_type,value: None, });
            new_params.push(Rc::clone(&variable));
            resolver.params.insert(param.ident,variable);
        }
        let new_return_type = resolver.type_(&function.return_type);
        let new_block = resolver.block(&function.block);
        resolver.params.clear();
        new_functions.insert(
            function.ident.clone(),
            Rc::new(
                sr::Function {
                    ident: function.ident.clone(),
                    params: new_params,
                    return_type: new_return_type,
                    block: new_block,
                }
            )
        );
    }

    let mut new_structs: HashMap<String,Rc<sr::Struct>> = HashMap::new();
    for (_,struct_) in module.structs.iter() {
        let mut new_fields: Vec<sr::Field> = Vec::new();
        for field in struct_.fields.iter() {
            new_fields.push(sr::Field { ident: field.ident.clone(),type_: resolver.type_(&field.type_), });
        }
        new_structs.insert(
            struct_.ident.clone(),
            Rc::new(
                sr::Struct {
                    ident: struct_.ident.clone(),
                    fields: new_fields,
                }
            ),
        );
    }

    let mut new_enums: HashMap<String,Rc<sr::Enum>> = HashMap::new();
    for (_,enum_) in module.enums.iter() {
        let mut new_variants: Vec<sr::Variant> = Vec::new();
        for variant in enum_.variants {
            new_variants.push(match variant {
                sr::Variant::Naked(ident) => sr::Variant::Naked(ident.clone()),
                sr::Variant::Tuple(ident,types) => {
                    let mut new_types: Vec<sr::Type> = Vec::new();
                    for type_ in types.iter() {
                        new_types.push(resolver.type_(type_));
                    }
                    sr::Variant::Tuple(ident.clone(),new_types)
                },
                sr::Variant::Struct(ident,fields) => {
                    let mut new_fields: Vec<sr::Field> = Vec::new();
                    for field in fields.iter() {
                        new_fields.push(sr::Field { ident: field.ident.clone(),type_: resolver.type_(&field.type_), });
                    }
                    sr::Variant::Struct(ident.clone(),new_fields)
                },
            });
        }
        new_enums.insert(
            enum_.ident.clone(),
            Rc::new(
                sr::Enum {
                    ident: enum_.ident.clone(),
                    variants: new_variants,
                }
            )
        );
    }

    let mut new_consts: HashMap<String,Rc<sr::Variable>> = HashMap::new();
    for (_,const_) in module.consts.iter() {
        new_consts.insert(
            const_.ident.clone(),
            Rc::new(
                sr::Variable {
                    ident: const_.ident.clone(),
                    type_: resolver.type_(&const_.type_),
                    value: Some(resolver.expr(const_.value.as_ref().unwrap())),
                }
            )
        );        
    }

    sr::Module {
        ident: module.ident.clone(),
        functions: new_functions,
        structs: new_structs,
        enums: new_enums,
        consts: new_consts,
    }
}
