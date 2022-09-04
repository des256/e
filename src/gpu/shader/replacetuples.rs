use {
    crate::*,
    std::collections::HashMap,
};

struct Replacer<'module> {
    pub module: &'module sr::Module,
    pub new_structs: HashMap<String,Vec<(String,sr::Type)>>,
    pub locals: HashMap<String,sr::Type>,
    pub params: HashMap<String,sr::Type>,
    pub anon_tuple_count: usize,
}

impl<'module> Replacer<'module> {

    fn block(&mut self,block: &sr::Block) -> sr::Block {
        let local_frame = self.locals.clone();
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
        self.locals = local_frame;
        sr::Block { stats: new_stats,expr: new_expr, }
    }

    fn expr(&mut self,expr: &sr::Expr) -> sr::Expr {
        match expr {
            sr::Expr::Boolean(value) => sr::Expr::Boolean(*value),
            sr::Expr::Integer(value) => sr::Expr::Integer(*value),
            sr::Expr::Float(value) => sr::Expr::Float(*value),
            sr::Expr::Base(base_type,fields) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.expr(expr)));
                }
                sr::Expr::Base(base_type.clone(),new_fields)
            },
            sr::Expr::Local(ident,ty) => sr::Expr::Local(ident.clone(),ty.clone()),
            sr::Expr::Param(ident,ty) => sr::Expr::Param(ident.clone(),ty.clone()),
            sr::Expr::Const(ident,ty) => sr::Expr::Const(ident.clone(),ty.clone()),
            sr::Expr::Array(exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::Array(new_exprs)
            },
            sr::Expr::Cloned(expr,expr2) => sr::Expr::Cloned(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.expr(expr)));
                }
                sr::Expr::Struct(ident.clone(),new_fields)
            },
            sr::Expr::Tuple(ident,exprs) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                let mut i = 0usize;
                for expr in exprs {
                    let ident = format!("_{}",i);
                    i += 1;
                    new_fields.push((ident,self.expr(expr)));
                }
                sr::Expr::Struct(format!("Tuple{}",ident),new_fields)
            },
            sr::Expr::AnonTuple(exprs) => {
                let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
                let mut i = 0usize;
                for expr in exprs {
                    let ident = format!("_{}",i);
                    i += 1;
                    new_fields.push((ident,find_expr_type(self.module,expr)));
                }
                // TODO: figure out if this anonymous tuple already exists
                let ident = format!("AnonTuple{}",self.anon_tuple_count);
                self.new_structs.insert(ident.clone(),new_fields);
                self.anon_tuple_count += 1;

                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                let mut i = 0usize;
                for expr in exprs {
                    new_fields.push((format!("_{}",i),self.expr(expr)));
                    i += 1;
                }
                sr::Expr::Struct(ident,new_fields)
            },
            sr::Expr::Variant(ident,variantexpr) => {
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
                sr::Expr::Variant(ident.clone(),new_variantexpr)
            },
            sr::Expr::Call(ident,exprs,ty) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::Call(ident.clone(),new_exprs,ty.clone())
            },
            sr::Expr::Field(expr,ident,ty) => sr::Expr::Field(Box::new(self.expr(expr)),ident.clone(),ty.clone()),
            sr::Expr::Index(expr,expr2,ty) => sr::Expr::Index(Box::new(self.expr(expr)),Box::new(self.expr(expr2)),ty.clone()),
            sr::Expr::TupleIndex(expr,index,ty) => sr::Expr::Field(Box::new(self.expr(expr)),format!("_{}",index),ty.clone()),
            sr::Expr::Cast(expr,ty) => sr::Expr::Cast(Box::new(self.expr(expr)),self.type_(ty)),
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
            sr::Expr::Continue => sr::Expr::Continue,
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
            sr::Expr::Block(block) => sr::Expr::Block(self.block(block)),
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
            sr::Stat::Let(pat,ty,expr) => {
                let new_pat = self.pat(pat);
                let new_ty = if let Some(ty) = ty {
                    Some(Box::new(self.type_(ty)))
                }
                else {
                    None
                };
                let new_expr = self.expr(expr);
                sr::Stat::Let(new_pat,new_ty,Box::new(new_expr))
            },
            sr::Stat::Expr(expr) => sr::Stat::Expr(Box::new(self.expr(expr))),
        }
    }

    fn type_(&mut self,ty: &sr::Type) -> sr::Type {
        match ty {
            sr::Type::Inferred => sr::Type::Inferred,
            sr::Type::Integer => sr::Type::Integer,
            sr::Type::Float => sr::Type::Float,
            sr::Type::Void => sr::Type::Void,
            sr::Type::Base(base_type) => sr::Type::Base(base_type.clone()),
            sr::Type::Struct(ident) => sr::Type::Struct(ident.clone()),
            sr::Type::Tuple(ident) => sr::Type::Struct(format!("Tuple{}",ident)),
            sr::Type::Enum(ident) => sr::Type::Enum(ident.clone()),
            sr::Type::Array(ty,expr) => sr::Type::Array(Box::new(self.type_(ty)),Box::new(self.expr(expr))),
            sr::Type::AnonTuple(types) => {
                let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
                let mut i = 0usize;
                for ty in types {
                    let ident = format!("_{}",i);
                    i += 1;
                    new_fields.push((ident,self.type_(ty)));
                }
                // TODO: figure out if this anonymous tuple already exists
                let ident = format!("AnonTuple{}",self.anon_tuple_count);
                self.new_structs.insert(ident.clone(),new_fields);
                self.anon_tuple_count += 1;
                sr::Type::Struct(ident)
            },
        }
    }

    fn pat(&mut self,pat: &sr::Pat) -> sr::Pat {
        match pat {
            sr::Pat::Wildcard => sr::Pat::Wildcard,
            sr::Pat::Rest => sr::Pat::Rest,
            sr::Pat::Boolean(value) => sr::Pat::Boolean(*value),
            sr::Pat::Integer(value) => sr::Pat::Integer(*value),
            sr::Pat::Float(value) => sr::Pat::Float(*value),
            sr::Pat::Ident(ident) => sr::Pat::Ident(ident.clone()),
            sr::Pat::Const(ident) => sr::Pat::Const(ident.clone()),
            sr::Pat::Struct(ident,identpats) => {
                let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                for identpat in identpats {
                    new_identpats.push(match identpat {
                        sr::IdentPat::Wildcard => sr::IdentPat::Wildcard,
                        sr::IdentPat::Rest => sr::IdentPat::Rest,
                        sr::IdentPat::Ident(ident) => sr::IdentPat::Ident(ident.clone()),
                        sr::IdentPat::IdentPat(ident,pat) => sr::IdentPat::IdentPat(ident.clone(),self.pat(pat)),
                    });
                }
                sr::Pat::Struct(ident.clone(),new_identpats)
            },
            sr::Pat::Tuple(ident,pats) => {
                let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                let mut i = 0usize;
                for pat in pats {
                    new_identpats.push(sr::IdentPat::IdentPat(format!("_{}",i),self.pat(pat)));
                    i += 1;
                }
                sr::Pat::Struct(format!("Tuple{}",ident),new_identpats)
            },
            sr::Pat::Array(pats) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                sr::Pat::Array(new_pats)
            },
            sr::Pat::AnonTuple(pats) => {
                let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
                let mut i = 0usize;
                for pat in pats {
                    let ident = format!("_{}",i);
                    i += 1;
                    // TODO: figure out what the actual type is of the pattern
                    new_fields.push((ident,sr::Type::Void));
                }
                // TODO: figure out if this anonymous tuple already exists
                let ident = format!("AnonTuple{}",self.anon_tuple_count);
                self.new_structs.insert(ident.clone(),new_fields);
                self.anon_tuple_count += 1;
                let mut new_identpats: Vec<sr::IdentPat> = Vec::new();
                let mut i = 0usize;
                for pat in pats {
                    let ident = format!("_{}",i);
                    i += 1;
                    new_identpats.push(sr::IdentPat::IdentPat(ident,self.pat(pat)));
                }
                sr::Pat::Struct(ident,new_identpats)
            },
            sr::Pat::Variant(ident,variantpat) => {
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
                sr::Pat::Variant(ident.clone(),new_variantpat)
            },
            sr::Pat::Range(pat,pat2) => sr::Pat::Range(Box::new(self.pat(pat)),Box::new(self.pat(pat2))),
        }
    }
}

pub fn replace_tuples(module: sr::Module,vertex_ident: Option<String>,vertex_fields: &Vec<(String,sr::Type)>) -> sr::Module {

    let mut replacer = Replacer {
        module: &module,
        new_structs: HashMap::new(),
        locals: HashMap::new(),
        params: HashMap::new(),
        anon_tuple_count: 0,
    };

    // replace tuples and references in functions
    let mut new_functions: HashMap<String,(Vec<(String,sr::Type)>,sr::Type,sr::Block)> = HashMap::new();
    for (ident,(params,return_type,block)) in &module.functions {
        let mut new_params: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in params {
            let new_type = replacer.type_(ty);
            new_params.push((ident.clone(),new_type.clone()));
            replacer.params.insert(ident.clone(),new_type);
        }
        let new_return_type = replacer.type_(return_type);
        let new_block = replacer.block(block);
        replacer.params.clear();
        new_functions.insert(ident.clone(),(new_params,new_return_type,new_block));
    }

    // replace tuples and references in structs
    let mut new_structs: HashMap<String,Vec<(String,sr::Type)>> = HashMap::new();
    for (ident,fields) in &module.structs {
        let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in fields {
            new_fields.push((ident.clone(),replacer.type_(ty)));
        }
        new_structs.insert(ident.clone(),new_fields);
    }

    // convert tuples to structs
    for (ident,types) in &module.tuples {
        let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
        let mut i = 0usize;
        for ty in types {
            new_fields.push((format!("_{}",i),replacer.type_(ty)));
            i += 1;
        }
        new_structs.insert(format!("Tuple{}",ident),new_fields);
    }

    // replace tuples and references in enums
    let mut new_enums: HashMap<String,Vec<sr::Variant>> = HashMap::new();
    for (ident,variants) in &module.enums {
        let mut new_variants: Vec<sr::Variant> = Vec::new();
        for variant in variants {
            new_variants.push(match variant {
                sr::Variant::Naked(ident) => sr::Variant::Naked(ident.clone()),
                sr::Variant::Tuple(ident,types) => {
                    let mut new_types: Vec<sr::Type> = Vec::new();
                    for ty in types {
                        new_types.push(replacer.type_(ty));
                    }
                    sr::Variant::Tuple(ident.clone(),new_types)
                },
                sr::Variant::Struct(ident,fields) => {
                    let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
                    for (ident,ty) in fields {
                        new_fields.push((ident.clone(),replacer.type_(ty)));
                    }
                    sr::Variant::Struct(ident.clone(),new_fields)
                },
            });
        }
        new_enums.insert(ident.clone(),new_variants);
    }

    // replace tuples and references in constants
    let mut new_consts: HashMap<String,(sr::Type,sr::Expr)> = HashMap::new();
    for (ident,(ty,expr)) in &module.consts {
        new_consts.insert(ident.clone(),(replacer.type_(ty),replacer.expr(expr)));        
    }

    // append new structs that were created from anonymous tuples
    for (ident,fields) in &replacer.new_structs {
        let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in fields {
            new_fields.push((ident.clone(),ty.clone()));
        }
        new_structs.insert(ident.clone(),new_fields);
    }

    sr::Module {
        ident: module.ident,
        functions: new_functions,
        structs: new_structs,
        tuples: HashMap::new(),
        enums: new_enums,
        consts: new_consts,
    }
}
