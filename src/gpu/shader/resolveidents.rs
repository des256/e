// resolve Type::Ident, Expr::Ident and Pat::Ident

use {
    crate::*,
    std::collections::HashMap,
};

struct Resolver<'module> {
    pub module: &'module sr::Module,
    pub extern_vertex: Option<String>,
    pub locals: HashMap<String,sr::Type>,
    pub params: HashMap<String,sr::Type>,
    pub anon_tuple_structs: HashMap<String,Vec<(String,sr::Type)>>,
}

impl<'module> Resolver<'module> {

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
            sr::Expr::Base(ident,fields) => {
                let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                for (ident,expr) in fields {
                    new_fields.push((ident.clone(),self.expr(expr)));
                }
                sr::Expr::Base(ident.clone(),new_fields)
            },
            sr::Expr::Ident(ident) => {
                if self.module.consts.contains_key(ident) {
                    sr::Expr::Const(ident.clone())
                }
                else if self.locals.contains_key(ident) {
                    sr::Expr::Local(ident.clone())
                }
                else if self.params.contains_key(ident) {
                    sr::Expr::Param(ident.clone())
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },
            sr::Expr::Const(ident) => sr::Expr::Const(ident.clone()),
            sr::Expr::Local(ident) => sr::Expr::Local(ident.clone()),
            sr::Expr::Param(ident) => sr::Expr::Param(ident.clone()),
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
            sr::Expr::Call(ident,exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                if self.module.functions.contains_key(ident) {
                    sr::Expr::Call(ident.clone(),new_exprs)
                }
                else {
                    let mut new_fields: Vec<(String,sr::Expr)> = Vec::new();
                    let mut i = 0usize;
                    for expr in new_exprs {
                        new_fields.push((format!("_{}",i),expr));
                        i += 1;
                    }
                    sr::Expr::Struct(ident.clone(),new_fields)
                }
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
            sr::Expr::Field(expr,ident) => sr::Expr::Field(Box::new(self.expr(expr)),ident.clone()),
            sr::Expr::Index(expr,expr2) => sr::Expr::Index(Box::new(self.expr(expr)),Box::new(self.expr(expr2))),
            sr::Expr::Cast(expr,ty) => sr::Expr::Cast(Box::new(self.expr(expr)),self.type_(ty)),
            sr::Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<sr::Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.expr(expr));
                }
                sr::Expr::AnonTuple(new_exprs)
            },
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
            sr::Stat::Let(ident,ty,expr) => {
                let new_ty = if let Some(ty) = ty {
                    Some(self.type_(ty))
                }
                else {
                    None
                };
                let new_expr = self.expr(expr);
                if let Some(ty) = &new_ty {
                    self.locals.insert(ident.clone().to_string(),ty.clone());
                }
                else {
                    let ty = infer_expr_type(self.module,&mut self.anon_tuple_structs,expr);
                    self.locals.insert(ident.clone().to_string(),ty);
                }
                sr::Stat::Let(ident.clone(),new_ty,Box::new(new_expr))
            },
            sr::Stat::Expr(expr) => sr::Stat::Expr(Box::new(self.expr(expr))),
        }
    }

    fn type_(&mut self,ty: &sr::Type) -> sr::Type {
        match ty {
            sr::Type::Inferred => sr::Type::Inferred,
            sr::Type::Boolean => sr::Type::Boolean,
            sr::Type::Integer => sr::Type::Integer,
            sr::Type::Float => sr::Type::Float,
            sr::Type::Void => sr::Type::Void,
            sr::Type::Base(base_type) => sr::Type::Base(base_type.clone()),
            sr::Type::Ident(ident) => {
                if self.module.structs.contains_key(ident) {
                    sr::Type::Struct(ident.clone())
                }
                else if self.module.anon_tuple_structs.contains_key(ident) {
                    sr::Type::Struct(ident.clone())
                }
                else if self.module.enums.contains_key(ident) {
                    sr::Type::Enum(ident.clone())
                }
                else if let Some(extern_vertex) = &self.extern_vertex {
                    if ident == extern_vertex {
                        sr::Type::Struct(ident.clone())
                    }
                    else {
                        panic!("unknown type {}",ident);
                    }
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },
            sr::Type::Struct(ident) => sr::Type::Struct(ident.clone()),
            sr::Type::Enum(ident) => sr::Type::Enum(ident.clone()),
            sr::Type::Array(ty,expr) => sr::Type::Array(Box::new(self.type_(ty)),Box::new(self.expr(expr))),
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
            sr::Pat::Array(pats) => {
                let mut new_pats: Vec<sr::Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.pat(pat));
                }
                sr::Pat::Array(new_pats)
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

pub fn resolve_idents(module: &sr::Module,extern_vertex: Option<String>) -> sr::Module {
    let mut resolver = Resolver {
        module,
        extern_vertex,
        locals: HashMap::new(),
        params: HashMap::new(),
        anon_tuple_structs: HashMap::new(),
    };

    let mut new_functions: HashMap<String,(Vec<(String,sr::Type)>,sr::Type,sr::Block)> = HashMap::new();
    for (ident,(params,return_type,block)) in &module.functions {
        let mut new_params: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in params {
            let new_type = resolver.type_(ty);
            new_params.push((ident.clone(),new_type.clone()));
            resolver.params.insert(ident.clone(),new_type);
        }
        let new_return_type = resolver.type_(return_type);
        let new_block = resolver.block(block);
        resolver.params.clear();
        new_functions.insert(ident.clone(),(new_params,new_return_type,new_block));
    }

    let mut new_structs: HashMap<String,Vec<(String,sr::Type)>> = HashMap::new();
    for (ident,fields) in &module.structs {
        let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in fields {
            new_fields.push((ident.clone(),resolver.type_(ty)));
        }
        new_structs.insert(ident.clone(),new_fields);
    }

    let mut new_anon_tuple_structs: HashMap<String,Vec<(String,sr::Type)>> = HashMap::new();
    for (ident,fields) in &module.anon_tuple_structs {
        let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
        for (ident,ty) in fields {
            new_fields.push((ident.clone(),resolver.type_(ty)));
        }
        new_anon_tuple_structs.insert(ident.clone(),new_fields);
    }
    for (ident,fields) in &resolver.anon_tuple_structs {
        new_anon_tuple_structs.insert(ident.clone(),fields.clone());
    }

    let mut new_enums: HashMap<String,Vec<sr::Variant>> = HashMap::new();
    for (ident,variants) in &module.enums {
        let mut new_variants: Vec<sr::Variant> = Vec::new();
        for variant in variants {
            new_variants.push(match variant {
                sr::Variant::Naked(ident) => sr::Variant::Naked(ident.clone()),
                sr::Variant::Tuple(ident,types) => {
                    let mut new_types: Vec<sr::Type> = Vec::new();
                    for ty in types {
                        new_types.push(resolver.type_(ty));
                    }
                    sr::Variant::Tuple(ident.clone(),new_types)
                },
                sr::Variant::Struct(ident,fields) => {
                    let mut new_fields: Vec<(String,sr::Type)> = Vec::new();
                    for (ident,ty) in fields {
                        new_fields.push((ident.clone(),resolver.type_(ty)));
                    }
                    sr::Variant::Struct(ident.clone(),new_fields)
                },
            });
        }
        new_enums.insert(ident.clone(),new_variants);
    }

    let mut new_consts: HashMap<String,(sr::Type,sr::Expr)> = HashMap::new();
    for (ident,(ty,expr)) in &module.consts {
        new_consts.insert(ident.clone(),(resolver.type_(ty),resolver.expr(expr)));        
    }

    sr::Module {
        ident: module.ident.clone(),
        functions: new_functions,
        structs: new_structs,
        anon_tuple_structs: new_anon_tuple_structs,
        enums: new_enums,
        consts: new_consts,
    }
}
