// resolve const references

use {
    crate::*,
    std::collections::HashMap,
};

struct Resolver {
    pub consts: HashMap<String,(Type,Expr)>,
}

impl Resolver {

    fn block(&self,block: &mut Block) {
        for stat in block.stats.iter_mut() {
            self.stat(stat);
        }
        if let Some(expr) = block.expr.as_mut() {
            self.expr(expr);
        }
    }

    fn expr(&self,expr: &mut Expr) {

        match expr {

            Expr::Boolean(_) |
            Expr::Integer(_) |
            Expr::Float(_) |
            Expr::Const(_) |
            Expr::Continue => { },

            Expr::Base(_,fields) |
            Expr::UnknownStruct(_,fields) => {
                for field in fields.iter_mut() {
                    self.expr(&mut field.1);
                }
            },

            Expr::UnknownIdent(ident) => {
                if self.consts.contains_key(ident) {
                    *expr = Expr::Const(ident.clone())
                }
            },

            Expr::Array(exprs) |
            Expr::AnonTuple(exprs) |
            Expr::UnknownCallOrTuple(_,exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },

            Expr::UnknownVariant(_,variantexpr) => {
                match variantexpr {
                    VariantExpr::Naked(_) => { },
                    VariantExpr::Tuple(_,exprs) => {
                        for expr in exprs.iter_mut() {
                            self.expr(expr);
                        }
                    },
                    VariantExpr::Struct(_,fields) => {
                        for field in fields.iter_mut() {
                            self.expr(&mut field.1);
                        }
                    },
                }
            },

            Expr::Field(expr,_) |
            Expr::Neg(expr) |
            Expr::Not(expr) => self.expr(expr),

            Expr::Cast(expr,type_) => {
                self.expr(expr);
                self.type_(type_);
            },

            Expr::Cloned(expr,expr2) |
            Expr::Index(expr,expr2) |
            Expr::Mul(expr,expr2) |
            Expr::Div(expr,expr2) |
            Expr::Mod(expr,expr2) |
            Expr::Add(expr,expr2) |
            Expr::Sub(expr,expr2) |
            Expr::Shl(expr,expr2) |
            Expr::Shr(expr,expr2) |
            Expr::And(expr,expr2) |
            Expr::Or(expr,expr2) |
            Expr::Xor(expr,expr2) |
            Expr::Eq(expr,expr2) |
            Expr::NotEq(expr,expr2) |
            Expr::Greater(expr,expr2) |
            Expr::Less(expr,expr2) |
            Expr::GreaterEq(expr,expr2) |
            Expr::LessEq(expr,expr2) |
            Expr::LogAnd(expr,expr2) |
            Expr::LogOr(expr,expr2) |
            Expr::Assign(expr,expr2) |
            Expr::AddAssign(expr,expr2) |
            Expr::SubAssign(expr,expr2) |
            Expr::MulAssign(expr,expr2) |
            Expr::DivAssign(expr,expr2) |
            Expr::ModAssign(expr,expr2) |
            Expr::AndAssign(expr,expr2) |
            Expr::OrAssign(expr,expr2) |
            Expr::XorAssign(expr,expr2) |
            Expr::ShlAssign(expr,expr2) |
            Expr::ShrAssign(expr,expr2) => {
                self.expr(expr);
                self.expr(expr2);
            },

            Expr::Break(expr) |
            Expr::Return(expr) => if let Some(expr) = expr {
                self.expr(expr);
            },

            Expr::Block(block) => self.block(block),

            Expr::If(expr,block,else_expr) => {
                self.expr(expr);
                self.block(block);
                if let Some(else_expr) = else_expr {
                    self.expr(else_expr);
                }
            },
            Expr::IfLet(pats,expr,block,else_expr) => {
                for pat in pats.iter_mut() {
                    self.pat(pat);
                }
                self.expr(expr);
                self.block(block);
                if let Some(else_expr) = else_expr {
                    self.expr(else_expr);
                }
            },
            Expr::Loop(block) => self.block(block),
            Expr::For(pats,range,block) => {
                for pat in pats.iter_mut() {
                    self.pat(pat);
                }
                match range {
                    Range::Only(expr) => self.expr(expr),
                    Range::FromTo(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    Range::FromToIncl(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    Range::From(expr) => self.expr(expr),
                    Range::To(expr) => self.expr(expr),
                    Range::ToIncl(expr) => self.expr(expr),
                    Range::All => { },
                };
                self.block(block);
            },
            Expr::While(expr,block) => {
                self.expr(expr);
                self.block(block);
            },
            Expr::WhileLet(pats,expr,block) => {
                for pat in pats {
                    self.pat(pat);
                }
                self.expr(expr);
                self.block(block);
            },
            Expr::Match(expr,arms) => {
                self.expr(expr);
                for (pats,if_expr,expr) in arms {
                    for pat in pats {
                        self.pat(pat);
                    }
                    if let Some(if_expr) = if_expr {
                        self.expr(if_expr);
                    }
                    self.expr(expr);
                }
            },
        }
    }

    fn stat(&self,stat: &mut Stat) {
        match stat {
            Stat::Let(pat,type_,expr) => {
                self.pat(pat);
                self.type_(type_);
                self.expr(expr);
            },
            Stat::LetIdent(_,expr) |
            Stat::Expr(expr) => self.expr(expr),
        }
    }

    fn type_(&self,type_: &mut Type) {
        match type_ {
            Type::Inferred |
            Type::Void |
            Type::Base(_) |
            Type::UnknownIdent(_) => { },
            Type::Array(type_,expr) => {
                self.type_(type_);
                self.expr(expr);
            },
        }
    }

    fn pat(&self,pat: &mut Pat) {
        match pat {
            Pat::Wildcard |
            Pat::Rest |
            Pat::Boolean(_) |
            Pat::Integer(_) |
            Pat::Float(_) |
            Pat::Const(_) => { },
            Pat::Ident(ident) => {
                if self.consts.contains_key(ident) {
                    *pat = Pat::Const(ident.clone());
                }
            },
            Pat::UnknownStruct(_,identpats) => {
                for identpat in identpats.iter_mut() {
                    if let IdentPat::IdentPat(_,pat) = identpat {
                        self.pat(pat);
                    }
                }
            },
            Pat::Array(pats) |
            Pat::AnonTuple(pats) => {
                for pat in pats.iter_mut() {
                    self.pat(pat);
                }
            },
            Pat::UnknownVariant(_,variantpat) => {
                match variantpat {
                    VariantPat::Naked(_) => { },
                    VariantPat::Tuple(_,pats) => {
                        for pat in pats.iter_mut() {
                            self.pat(pat);
                        }
                    },
                    VariantPat::Struct(_,identpats) => {
                        for identpat in identpats.iter_mut() {
                            if let IdentPat::IdentPat(_,pat) = identpat {
                                self.pat(pat);
                            }
                        }
                    },
                };
            },
            Pat::Range(pat,pat2) => {
                self.pat(pat);
                self.pat(pat2);
            },
        }
    }
}

pub fn resolve_consts(module: &mut Module) {
    let resolver = Resolver {
        consts: module.consts.clone(),
    };

    for (_,(params,return_type,block)) in module.functions.iter_mut() {
        for (_,type_) in params.iter_mut() {
            resolver.type_(type_);
        }
        resolver.type_(return_type);
        resolver.block(block);
    }
    for (_,fields) in module.structs.iter_mut() {
        for (_,type_) in fields.iter_mut() {
            resolver.type_(type_);
        }
    }
    for (_,fields) in module.anon_tuple_structs.iter_mut() {
        for (_,type_) in fields.iter_mut() {
            resolver.type_(type_);
        }
    }
    for (_,(variants,_)) in module.enums.iter_mut() {
        for variant in variants.iter_mut() {
            match variant {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for type_ in types.iter_mut() {
                        resolver.type_(type_);
                    }
                },
                Variant::Struct(_,fields) => {
                    for (_,type_) in fields.iter_mut() {
                        resolver.type_(type_);
                    }
                },
            }
        }
    }
    for (_,(type_,expr)) in module.consts.iter_mut() {
        resolver.type_(type_);
        resolver.expr(expr);
    }
}
