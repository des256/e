use {
    crate::*,
    std::collections::HashMap,
};

pub struct Finder<'module> {
    pub module: &'module sr::Module,
    pub locals: HashMap<String,sr::Type>,
    pub params: HashMap<String,sr::Type>,
}

impl<'module> Finder<'module> {

    pub fn tightest(&self,type1: &sr::Type,type2: &sr::Type) -> Option<sr::Type> {
        // if types are exactly the same, return that
        if *type1 == *type2 {
            Some(type1.clone())
        }

        // otherwise see how they match
        else {
            match type1 {

                // inferred always matches with the other one
                sr::Type::Inferred => Some(type2.clone()),

                // check integer
                sr::Type::Integer => match type2 {

                    // integer becomes float if other is float
                    sr::Type::Float => Some(sr::Type::Float),

                    // integer becomes base type if other is compatible base type
                    sr::Type::Base(base_type) => match base_type {
                        sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                        sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                        sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type2.clone()),
                        _ => None,
                    },

                    // anything else is not compatible
                    _ => None,
                },

                // check float
                sr::Type::Float => match type2 {

                    // float remains float when other is float, integer or inferred
                    sr::Type::Integer | sr::Type::Float => Some(sr::Type::Float),

                    // float becomes base type if other is compatible base type
                    sr::Type::Base(base_type) => match base_type {
                        sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type2.clone()),
                        _ => None,
                    },

                    // anything else is not compatible
                    _ => None,
                },

                // check other side
                _ => match type2 {

                    // inferred always matches with the other one
                    sr::Type::Inferred => Some(type1.clone()),

                    // check integer
                    sr::Type::Integer => match type1 {

                        // integer becomes float if other is float
                        sr::Type::Float => Some(sr::Type::Float),

                        // integer becomes base type if other is compatible base type
                        sr::Type::Base(base_type) => match base_type {
                            sr::BaseType::U8 | sr::BaseType::U16 | sr::BaseType::U32 | sr::BaseType::U64 |
                            sr::BaseType::I8 | sr::BaseType::I16 | sr::BaseType::I32 | sr::BaseType::I64 |
                            sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type1.clone()),
                            _ => None,
                        },

                        // anything else is not compatible
                        _ => None,
                    },

                    // check float
                    sr::Type::Float => match type1 {

                        // float remains float when other is float, integer or inferred
                        sr::Type::Integer | sr::Type::Float => Some(sr::Type::Float),

                        // float becomes base type if other is compatible base type
                        sr::Type::Base(base_type) => match base_type {
                            sr::BaseType::F16 | sr::BaseType::F32 | sr::BaseType::F64 => Some(type1.clone()),
                            _ => None,
                        },

                        // anything else is not compatible
                        _ => None,
                    },

                    // anything else is not compatible
                    _ => None,
                }
            }
        }    
    }

    pub fn process_stat(&mut self,stat: &sr::Stat) {
        match stat {
            sr::Stat::Let(pat,ty,expr) => {
                let ident = if let sr::Pat::Ident(ident) = &pat {
                    ident
                }
                else {
                    panic!("only single identifier pattern supported in let-statement");
                };
                if let Some(ty) = &ty {
                    self.locals.insert(ident.to_string(),*ty.clone());
                }
                else {
                    panic!("TODO: find type of expression in let-statement");
                }    
            },
            _ => { },
        }
    }

    pub fn block_type(&mut self,block: &sr::Block) -> sr::Type {
        let local_frame = self.locals.clone();
        for stat in &block.stats {
            self.process_stat(stat);
        }
        let result = if let Some(expr) = &block.expr {
            self.expr_type(expr)
        }
        else {
            sr::Type::Void
        };
        self.locals = local_frame;
        result
    }

    pub fn expr_type(&mut self,expr: &sr::Expr) -> sr::Type {
        match expr {
            sr::Expr::Boolean(_) => sr::Type::Base(sr::BaseType::Bool),
            sr::Expr::Integer(_) => sr::Type::Integer,
            sr::Expr::Float(_) => sr::Type::Float,
            sr::Expr::Base(base_type,_) => sr::Type::Base(base_type.clone()),
            sr::Expr::Local(_,ty) => ty.clone(),
            sr::Expr::Param(_,ty) => ty.clone(),
            sr::Expr::Const(_,ty) => ty.clone(),
            sr::Expr::Array(exprs) => {
                let mut result = sr::Type::Inferred;
                for expr in exprs {
                    let other_ty = self.expr_type(expr);
                    if let Some(ty) = self.tightest(&result,&other_ty) {
                        result = ty;
                    }
                }
                result
            },
            sr::Expr::Cloned(expr,_) => self.expr_type(expr),
            sr::Expr::Struct(ident,_) => sr::Type::Struct(ident.clone()),
            sr::Expr::Tuple(ident,_) => sr::Type::Tuple(ident.clone()),
            sr::Expr::AnonTuple(exprs) => {
                let mut types: Vec<sr::Type> = Vec::new();
                for expr in exprs.iter() {
                    types.push(self.expr_type(expr));
                }
                sr::Type::AnonTuple(types)
            },
            sr::Expr::Variant(ident,_) => sr::Type::Enum(ident.clone()),
            sr::Expr::Call(_,_,ty) => ty.clone(),
            sr::Expr::Field(expr,ident,ty) => {
                if let sr::Type::Struct(struct_ident) = self.expr_type(expr) {
                    let mut result: Option<sr::Type> = None;
                    for (param_ident,ty) in self.module.structs[&struct_ident].iter() {
                        if *param_ident == *ident {
                            result = Some(ty.clone());
                            break;
                        }
                    }
                    result.unwrap() // shouldn't happen
                }
                else {
                    sr::Type::Void // shouldn't happen
                }
            },
            sr::Expr::TupleIndex(expr,index,ty) => {
                if let sr::Type::Tuple(tuple_ident) = self.expr_type(expr) {
                    self.module.tuples[&tuple_ident][*index as usize].clone()
                }
                else {
                    sr::Type::Void // shouldn't happen
                }
            },
            sr::Expr::Index(expr,_,ty) => {
                if let sr::Type::Array(ty,_) = self.expr_type(expr) {
                    *ty
                }
                else {
                    sr::Type::Void // shouldn't happen
                }
            },
            sr::Expr::Cast(_,ty) => ty.clone(),
            sr::Expr::Neg(expr) => self.expr_type(expr),
            sr::Expr::Not(expr) => self.expr_type(expr),
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
            sr::Expr::AddAssign(expr,expr2) |
            sr::Expr::SubAssign(expr,expr2) |
            sr::Expr::MulAssign(expr,expr2) |
            sr::Expr::DivAssign(expr,expr2) |
            sr::Expr::ModAssign(expr,expr2) |
            sr::Expr::AndAssign(expr,expr2) |
            sr::Expr::OrAssign(expr,expr2) |
            sr::Expr::XorAssign(expr,expr2) |
            sr::Expr::ShlAssign(expr,expr2) |
            sr::Expr::ShrAssign(expr,expr2) => {
                let ty = self.expr_type(expr);
                let ty2 = self.expr_type(expr2);
                if let Some(ty) = self.tightest(&ty,&ty2) {
                    ty
                }
                else {
                    sr::Type::Void // this shouldn't happen
                }
            },
            sr::Expr::Assign(_,expr2) => self.expr_type(expr2),
            sr::Expr::Continue => sr::Type::Void,
            sr::Expr::Break(expr) => {
                if let Some(expr) = expr {
                    self.expr_type(expr)
                }
                else {
                    sr::Type::Void
                }
            },
            sr::Expr::Return(expr) => {
                if let Some(expr) = expr {
                    self.expr_type(expr)
                }
                else {
                    sr::Type::Void
                }
            },
            sr::Expr::Block(block) => self.block_type(block),
            sr::Expr::If(_,block,else_expr) | sr::Expr::IfLet(_,_,block,else_expr) => {
                let block_ty = self.block_type(block);
                if let Some(else_expr) = else_expr {
                    let else_ty = self.expr_type(else_expr);
                    if let Some(ty) = self.tightest(&block_ty,&else_ty) {
                        ty
                    }
                    else {
                        sr::Type::Void // shouldn't happen
                    }
                }
                else {
                    block_ty
                }
            },
            sr::Expr::Loop(block) => {
                // TODO: only look at break statements
                self.block_type(block);
                sr::Type::Void
            },
            sr::Expr::For(_,_,_) => {            
                // TODO: find tightest type of any Expr::Break in block
                sr::Type::Void
            },
            sr::Expr::While(_,_) | sr::Expr::WhileLet(_,_,_) => {
                // TODO: find tightest type of any Expr::Break in block
                sr::Type::Void
            },
            sr::Expr::Match(_,arms) => {
                // TODO: find tightest type of all arm expr
                let mut result = sr::Type::Inferred;
                for (_,_,expr) in arms {
                    let other_ty = self.expr_type(expr);
                    if let Some(ty) = self.tightest(&result,&other_ty) {
                        result = ty;
                    }
                }
                result
            },
        }    
    }
}

pub fn find_expr_type(module: &sr::Module,expr: &sr::Expr) -> sr::Type {
    let mut finder = Finder {
        module,
        locals: HashMap::new(),
        params: HashMap::new(),
    };
    finder.expr_type(expr)
}