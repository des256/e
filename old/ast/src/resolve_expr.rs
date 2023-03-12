use super::*;

impl Resolver {

    // resolve expr when the type is already known
    pub fn resolve_expected_expr(&mut self,expr: &Expr,expected_type: &Type) -> Expr {
        match expr {

            Expr::Boolean(value) => if let Type::Bool = expected_type {
                Expr::Boolean(*value)
            }
            else {
                panic!("{} expected, instead of boolean {}",expected_type,expr);
            },

            Expr::Integer(value) => if let Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 = expected_type {
                Expr::Integer(*value)
            }
            else {
                panic!("{} expected, instead of integer {}",expected_type,expr);
            },

            Expr::Float(value) => if let Type::F16 | Type::F32 | Type::F64 = expected_type {
                Expr::Float(*value)
            }
            else {
                panic!("{} expected, instead of float {}",expected_type,expr);
            },

            Expr::Array(exprs) => if let Type::Array(expected_type,expected_expr) = expected_type {

                self.push_context("array".to_string());

                // TODO: test exprs.len() vs. evaluated expected_expr, probably move this evaluation elsewhere

                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    let new_expr = self.resolve_expected_expr(expr,expected_type);
                    new_exprs.push(new_expr);
                }

                self.pop_context();

                Expr::Array(new_exprs)
            }
            else {
                panic!("{} expected, instead of array {}",expected_type,expr);
            },

            Expr::Cloned(item_expr,count_expr) => if let Type::Array(expected_type,expected_expr) = expected_type {

                self.push_context(format!("cloned element for {}",expected_type));

                let new_item_expr = self.resolve_expected_expr(item_expr,expected_type);
                let new_count_expr = self.resolve_expr(count_expr);

                self.pop_context();

                Expr::Cloned(Box::new(new_item_expr),Box::new(new_count_expr))
            }
            else {
                panic!("{} expected, instead of cloned element {}",expected_type,expr);
            },

            Expr::Index(array_expr,index_expr) => {

                self.push_context(format!("array index to {}",expected_type));

                let new_array_expr = self.resolve_expr(array_expr);
                // TODO: verify this is indeed an Expr::Array
                // TODO: verify the element type matches expected_type
                let new_index_expr = self.resolve_expr(index_expr);

                self.pop_context();

                Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr))
            },

            Expr::Cast(expr,type_) => {

                self.push_context(format!("cast to {}",expected_type));

                let new_expr = self.resolve_expr(expr);
                let new_type = self.resolve_expected_type(type_,expected_type);
                // TODO: maybe verify the cast is valid

                self.pop_context();

                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },

            Expr::AnonTuple(exprs) => if let Type::Struct(struct_ident) = expected_type {

                if !self.module.anon_tuple_structs.contains_key(struct_ident) {
                    panic!("{} expected instead of anonymous tuple literal {}",expected_type,expr);
                }
                let struct_ = self.module.anon_tuple_structs[struct_ident].clone();

                self.push_context(format!("anonymous tuple literal {}",expected_type));

                if exprs.len() != struct_.fields.len() {
                    panic!("anonymous tuple literal {} should have {} elements",expr,struct_.fields.len());
                }

                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..struct_.fields.len() {
                    let new_expr = self.resolve_expected_expr(&exprs[i],&struct_.fields[i].1);
                    new_exprs.push(new_expr);
                }

                self.log_change(format!("converted anonymous tuple literal {} to struct literal",expr));

                self.pop_context();

                Expr::Struct(struct_ident.clone(),new_exprs)
            }
            else {
                panic!("{} expected instead of anonymous tuple literal {}",expected_type,expr);
            },

            Expr::Unary(op,expr) => {

                match op {
                    UnaryOp::Neg => if let Type::Integer | Type::Float | Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::F16 | Type::F32 | Type::F64 | Type::UnknownStructTupleEnumAlias(_) | Type::Struct(_) | Type::Tuple(_) | Type::Alias(_) = expected_type { } else {
                        panic!("{} expected instead of {}",expected_type,expr);
                    },
                    UnaryOp::Not => if let Type::Bool | Type::UnknownStructTupleEnumAlias(_) | Type::Struct(_) | Type::Tuple(_) | Type::Alias(_) = expected_type { } else {
                        panic!("{} expected instead of boolean result of !",expected_type);
                    },
                }

                self.push_context(format!("unary {} to {}",op,expected_type));

                let new_expr = self.resolve_expected_expr(expr,expected_type);

                self.pop_context();

                Expr::Unary(op.clone(),Box::new(new_expr))
            },

            Expr::Binary(expr1,op,expr2) => {

                match op {
                    BinaryOp::Mul | BinaryOp::Div | BinaryOp::Mod | BinaryOp::Add | BinaryOp::Sub | 
                    BinaryOp::Shl | BinaryOp::Shr | BinaryOp::And | BinaryOp::Or | BinaryOp::Xor |
                    BinaryOp::AddAssign | BinaryOp::SubAssign | BinaryOp::MulAssign | BinaryOp::DivAssign | BinaryOp::ModAssign |
                    BinaryOp::AndAssign | BinaryOp::OrAssign | BinaryOp::XorAssign | BinaryOp::ShlAssign | BinaryOp::ShrAssign => if let Type::Integer | Type::Float | Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 | Type::UnknownStructTupleEnumAlias(_) | Type::Struct(_) | Type::Tuple(_) | Type::Alias(_) = expected_type { } else {
                        panic!("{} expected instead of {}",expected_type,expr);
                    },
                    BinaryOp::Eq | BinaryOp::NotEq |
                    BinaryOp::Greater | BinaryOp::Less | BinaryOp::GreaterEq | BinaryOp::LessEq |
                    BinaryOp::LogAnd | BinaryOp::LogOr => if let Type::Bool = expected_type { } else {
                        panic!("{} expected instead of boolean {}",expected_type,expr);
                    },
                    BinaryOp::Assign => { },
                }

                self.push_context(format!("binary {} to {}",op,expected_type));

                // TODO: you can actually figure out expected_type for expr1 and expr2 as well
                let new_expr1 = self.resolve_expr(expr1);
                let new_expr2 = self.resolve_expr(expr2);

                self.pop_context();

                Expr::Binary(Box::new(new_expr1),op.clone(),Box::new(new_expr2))
            },

            Expr::Continue => Expr::Continue,

            Expr::Break(expr) => {

                self.push_context("break".to_string());

                let new_expr = if let Some(expr) = expr {
                    Some(Box::new(self.resolve_expr(expr)))
                }
                else {
                    None
                };

                self.pop_context();

                Expr::Break(new_expr)
            },

            Expr::Return(expr) => {

                self.push_context("return".to_string());

                let new_expr = if let Some(expr) = expr {
                    Some(Box::new(self.resolve_expr(expr)))
                }
                else {
                    None
                };

                self.pop_context();

                Expr::Break(new_expr)
            },

            // TODO:
            Expr::Block(block) => expr.clone(),
            Expr::If(cond_expr,block,else_expr) => expr.clone(),
            Expr::While(cond_expr,block) => expr.clone(),
            Expr::Loop(block) => expr.clone(),
            Expr::IfLet(pats,test_expr,block,else_expr) => expr.clone(),
            Expr::For(pats,range,block) => expr.clone(),
            Expr::WhileLet(pats,test_expr,block) => expr.clone(),
            Expr::Match(cond_expr,arms) => expr.clone(),
            Expr::UnknownLocalConst(ident) => expr.clone(),
            Expr::Local(ident) => expr.clone(),
            Expr::Const(ident) => expr.clone(),
            Expr::UnknownTupleFunctionCall(ident,exprs) => expr.clone(),
            Expr::Tuple(tuple_ident,exprs) => expr.clone(),
            Expr::FunctionCall(function_ident,exprs) => expr.clone(),
            Expr::UnknownStruct(ident,fields) => expr.clone(),
            Expr::Struct(struct_ident,exprs) => expr.clone(),
            Expr::UnknownVariant(enum_ident,variant_ident,variant_expr) => expr.clone(),
            Expr::Variant(enum_ident,index,variant_expr) => expr.clone(),
            Expr::UnknownMethodCall(from_expr,ident,exprs) => expr.clone(),
            Expr::MethodCall(from_expr,ident,exprs) => expr.clone(),

            Expr::UnknownField(struct_expr,ident) => {
                // TODO: somehow figure out what the struct is this field belongs to
                expr.clone()
            },

            Expr::Field(struct_expr,struct_ident,index) => {
                // TODO: verify struct_ident.index is indeed a expected_type
                expr.clone()
            },

            Expr::UnknownTupleIndex(tuple_expr,index) => {
                // TODO: somehow figure out what the tuple is this index belongs to
                expr.clone()
            },

            Expr::TupleIndex(tuple_expr,tuple_ident,index) => {
                // TODO: verify tuple_ident.index is indeed a expected_type
                expr.clone()                
            },
        }
    }

    // resolve expr when the type is not known
    pub fn resolve_expr(&mut self,expr: &Expr) -> Expr {
        match expr {

            Expr::Boolean(value) => Expr::Boolean(*value),

            Expr::Integer(value) => Expr::Integer(*value),

            Expr::Float(value) => Expr::Float(*value),

            Expr::Array(exprs) => {

                self.push_context("array".to_string());

                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    let new_expr = self.resolve_expr(expr);
                    new_exprs.push(new_expr);
                }

                self.pop_context();

                Expr::Array(new_exprs)
            },

            Expr::Cloned(item_expr,count_expr) => {

                self.push_context("cloned element".to_string());

                let new_item_expr = self.resolve_expr(item_expr);
                let new_count_expr = self.resolve_expr(count_expr);

                self.pop_context();

                Expr::Cloned(Box::new(new_item_expr),Box::new(new_count_expr))
            },
 
            Expr::Index(array_expr,index_expr) => {
                self.push_context(format!("array index"));

                let new_array_expr = self.resolve_expr(array_expr);
                // TODO: verify this is indeed an Expr::Array
                // TODO: verify the element type matches expected_type
                let new_index_expr = self.resolve_expr(index_expr);

                self.pop_context();

                Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr))
            },
 
            Expr::Cast(expr,type_) => {

                self.push_context(format!("cast"));

                let new_expr = self.resolve_expr(expr);
                let new_type = self.resolve_type(type_);
                // TODO: maybe verify the cast is valid

                self.pop_context();

                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },
 
            Expr::AnonTuple(exprs) => {

                self.push_context("anonymous tuple literal".to_string());

                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    let new_expr = self.resolve_expr(expr);
                    new_exprs.push(new_expr);
                }

                self.pop_context();

                // TODO: convert to anon_tuple_struct here

                Expr::AnonTuple(new_exprs)
            },

            Expr::Unary(op,expr) => {

                self.push_context(format!("unary {}",op));

                let new_expr = self.resolve_expr(expr);

                self.pop_context();

                Expr::Unary(op.clone(),Box::new(new_expr))
            },

            Expr::Binary(expr1,op,expr2) => {

                self.push_context(format!("binary {}",op));

                let new_expr1 = self.resolve_expr(expr1);
                let new_expr2 = self.resolve_expr(expr2);

                self.pop_context();

                Expr::Binary(Box::new(new_expr1),op.clone(),Box::new(new_expr2))
            },

            Expr::Continue => Expr::Continue,

            Expr::Break(expr) => {

                self.push_context("break".to_string());

                let new_expr = if let Some(expr) = expr {
                    Some(Box::new(self.resolve_expr(expr)))
                }
                else {
                    None
                };

                self.pop_context();

                Expr::Break(new_expr)
            },

            Expr::Return(expr) => {

                self.push_context("return".to_string());

                let new_expr = if let Some(expr) = expr {
                    Some(Box::new(self.resolve_expr(expr)))
                }
                else {
                    None
                };

                self.pop_context();

                Expr::Break(new_expr)
            },

            // TODO:
            Expr::Block(block) => expr.clone(),
            Expr::If(cond_expr,block,else_expr) => expr.clone(),
            Expr::While(cond_expr,block) => expr.clone(),
            Expr::Loop(block) => expr.clone(),
            Expr::IfLet(pats,test_expr,block,else_expr) => expr.clone(),
            Expr::For(pats,range,block) => expr.clone(),
            Expr::WhileLet(pats,test_expr,block) => expr.clone(),
            Expr::Match(cond_expr,arms) => expr.clone(),
            Expr::UnknownLocalConst(ident) => expr.clone(),
            Expr::Local(ident) => expr.clone(),
            Expr::Const(ident) => expr.clone(),
            Expr::UnknownTupleFunctionCall(ident,exprs) => expr.clone(),
            Expr::Tuple(tuple_ident,exprs) => expr.clone(),
            Expr::FunctionCall(function_ident,exprs) => expr.clone(),
            Expr::UnknownStruct(ident,fields) => expr.clone(),
            Expr::Struct(struct_ident,exprs) => expr.clone(),
            Expr::UnknownVariant(enum_ident,variant_ident,variant_expr) => expr.clone(),
            Expr::Variant(enum_ident,index,variant_expr) => expr.clone(),
            Expr::UnknownMethodCall(from_expr,ident,exprs) => expr.clone(),
            Expr::MethodCall(from_expr,ident,exprs) => expr.clone(),

            Expr::UnknownField(struct_expr,ident) => {
                // TODO: somehow figure out what the struct is this field belongs to
                expr.clone()
            },

            Expr::Field(struct_expr,struct_ident,index) => {
                expr.clone()
            },

            Expr::UnknownTupleIndex(tuple_expr,index) => {
                // TODO: somehow figure out what the tuple is this index belongs to
                expr.clone()
            },

            Expr::TupleIndex(tuple_expr,tuple_ident,index) => {
                expr.clone()                
            },
        }
    }
}
