use {
    super::*,
    super::ast::*,
    std::{
        collections::HashMap,
        cmp::PartialEq,
    },
};

// - remove aliases
// - remove anonymous tuples

impl PartialEq for Type {
    fn eq(&self,other: &Type) -> bool {
        match (self,other) {
            (Type::Inferred,_) => true,
            (_,Type::Inferred) => true,
            (Type::Void,Type::Void) => true,
            (Type::Bool,Type::Bool) => true,
            (Type::U8,Type::U8) => true,
            (Type::I8,Type::I8) => true,
            (Type::U16,Type::U16) => true,
            (Type::I16,Type::I16) => true,
            (Type::U32,Type::U32) => true,
            (Type::I32,Type::I32) => true,
            (Type::U64,Type::U64) => true,
            (Type::I64,Type::I64) => true,
            (Type::F16,Type::F16) => true,
            (Type::F32,Type::F32) => true,
            (Type::F64,Type::F64) => true,
            (Type::Vec2Bool,Type::Vec2Bool) => true,
            (Type::Vec2U8,Type::Vec2U8) => true,
            (Type::Vec2I8,Type::Vec2I8) => true,
            (Type::Vec2U16,Type::Vec2U16) => true,
            (Type::Vec2I16,Type::Vec2I16) => true,
            (Type::Vec2U32,Type::Vec2U32) => true,
            (Type::Vec2I32,Type::Vec2I32) => true,
            (Type::Vec2U64,Type::Vec2U64) => true,
            (Type::Vec2I64,Type::Vec2I64) => true,
            (Type::Vec2F16,Type::Vec2F16) => true,
            (Type::Vec2F32,Type::Vec2F32) => true,
            (Type::Vec2F64,Type::Vec2F64) => true,
            (Type::Vec3Bool,Type::Vec3Bool) => true,
            (Type::Vec3U8,Type::Vec3U8) => true,
            (Type::Vec3I8,Type::Vec3I8) => true,
            (Type::Vec3U16,Type::Vec3U16) => true,
            (Type::Vec3I16,Type::Vec3I16) => true,
            (Type::Vec3U32,Type::Vec3U32) => true,
            (Type::Vec3I32,Type::Vec3I32) => true,
            (Type::Vec3U64,Type::Vec3U64) => true,
            (Type::Vec3I64,Type::Vec3I64) => true,
            (Type::Vec3F16,Type::Vec3F16) => true,
            (Type::Vec3F32,Type::Vec3F32) => true,
            (Type::Vec3F64,Type::Vec3F64) => true,
            (Type::Vec4Bool,Type::Vec4Bool) => true,
            (Type::Vec4U8,Type::Vec4U8) => true,
            (Type::Vec4I8,Type::Vec4I8) => true,
            (Type::Vec4U16,Type::Vec4U16) => true,
            (Type::Vec4I16,Type::Vec4I16) => true,
            (Type::Vec4U32,Type::Vec4U32) => true,
            (Type::Vec4I32,Type::Vec4I32) => true,
            (Type::Vec4U64,Type::Vec4U64) => true,
            (Type::Vec4I64,Type::Vec4I64) => true,
            (Type::Vec4F16,Type::Vec4F16) => true,
            (Type::Vec4F32,Type::Vec4F32) => true,
            (Type::Vec4F64,Type::Vec4F64) => true,
            (Type::Mat2x2F32,Type::Mat2x2F32) => true,
            (Type::Mat2x2F64,Type::Mat2x2F64) => true,
            (Type::Mat2x3F32,Type::Mat2x3F32) => true,
            (Type::Mat2x3F64,Type::Mat2x3F64) => true,
            (Type::Mat2x4F32,Type::Mat2x4F32) => true,
            (Type::Mat2x4F64,Type::Mat2x4F64) => true,
            (Type::Mat3x2F32,Type::Mat3x2F32) => true,
            (Type::Mat3x2F64,Type::Mat3x2F64) => true,
            (Type::Mat3x3F32,Type::Mat3x3F32) => true,
            (Type::Mat3x3F64,Type::Mat3x3F64) => true,
            (Type::Mat3x4F32,Type::Mat3x4F32) => true,
            (Type::Mat3x4F64,Type::Mat3x4F64) => true,
            (Type::Mat4x2F32,Type::Mat4x2F32) => true,
            (Type::Mat4x2F64,Type::Mat4x2F64) => true,
            (Type::Mat4x3F32,Type::Mat4x3F32) => true,
            (Type::Mat4x3F64,Type::Mat4x3F64) => true,
            (Type::Mat4x4F32,Type::Mat4x4F32) => true,
            (Type::Mat4x4F64,Type::Mat4x4F64) => true,
            (Type::AnonTuple(types1),Type::AnonTuple(types2)) => {
                if types1.len() == types2.len() {
                    for i in 0..types1.len() {
                        if types1[i] != types2[i] {
                            return false;
                        }
                    }
                    true
                }
                else {
                    false
                }
            },
            (Type::Array(type1,count1),Type::Array(type2,count2)) => (type1 == type2) && (count1 == count2),
            (Type::IdentRef(ident1),Type::IdentRef(ident2)) => ident1 == ident2,
            (Type::IdentRef(ident1),Type::Ident(ident2)) => ident1 == ident2,
            (Type::Ident(ident1),Type::IdentRef(ident2)) => ident1 == ident2,
            (Type::Ident(ident1),Type::Ident(ident2)) => ident1 == ident2,
            _ => false,
        }
    }
}

impl Context {

    fn prepare_type(&self,type_: Type) -> Result<Type,String> {

        match type_ {
            
            // convert anonymous tuple type into a struct that can be referenced by index
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(self.prepare_type(*type_)?);
                }
                let found_index: Option<usize> = None;
                for i in 0..self.anon_tuple_structs.len() {
                    if self.anon_tuple_structs[i].len() == types.len() {
                        let mut all_types_match = true;
                        for k in 0..types.len() {
                            if self.anon_tuple_structs[i][k] != new_types[k] {
                                all_types_match = false;
                                break;
                            }
                        }
                        if all_types_match {
                            found_index = Some(i);
                            break;
                        }
                    }
                }
                if let Some(index) = found_index {
                    Ok(Type::AnonTupleRef(index))
                }
                else {
                    self.anon_tuple_structs.push(new_types);
                    Ok(Type::AnonTupleRef(self.anon_tuple_structs.len() - 1))
                }
            },

            Type::Array(type_,expr) => {
                let new_type = self.prepare_type(*type_)?;
                let new_expr = self.prepare_expr(*expr,new_type.clone())?;
                Ok(Type::Array(Box::new(new_type),Box::new(new_expr)))
            },

            Type::Ident(ident) => {
                if self.alias_types.contains_key(&ident) {
                    Ok(self.alias_types[&ident].clone())
                }
                else {
                    Ok(Type::Ident(ident))
                }
            },

            _ => Ok(type_),
        }
    }

    fn prepare_pat(&self,pat: Pat) -> Result<Pat,String> {
        Ok(pat)
    }

    fn prepare_block(&self,block: Block,should_type: Type) -> Result<Block,String> {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {
                Stat::Let(pat,type_,expr) => {
                    let new_pat = self.prepare_pat(*pat.clone())?;
                    let new_type = self.prepare_type(*type_.clone())?;
                    let new_expr = self.prepare_expr(*expr.clone(),new_type.clone())?;
                    new_stats.push(Stat::Let(pat,new_expr,new_type));
                },
                Stat::Expr(expr) => {
                    let new_expr = self.prepare_expr(*expr.clone(),Type::Void)?;
                    new_stats.push(Stat::Expr(new_expr));
                },
                Stat::Local(ident,type_,expr) => {
                    // parser does not generate Stat::Local nodes
                    return Err("illegal Stat::Local".to_string());
                },
            }
        }
        let new_expr = if let Some(expr) = block.expr {
            Some(Box::new(self.prepare_expr(*expr,should_type)?))
        }
        else {
            None
        };
        Ok(Block {
            stats: new_stats,
            expr: new_expr,
        })
    }

    fn prepare_expr(&self,expr: Expr,should_type: Type) -> Result<Expr,String> {
        match expr {
            Expr::Boolean(value) => Ok(Expr::Boolean(value)),
            Expr::Integer(value) => Ok(Expr::Integer(value)),
            Expr::Float(value) => Ok(Expr::Float(value)),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.prepare_expr(expr)?);
                }
                Ok(Expr::Array(new_exprs))
            },
            Expr::Cloned(value_expr,count_expr) => {
                let new_value_expr = self.prepare_expr(*value_expr)?;
                let new_count_expr = self.prepare_expr(*count_expr)?;
                Ok(Expr::Cloned(Box::new(new_value_expr),Box::new(new_count_expr)))
            },
            Expr::Index(array_expr,index_expr) => {
                let new_array_expr = self.prepare_expr(*array_expr)?;
                let new_index_expr = self.prepare_expr(*index_expr)?;
                Ok(Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr)))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.prepare_expr(*expr)?;
                let new_type = self.prepare_type(*type_)?;
                Ok(Expr::Cast(Box::new(new_expr),Box::new(new_type)))
            },
            Expr::AnonTuple(exprs) => if let Type::AnonTupleRef(index) = should_type {
                if self.anon_tuple_structs[index].len() == exprs.len() {
                    let new_exprs: Vec<Expr> = Vec::new();
                    for i in 0..exprs.len() {
                        new_exprs.push(self.prepare_expr(exprs[i],self.anon_tuple_structs[index][i])?);
                    }
                    Ok(Expr::AnonTupleRef(index,new_exprs))
                }
                else {
                    Err(format!("{} and {} have different dimensions",expr,should_type))
                }
            }
            else {
                Err(format!("{} expected instead of {}",should_type,expr))
            },
            Expr::Unary(op,expr) => {
                let new_expr = self.prepare_expr(*expr)?;
                Ok(Expr::Unary(op,Box::new(new_expr)))
            },
            Expr::Binary(expr1,op,expr2) => {
                let new_expr1 = self.prepare_expr(*expr1)?;
                let new_expr2 = self.prepare_expr(*expr2)?;
                Ok(Expr::Binary(Box::new(new_expr1),op,Box::new(new_expr2)))
            },
            Expr::Continue => Ok(Expr::Continue),
            Expr::Break(expr) => if let Some(expr) = expr {
                Ok(Expr::Break(Some(Box::new(self.prepare_expr(*expr)?))))
            }
            else {
                Ok(Expr::Break(None))
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                Ok(Expr::Return(Some(Box::new(self.prepare_expr(*expr)?))))
            }
            else {
                Ok(Expr::Return(None))
            },
            Expr::Block(block) => Ok(Expr::Block(self.prepare_block(block)?)),
            Expr::If(cond_expr,block,else_expr) => {
                let new_cond_expr = self.prepare_expr(*cond_expr)?;
                let new_block = self.prepare_block(block)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.prepare_expr(*else_expr)?))
                }
                else {
                    None
                };
                Ok(Expr::If(Box::new(new_cond_expr),new_block,new_else_expr))
            },
            Expr::While(cond_expr,block) => {
                let new_cond_expr = self.prepare_expr(*cond_expr)?;
                let new_block = self.prepare_block(block)?;
                Ok(Expr::While(Box::new(new_cond_expr),new_block))
            },
            Expr::Loop(block) => Ok(Expr::Loop(self.prepare_block(block)?)),
        }
    }
}

pub fn prepare_module(module: &Module) -> Result<Module,String> {

    // initialize the context   
    let mut context = Context {
        stdlib: StandardLib::new(),
        alias_types: HashMap::new(),
        consts: HashMap::new(),
        structs: HashMap::new(),
        enums: HashMap::new(),
        functions: HashMap::new(),
    };

    // simplify all aliases into their proper types
    for alias in module.aliases.values() {
        let mut type_ = alias.type_.clone();
        while let Type::Ident(ident) = type_ {
            if module.aliases.contains_key(ident) {
                type_ = module.aliases[ident].type_.clone();               
            }
            else {
                break;
            }
        }
        context.alias_types.insert(alias.ident,type_);
    }

    // process const
    for const_ in module.consts.values() {
        let new_type = context.process_type(const_.type_)?;
        let new_expr = context.process_expr(const_.expr)?;
        context.consts.insert(const_.ident,Const {
            ident: const_.ident,
            type_: new_type,
            expr: new_expr,
        });
    }

    Ok(Module {
        ident: module.ident,
        consts: context.consts,
        structs: context.structs,
        extern_structs: context.extern_structs,
        enums: context.enums,
        functions: context.functions,
        aliases: HashMap::new(),
    })
}