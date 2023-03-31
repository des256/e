use {
    super::*,
    std::{
        cell::RefCell,
        cmp::PartialEq,
    },
};

// - convert aliases to their types
// - convert anonymous tuples to structs

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
            (Type::Ident(ident1),Type::Ident(ident2)) => ident1 == ident2,
            _ => false,
        }
    }
}

struct Pass1Context {
    pub stdlib: StandardLib,
    pub module: Module,
    pub anon_tuple_types: RefCell<Vec<Vec<Type>>>,
}

impl Pass1Context {

    fn pat_eq(&self,pat: &Pat,scrut: &Expr) -> Result<Option<Expr>,String> {

        match pat {

            Pat::Wildcard | Pat::Rest => Ok(None),

            Pat::Boolean(value) => Ok(Some(
                if *value {
                    scrut.clone()
                }
                else {
                    Expr::Unary(UnaryOp::Not,Box::new(scrut.clone()))
                }
            )),

            Pat::Integer(value) => Ok(Some(
                Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Integer(*value)))
            )),

            Pat::Float(value) => Ok(Some(
                Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Float(*value)))
            )),

            Pat::AnonTuple(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.pat_eq(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr),));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Array(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.pat_eq(&pats[i],&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64))))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Range(pat_lo,pat_hi) => {
                if let Pat::Integer(value_lo) = **pat_lo {
                    if let Pat::Integer(value_hi) = **pat_hi {
                        Ok(Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Integer(value_lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Integer(value_hi)))),
                            )
                        ))
                    }
                    else {
                        Err("pattern range can only be integers or floats".to_string())
                    }
                }
                else if let Pat::Float(value_lo) = **pat_lo {
                    if let Pat::Float(value_hi) = **pat_hi {
                        Ok(Some(
                            Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Float(value_lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Float(value_hi)))),
                            )
                        ))
                    }
                    else {
                        Err("pattern range can only be integers or floats".to_string())
                    }
                }
                else {
                    Err("pattern range can only be integers or floats".to_string())
                }
            },

            // TODO: if identifier refers to a constant or a constant from stdlib, insert == node
            Pat::Ident(/* ident */_) => /*if self.consts.contains_key(ident) {
                Ok(Some(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Const(ident.clone())))))
            }
            else*/ {
                Ok(None)
            },

            Pat::Tuple(_,pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.pat_eq(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))? {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Struct(_,fields) => {
                let mut accum: Option<Expr> = None;
                for ident_pat in fields.iter() {
                    if let FieldPat::IdentPat(ident,pat) = ident_pat {
                        if let Some(expr) = self.pat_eq(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))? {
                            if let Some(accum_inner) = accum {
                                accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                            }
                            else {
                                accum = Some(expr);
                            }
                        }
                    }
                }
                Ok(accum)
            },

            Pat::Variant(enum_ident,variant_ident,variant_pat) => {
                // TODO: variant_ident should be usize
                let mut accum = Expr::EnumDiscr(Box::new(scrut.clone()),/* TODO: variant_ident */0);
                match variant_pat {
                    VariantPat::Naked => { },
                    VariantPat::Tuple(pats) => {
                        for i in 0..pats.len() {
                            // TODO: variant_ident should be usize
                            if let Some(expr) = self.pat_eq(&pats[i],&Expr::EnumArg(Box::new(scrut.clone()),/* TODO: variant_ident */0,i))? {
                                accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                            }
                        }
                    },
                    VariantPat::Struct(fields) => {
                        for field in fields.iter() {
                            if let FieldPat::IdentPat(_,pat) = field {
                                // TODO: variant_ident and ident should be usize
                                if let Some(expr) = self.pat_eq(pat,&Expr::EnumArg(Box::new(scrut.clone()),/* TODO: variant_ident */0,/* TODO: ident */0))? {
                                    accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                                }
                            }
                        }
                    },
                }
                Ok(Some(accum))
            },
        }
    }

    fn pats_eq(&self,pats: &Vec<Pat>,local: &'static str) -> Result<Option<Expr>,String> {
        let mut accum: Option<Expr> = None;
        for pat in pats.iter() {
            if let Some(expr) = self.pat_eq(pat,&Expr::Ident(local))? {
                if let Some(accum_inner) = accum {
                    accum = Some(Expr::Binary(Box::new(accum_inner),BinaryOp::LogOr,Box::new(expr)));
                }
                else {
                    accum = Some(expr);
                }
            }
        }
        Ok(accum)
    }

    fn pat(&self,pat: &Pat,scrut: &Expr) -> Result<Vec<Stat>,String> {
        let mut stats: Vec<Stat> = Vec::new();
        match pat {
            // TODO: Pat::Const
            Pat::Wildcard | Pat::Rest | Pat::Boolean(_) | Pat::Integer(_) | Pat::Float(_) | Pat::Range(_,_) /*| Pat::Const(_)*/ => { },
            Pat::AnonTuple(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                }
            },
            Pat::Array(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.pat(&pats[i],&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64))))?);
                }
            },
            Pat::Ident(ident) => stats.push(Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(scrut.clone()))),
            Pat::Tuple(_,pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                }
            },
            Pat::Struct(_,fields) => {
                for field in fields.iter() {
                    match field {
                        FieldPat::Wildcard | FieldPat::Rest => { },
                        FieldPat::Ident(ident) => stats.push(
                            Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(Expr::Field(Box::new(scrut.clone()),ident.clone())))
                        ),
                        FieldPat::IdentPat(ident,pat) => stats.append(
                            &mut self.pat(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))?
                        ),
                    }
                }
            },
            Pat::Variant(_,_,variant_pat) => match variant_pat {
                VariantPat::Naked => { },
                VariantPat::Tuple(pats) => {
                    for i in 0..pats.len() {
                        stats.append(&mut self.pat(&pats[i],&Expr::TupleIndex(Box::new(scrut.clone()),i))?);
                    }
                },
                VariantPat::Struct(field_pats) => {
                    for field_pat in field_pats.iter() {
                        if let FieldPat::IdentPat(ident,pat) = field_pat {
                            stats.append(&mut self.pat(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone()))?);
                        }
                    }
                },
            },
        }
        Ok(stats)
    }

    fn pats(&self,pats: &Vec<Pat>,scrut: &'static str) -> Result<Vec<Stat>,String> {
        let mut stats: Vec<Stat> = Vec::new();
        for pat in pats.iter() {
            stats.append(&mut self.pat(pat,&Expr::Ident(scrut))?);
        }
        Ok(stats)
    }

    fn type_(&self,type_: &Type) -> Result<Type,String> {

        match type_ {

            // convert anonymous tuple type into a struct that can be referenced by unique index
            Type::AnonTuple(types) => {
                let mut anon_tuple_types = self.anon_tuple_types.borrow_mut();
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(self.type_(type_)?);
                }
                let mut found_index: Option<usize> = None;
                for i in 0..anon_tuple_types.len() {
                    if anon_tuple_types[i].len() == types.len() {
                        let mut all_types_match = true;
                        for k in 0..types.len() {
                            if anon_tuple_types[i][k] != new_types[k] {
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
                    let index = anon_tuple_types.len();
                    anon_tuple_types.push(new_types);
                    Ok(Type::AnonTupleRef(index))
                }
            },

            // recursively prepare type
            Type::Array(type_,count) => {
                let new_type = self.type_(type_)?;
                Ok(Type::Array(Box::new(new_type),*count))
            },

            Type::Ident(ident) => {
                let found_alias = self.module.aliases.iter().find(|alias| &alias.ident == ident);
                if let Some(alias) = found_alias {
                    return Ok(alias.type_.clone());
                }
                let found_struct = self.module.structs.iter().find(|struct_| &struct_.ident == ident);
                if let Some(struct_) = found_struct {
                    return Ok(Type::StructRef(struct_.ident));
                }
                let found_tuple = self.module.tuples.iter().find(|tuple| &tuple.ident == ident);
                if let Some(tuple) = found_tuple {
                    return Ok(Type::TupleRef(tuple.ident));
                }
                let found_enum = self.module.enums.iter().find(|enum_| &enum_.ident == ident);
                if let Some(enum_) = found_enum {
                    return Ok(Type::EnumRef(enum_.ident));
                }
                let found_struct = self.stdlib.structs.iter().find(|struct_| &struct_.ident == ident);
                if let Some(struct_) = found_struct {
                    return Ok(Type::StructRef(struct_.ident));
                }
                let found_tuple = self.stdlib.tuples.iter().find(|tuple| &tuple.ident == ident);
                if let Some(tuple) = found_tuple {
                    return Ok(Type::TupleRef(tuple.ident));
                }
                let found_enum = self.stdlib.enums.iter().find(|enum_| &enum_.ident == ident);
                if let Some(enum_) = found_enum {
                    return Ok(Type::EnumRef(enum_.ident));
                }
                Err(format!("Unknown identifier {}",ident))
            },

            // everything else is fine as it is
            _ => Ok(type_.clone()),
        }
    }

    fn block(&self,block: &Block,expected_type: &Type) -> Result<Block,String> {

        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {

                // recurse into let statement, no need to fix anything further just yet
                Stat::Let(pat,type_,expr) => {
                    let new_type = self.type_(type_)?;
                    let new_expr = Expr::Cast(Box::new(self.expr(expr,&new_type)?),Box::new(new_type));
                    new_stats.append(&mut self.pat(&pat,&new_expr)?);
                },

                // recurse into expression, keep type inferred here and ignore the result
                Stat::Expr(expr) => {
                    let new_expr = self.expr(expr,&Type::Inferred)?;
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },

                Stat::Local(ident,type_,expr) => {
                    let new_expr = self.expr(expr,type_)?;
                    new_stats.push(Stat::Local(ident.clone(),(*type_).clone(),Box::new(new_expr)));
                },
            }
        }

        // recurse into return expression, expecting expected_type as result
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.expr(&expr,expected_type)?))
        }
        else {
            None
        };

        Ok(Block {
            stats: new_stats,
            expr: new_expr,
        })
    }

    fn range(&self,range: &Range) -> Result<Range,String> {
        match range {
            Range::Only(expr) => Ok(Range::Only(Box::new(self.expr(expr,&Type::Inferred)?))),
            Range::FromTo(lo_expr,hi_expr) => Ok(Range::FromTo(Box::new(self.expr(lo_expr,&Type::Inferred)?),Box::new(self.expr(hi_expr,&Type::Inferred)?))),
            Range::FromToIncl(lo_expr,hi_expr) => Ok(Range::FromToIncl(Box::new(self.expr(lo_expr,&Type::Inferred)?),Box::new(self.expr(hi_expr,&Type::Inferred)?))),
            Range::From(expr) => Ok(Range::From(Box::new(self.expr(expr,&Type::Inferred)?))),
            Range::To(expr) => Ok(Range::To(Box::new(self.expr(expr,&Type::Inferred)?))),
            Range::ToIncl(expr) => Ok(Range::ToIncl(Box::new(self.expr(expr,&Type::Inferred)?))),
            Range::All => Ok(Range::All),
        }
    }

    fn expr(&self,expr: &Expr,expected_type: &Type) -> Result<Expr,String> {
        match expr {

            // only allow boolean literal if boolean or inferred are expected
            Expr::Boolean(value) => match expected_type {
                Type::Inferred | Type::Bool => Ok(Expr::Boolean(*value)),
                _ => Err(format!("{} expected instead of boolean literal",expected_type)),
            },

            // only allow integer literal if integer, float or inferred are expected
            Expr::Integer(value) => match expected_type {
                Type::Inferred | Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::F16 | Type::F32 | Type::F64 => Ok(Expr::Integer(*value)),
                _ => Err(format!("{} expected instead of integer literal",expected_type)),
            },

            // only allow float literal if float or inferred are expected
            Expr::Float(value) => match expected_type {
                Type::Inferred | Type::F16 | Type::F32 | Type::F64 => Ok(Expr::Float(*value)),
                _ => Err(format!("{} expected instead of float literal",expected_type)),
            },

            Expr::Array(exprs) => match expected_type {

                // just recurse into array when expected_type is inferred
                Type::Inferred => {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for expr in exprs {
                        new_exprs.push(self.expr(expr,&Type::Inferred)?);
                    }
                    Ok(Expr::Array(new_exprs))
                },

                // match exact types while recursing into array when expected_type is array
                Type::Array(expected_type,expected_count) => {
                    if exprs.len() == *expected_count {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for _ in 0..*expected_count {
                            new_exprs.push(self.expr(&expr,expected_type)?);
                        }
                        Ok(Expr::Array(new_exprs))
                    }
                    else {
                        Err(format!("{} elements expected, {} found",expected_count,exprs.len()))
                    }
                },

                _ => Err(format!("{} expected instead of array literal",expected_type)),
            },

            Expr::Cloned(value_expr,count) => match expected_type {

                // just recurse into expression when expected_type is inferred
                Type::Inferred => {
                    let new_value_expr = self.expr(value_expr,&Type::Inferred)?;
                    Ok(Expr::Cloned(Box::new(new_value_expr),*count))
                },

                // match exact types while recursing into array when expected_type is array
                Type::Array(expected_type,expected_count) => {
                    if count == expected_count {
                        let new_value_expr = self.expr(value_expr,expected_type)?;
                        Ok(Expr::Cloned(Box::new(new_value_expr),*count))
                    }
                    else {
                        Err(format!("{} elements expected, {} found",expected_count,count))
                    }
                },

                _ => Err(format!("{} expected instead of array clone literal",expected_type)),
            },

            Expr::Index(array_expr,index_expr) => {
                // infer for now
                let new_array_expr = self.expr(array_expr,&Type::Inferred)?;
                let new_index_expr = self.expr(index_expr,&Type::Inferred)?;
                Ok(Expr::Index(Box::new(new_array_expr),Box::new(new_index_expr)))
            },

            Expr::Cast(expr,type_) => {
                let new_type = self.type_(&type_)?;
                if let Type::Inferred = expected_type { } else {
                    if new_type != *expected_type {
                        return Err(format!("{} expected instead of {}",expected_type,new_type));
                    }
                }
                let new_expr = self.expr(expr,&new_type)?;
                Ok(Expr::Cast(Box::new(new_expr),Box::new(new_type)))
            },

            Expr::AnonTuple(exprs) => if let Type::AnonTupleRef(index) = expected_type {
                let anon_tuple_types = self.anon_tuple_types.borrow();
                if anon_tuple_types[*index].len() == exprs.len() {
                    let mut new_exprs: Vec<Expr> = Vec::new();
                    for i in 0..exprs.len() {
                        new_exprs.push(self.expr(&exprs[i],&anon_tuple_types[*index][i])?);
                    }
                    Ok(Expr::AnonTupleLit(*index,new_exprs))
                }
                else {
                    Err(format!("{} and {} have different dimensions",expr,expected_type))
                }
            }
            else {
                Err(format!("{} expected instead of {}",expected_type,expr))
            },

            Expr::Unary(op,expr) => {
                // infer for now
                let new_expr = self.expr(expr,&Type::Inferred)?;
                Ok(Expr::Unary(op.clone(),Box::new(new_expr)))
            },

            Expr::Binary(expr1,op,expr2) => {
                // infer for now
                let new_expr1 = self.expr(expr1,&Type::Inferred)?;
                let new_expr2 = self.expr(expr2,&Type::Inferred)?;
                Ok(Expr::Binary(Box::new(new_expr1),op.clone(),Box::new(new_expr2)))
            },

            Expr::Continue => Ok(Expr::Continue),

            Expr::Break(expr) => if let Some(expr) = expr {
                Ok(Expr::Break(Some(Box::new(self.expr(expr,&Type::Inferred)?))))
            }
            else {
                Ok(Expr::Break(None))
            },

            Expr::Return(expr) => if let Some(expr) = expr {
                Ok(Expr::Return(Some(Box::new(self.expr(expr,&Type::Inferred)?))))
            }
            else {
                Ok(Expr::Return(None))
            },

            Expr::Block(block) => Ok(Expr::Block(self.block(block,expected_type)?)),

            Expr::If(cond_expr,block,else_expr) => {
                // recurse, and make sure block and else_expr have the expected type
                let new_cond_expr = self.expr(cond_expr,&Type::Bool)?;
                let new_block = self.block(block,expected_type)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.expr(else_expr,expected_type)?))
                }
                else {
                    None
                };
                Ok(Expr::If(Box::new(new_cond_expr),new_block,new_else_expr))
            },

            Expr::While(cond_expr,block) => {
                // recurse, make sure cond_expr is boolean
                let new_cond_expr = self.expr(cond_expr,&Type::Bool)?;
                let new_block = self.block(block,&Type::Inferred)?;
                Ok(Expr::While(Box::new(new_cond_expr),new_block))
            },

            Expr::Loop(block) => Ok(Expr::Loop(self.block(block,&Type::Inferred)?)),

            Expr::IfLet(pats,expr,block,else_expr) => {
                let new_expr = self.expr(expr,&Type::Bool)?;
                let mut new_block = self.block(block,expected_type)?;
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.expr(else_expr,expected_type)?))
                }
                else {
                    None
                };
                let mut if_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let condition = self.pats_eq(&pats,"scrut")?;
                if condition.is_none() {
                    return Err("unable to create boolean condition from if-let patterns".to_string());
                }
                let condition = condition.unwrap();
                let mut then_block = Block {
                    stats: self.pats(&pats,"scrut")?,
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                if_block.expr = Some(Box::new(Expr::If(Box::new(condition),then_block,new_else_expr)));
                Ok(Expr::Block(if_block))
            },
        
            Expr::For(pats,range,block) => {
                if pats.len() == 1 {
                    if let Pat::Ident(ident) = &pats[0] {
                        let (expr,op,expr2) = match range {
                            Range::FromTo(expr1,expr2) => (self.expr(expr1,expected_type)?,BinaryOp::Less,self.expr(expr2,expected_type)?),
                            Range::FromToIncl(expr1,expr2) => (self.expr(expr1,expected_type)?,BinaryOp::LessEq,self.expr(expr2,expected_type)?),
                            _ => { return Err("for range can only be .. or ..=".to_string()) },
                        };
                        let mut stats = block.stats.clone();
                        if let Some(expr) = &block.expr {
                            stats.push(Stat::Expr(expr.clone()));
                        }
                        let loop_block = Block {
                            stats,
                            expr: Some(Box::new(Expr::Binary(Box::new(Expr::Ident(ident)),BinaryOp::AddAssign,Box::new(Expr::Integer(1))))),
                        };
                        let for_block = Block {
                            stats: vec![
                                Stat::Local(ident.clone(),Box::new(Type::Inferred),Box::new(expr)),
                            ],
                            expr: Some(Box::new(Expr::While(Box::new(Expr::Binary(Box::new(Expr::Ident(ident)),op,Box::new(expr2))),loop_block))),
                        };
                        Ok(Expr::Block(for_block))
                    }
                    else {
                        Err("for-pattern can only be a single variable".to_string())
                    }
                }
                else {
                    Err("for-pattern can only be a single variable".to_string())
                }
            },

            Expr::WhileLet(pats,expr,block) => {
                let new_expr = self.expr(expr,&Type::Bool)?;
                let mut new_block = self.block(block,&Type::Inferred)?;
                let mut while_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let condition = self.pats_eq(&pats,"scrut")?;
                if condition.is_none() {
                    return Err("unable to create boolean condition from while-let patterns".to_string());
                }
                let condition = condition.unwrap();
                let mut then_block = Block {
                    stats: self.pats(&pats,"scrut")?,
                    expr: new_block.expr.clone(),
                };
                then_block.stats.append(&mut new_block.stats);
                while_block.expr = Some(Box::new(Expr::While(Box::new(condition),then_block)));
                Ok(Expr::Block(while_block))
            },

            Expr::Match(expr,arms) => {
                let new_expr = self.expr(expr,&Type::Inferred)?;
                let mut match_block = Block {
                    stats: vec![
                        Stat::Local("scrut",Box::new(Type::Inferred),Box::new(new_expr)),
                    ],
                    expr: None,
                };
                let mut exprs: Vec<Expr> = Vec::new();
                let mut else_expr: Option<Box<Expr>> = None;
                for (pats,_,expr) in arms {
                    // TODO: consider if_expr
                    let new_expr = self.expr(expr,&Type::Inferred)?;
                    if let Some(condition) = self.pats_eq(&pats,"scrut")? {
                        let arm_block = Block {
                            stats: self.pats(&pats,"scrut")?,
                            expr: Some(Box::new(new_expr)),
                        };
                        exprs.push(Expr::If(Box::new(condition),arm_block,None));
                    }
                    else if let None = else_expr {
                        let arm_block = Block {
                            stats: Vec::new(),
                            expr: Some(Box::new(new_expr)),
                        };
                        else_expr = Some(Box::new(Expr::Block(arm_block)));
                    }
                    else {
                        return Err("match expression can only have one wildcard".to_string());
                    }
                }
                let mut result_expr: Option<Box<Expr>> = else_expr;
                for i in 0..exprs.len() {
                    if let Expr::If(condition,block,_) = &exprs[exprs.len() - i - 1] {
                        result_expr = Some(Box::new(Expr::If(condition.clone(),block.clone(),result_expr)));
                    }
                }
                match_block.expr = Some(Box::new(*result_expr.unwrap()));
                Ok(Expr::Block(match_block))
            },

            Expr::Ident(ident) => {
                let found_module_const = self.module.consts.iter().find(|const_| &const_.ident == ident);
                let found_stdlib_const = self.stdlib.consts.iter().find(|const_| &const_.ident == ident);
                let const_ = match (found_module_const,found_stdlib_const) {
                    (Some(const_),_) | (_,Some(const_)) => const_,
                    _ => return Ok(Expr::LocalRefOrParamRef(ident)),
                };
                if let Type::Inferred = expected_type {
                    Ok(Expr::ConstRef(ident))
                }
                else {
                    if const_.type_ == *expected_type {
                        Ok(Expr::ConstRef(ident))
                    }
                    else {
                        Err(format!("{} expected instead of {}",expected_type,expr))
                    }
                }
            },

            Expr::TupleLitOrFunctionCall(ident,exprs) => {
                let found_module_function = self.module.functions.iter().find(|function| &function.ident == ident);
                let found_stdlib_function = self.stdlib.functions.iter().find(|function| &function.ident == ident);
                match (found_module_function,found_stdlib_function) {
                    (Some(function),_) | (_,Some(function)) => {
                        if exprs.len() != function.params.len() {
                            return Err(format!("{} parameters expected instead of {}",function.params.len(),exprs.len()));
                        }
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for i in 0..exprs.len() {
                            new_exprs.push(self.expr(&exprs[i],&function.params[i].1)?);
                        }
                        if let Type::Inferred = expected_type {
                            // just infer
                            Ok(Expr::FunctionCall(ident,new_exprs))
                        } else {
                            // make sure the return value is the expected type
                            if function.return_type != *expected_type {
                                return Err(format!("{} expected instead of {}",expected_type,expr))
                            }
                            Ok(Expr::FunctionCall(ident,new_exprs))
                        }
                    },
                    _ => {
                        let found_module_tuple = self.module.tuples.iter().find(|tuple| &tuple.ident == ident);
                        let found_stdlib_tuple = self.stdlib.tuples.iter().find(|tuple| &tuple.ident == ident);
                        match (found_module_tuple,found_stdlib_tuple) {
                            (Some(tuple),_) | (_,Some(tuple)) => {
                                if exprs.len() != tuple.types.len() {
                                    return Err(format!("{} components expected instead of {}",tuple.types.len(),exprs.len()));
                                }
                                let mut new_exprs: Vec<Expr> = Vec::new();
                                for i in 0..exprs.len() {
                                    new_exprs.push(self.expr(&exprs[i],&tuple.types[i])?);
                                }
                                if let Type::Inferred = expected_type {
                                    // just infer
                                    Ok(Expr::TupleLit(ident,new_exprs))
                                }
                                else {
                                    if let Type::TupleRef(expected_ident) = expected_type {
                                        if ident == expected_ident {
                                            Ok(Expr::TupleLit(ident,new_exprs))
                                        }
                                        else {
                                            Err(format!("{} expected instead of {}",expected_ident,ident))
                                        }
                                    }
                                    else {
                                        Err(format!("{} expected instead of {}",expected_type,ident))
                                    }
                                }            
                            },
                            _ => {
                                Err(format!("tuple or function {} not found",ident))
                            },
                        }
                    },
                }
            },

            Expr::StructLit(ident,fields) => {
                let found_module_struct = self.module.structs.iter().find(|struct_| &struct_.ident == ident);
                let found_extern_struct = self.module.extern_structs.iter().find(|struct_| &struct_.ident == ident);
                let found_stdlib_struct = self.stdlib.structs.iter().find(|struct_| &struct_.ident == ident);
                match (found_module_struct,found_extern_struct,found_stdlib_struct) {
                    (Some(struct_),_,_) | (_,Some(struct_),_) | (_,_,Some(struct_)) => {
                        if fields.len() != struct_.fields.len() {
                            return Err(format!("{} fields expected instead of {}",struct_.fields.len(),fields.len()));
                        }
                        let mut new_exprs: Vec<(&'static str,Expr)> = Vec::new();
                        for i in 0..fields.len() {
                            new_exprs.push((struct_.fields[i].0,self.expr(&fields[i].1,&struct_.fields[i].1)?));
                        }
                        if let Type::Inferred = expected_type {
                            // just infer
                            Ok(Expr::StructLit(ident,new_exprs))
                        }
                        else {
                            if let Type::StructRef(expected_ident) = expected_type {
                                if ident == expected_ident {
                                    Ok(Expr::StructLit(ident,new_exprs))
                                }
                                else {
                                    Err(format!("{} expected instead of {}",expected_ident,ident))
                                }
                            }
                            else {
                                Err(format!("{} expected instead of {}",expected_type,ident))
                            }
                        }    
                    },
                    _ => {
                        Err(format!("unknown struct {}",ident))
                    }
                }
            },

            Expr::VariantLit(enum_ident,variant_ident,variant_expr) => {
                let found_module_enum = self.module.enums.iter().find(|enum_| &enum_.ident == enum_ident);
                let found_stdlib_enum = self.stdlib.enums.iter().find(|enum_| &enum_.ident == enum_ident);
                match (found_module_enum,found_stdlib_enum) {
                    (Some(enum_),_) | (_,Some(enum_)) => {
                        let mut found_variant: Option<Variant> = None;
                        for variant in enum_.variants.iter() {
                            if variant.0 == *variant_ident {
                                found_variant = Some(variant.1.clone());
                            }
                        }
                        if let None = found_variant {
                            return Err(format!("enum {} has no variant {}",enum_ident,variant_ident));
                        }
                        let variant = found_variant.unwrap();
                        match (variant,variant_expr) {
                            (Variant::Naked,VariantExpr::Naked) => Ok(Expr::VariantLit(enum_ident,variant_ident,VariantExpr::Naked)),
                            (Variant::Tuple(types),VariantExpr::Tuple(exprs)) => {
                                if exprs.len() == types.len() {
                                    let mut new_exprs: Vec<Expr> = Vec::new();
                                    for i in 0..exprs.len() {
                                        new_exprs.push(self.expr(&exprs[i],&types[i])?);
                                    }
                                    Ok(Expr::VariantLit(enum_ident,variant_ident,VariantExpr::Tuple(new_exprs)))
                                }
                                else {
                                    Err(format!("{} components expected instead of {}",types.len(),exprs.len()))
                                }
                            },
                            (Variant::Struct(field_types),VariantExpr::Struct(field_exprs)) => {
                                if field_exprs.len() == field_types.len() {
                                    let mut new_field_exprs: Vec<(&'static str,Expr)> = Vec::new();
                                    for i in 0..field_exprs.len() {
                                        if field_exprs[i].0 == field_types[i].0 {
                                            new_field_exprs.push((field_exprs[i].0,self.expr(&field_exprs[i].1,&field_types[i].1)?));
                                        }
                                        else {
                                            return Err(format!("{} field not found in {}::{}",field_exprs[i].0,enum_ident,variant_ident))
                                        }
                                    }
                                    Ok(Expr::VariantLit(enum_ident,variant_ident,VariantExpr::Struct(new_field_exprs)))
                                }
                                else {
                                    Err(format!("{} fields expected instead of {}",field_types.len(),field_exprs.len()))
                                }
                            },
                            (Variant::Naked,_) => Err(format!("{}::{} has no tuple components or struct fields",enum_ident,variant_ident)),
                            (Variant::Tuple(_),_) => Err(format!("{}::{}(...) requires tuple components",enum_ident,variant_ident)),
                            (Variant::Struct(_),_) => Err(format!("{}::{} {{ ... }} requires struct fields",enum_ident,variant_ident)),
                        }    
                    },
                    _ => {
                        Err(format!("unknown enum {}",enum_ident))
                    },
                }
            },

            Expr::MethodCall(expr,ident,exprs) => {
                let method = match expected_type {
                    Type::Inferred => {
                        // TODO: get expr's type
                        let found_method = self.stdlib.methods.iter().find(|method| &method.ident == ident);
                        if let Some(method) = found_method {
                            method
                        }
                        else {
                            return Err(format!("unknown method {}",ident));
                        }
                    },
                    _ => {
                        let found_method = self.stdlib.methods.iter().find(|method| (&method.ident == ident) && (&method.return_type == expected_type));
                        if let Some(method) = found_method {
                            method
                        }
                        else {
                            return Err(format!("unknown method {}",ident));
                        }
                    },
                };
                let new_expr = self.expr(expr,&Type::Inferred)?;
                if exprs.len() != method.params.len() {
                    return Err(format!("{} parameters expected instead of {}",method.params.len(),exprs.len()));
                }
                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..exprs.len() {
                    new_exprs.push(self.expr(&exprs[i],&method.params[i].1)?);
                }
                Ok(Expr::MethodCall(Box::new(new_expr),ident,new_exprs))
            },

            Expr::Field(expr,ident) => {
                // TODO: figure out type of expr
                Ok(Expr::Field(Box::new(self.expr(expr,&Type::Inferred)?),ident))
            },

            Expr::TupleIndex(expr,index) => {
                // TODO: figure out type of expr
                Ok(Expr::TupleIndex(Box::new(self.expr(expr,&Type::Inferred)?),*index))
            },

            Expr::TupleLit(ident,exprs) => Ok(Expr::TupleLit(ident,exprs.clone())),

            Expr::FunctionCall(ident,exprs) => Ok(Expr::FunctionCall(ident,exprs.clone())),

            Expr::StructLit(ident,exprs) => Ok(Expr::StructLit(ident,exprs.clone())),

            Expr::AnonTupleLit(index,exprs) => Ok(Expr::AnonTupleLit(*index,exprs.clone())),

            Expr::ConstRef(ident) => Ok(Expr::ConstRef(ident)),

            Expr::LocalRefOrParamRef(ident) => Ok(Expr::LocalRefOrParamRef(ident)),

            Expr::EnumDiscr(expr,index) => Ok(Expr::EnumDiscr(expr.clone(),*index)),

            Expr::EnumArg(expr,variant_index,index) => Ok(Expr::EnumArg(expr.clone(),*variant_index,*index)),
        }
    }
}

pub fn pass1(module: &Module) -> Result<Module,String> {

    let mut context = Pass1Context {
        stdlib: StandardLib::new(),
        module: module.clone(),
        anon_tuple_types: RefCell::new(Vec::new()),
    };

    // create aliases with final type
    let mut aliases: Vec<Alias> = Vec::new();
    for alias in context.module.aliases.iter() {
        let mut type_ = &alias.type_;
        while let Type::Ident(ident) = type_ {
            let found_alias = context.module.aliases.iter().find(|alias| &alias.ident == ident);
            if let Some(alias) = found_alias {
                type_ = &alias.type_;
            }
            else {
                break;
            }
        }
        let type_ = context.type_(type_)?;
        aliases.push(Alias { ident: alias.ident,type_, });
    }
    context.module.aliases = aliases;

    // prepare tuples
    let mut tuples: Vec<Tuple> = Vec::new();
    for tuple in context.module.tuples.iter() {
        let mut new_types: Vec<Type> = Vec::new();
        for type_ in tuple.types.iter() {
            new_types.push(context.type_(type_)?);
        }
        tuples.push(Tuple { ident: tuple.ident,types: new_types, });
    }
    context.module.tuples = tuples;

    // prepare structs
    let mut structs: Vec<Struct> = Vec::new();
    for struct_ in context.module.structs.iter() {
        let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
        for (ident,type_) in struct_.fields.iter() {
            new_fields.push((ident,context.type_(type_)?));
        }
        structs.push(Struct { ident: struct_.ident,fields: new_fields, });
    }
    context.module.structs = structs;

    // prepare external structs
    let mut structs: Vec<Struct> = Vec::new();
    for struct_ in context.module.extern_structs.iter() {
        let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
        for (ident,type_) in struct_.fields.iter() {
            new_fields.push((ident,context.type_(type_)?));
        }
        structs.push(Struct { ident: struct_.ident,fields: new_fields, });
    }
    context.module.extern_structs = structs;

    // prepare enums
    let mut enums: Vec<Enum> = Vec::new();
    for enum_ in context.module.enums.iter() {
        let mut new_variants: Vec<(&'static str,Variant)> = Vec::new();
        for (ident,variant) in enum_.variants.iter() {
            let new_variant = match variant {
                Variant::Naked => Variant::Naked,
                Variant::Tuple(types) => {
                    let mut new_types: Vec<Type> = Vec::new();
                    for type_ in types {
                        new_types.push(context.type_(type_)?);
                    }
                    Variant::Tuple(new_types)
                },
                Variant::Struct(fields) => {
                    let mut new_fields: Vec<(&'static str,Type)> = Vec::new();
                    for (ident,type_) in fields {
                        new_fields.push((ident,context.type_(type_)?))
                    }
                    Variant::Struct(new_fields)
                },
            };
            new_variants.push((ident,new_variant));
        }
        enums.push(Enum { ident: enum_.ident,variants: new_variants, });
    }
    context.module.enums = enums;

    // prepare consts
    let mut consts: Vec<Const> = Vec::new();
    for const_ in context.module.consts.iter() {
        let new_type = context.type_(&const_.type_)?;
        let new_expr = context.expr(&const_.expr,&new_type)?;
        consts.push(Const { ident: const_.ident,type_: new_type,expr: new_expr, });
    }
    context.module.consts = consts;

    // prepare functions
    let mut functions: Vec<Function> = Vec::new();
    for function in context.module.functions.iter() {
        let mut new_params: Vec<(&'static str,Type)> = Vec::new();
        for (ident,type_) in function.params.iter() {
            new_params.push((ident,context.type_(&type_)?));
        }
        let new_return_type = context.type_(&function.return_type)?;
        let new_block = context.block(&function.block,&new_return_type)?;
        functions.push(Function { ident: function.ident,params: new_params,return_type: new_return_type,block: new_block, });
    }
    context.module.functions = functions;

    Ok(PreparedModule {
        ident: context.module.ident,
        tuples: context.module.tuples,
        structs: context.module.structs,
        extern_structs: context.module.extern_structs,
        enums: context.module.enums,
        consts: context.module.consts,
        functions: context.module.functions,
        anon_tuple_types: context.anon_tuple_types.into_inner(),
    })
}
