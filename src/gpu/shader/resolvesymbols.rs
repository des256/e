use super::*;

use ast::*;

pub trait ResolveSymbols {
    fn resolve_symbols(self,context: &Context) -> Self;
}

/*
    fn convert_tuple(tuple: &Tuple) -> Struct {
        let mut fields: Vec<Symbol> = Vec::new();
        for i in 0..tuple.types.len() {
            let ident = format!("_{}",i);
            let type_ = tuple.types[i].clone();
            fields.push(Symbol { ident,type_, });
        }
        Struct { ident: tuple.ident.clone(),fields, }
    }

    fn count_variant_types(variant_types: &mut Vec<(Type,usize)>,type_: &Type) {
        let mut found = false;
        for variant_type in variant_types.iter_mut() {
            if *type_ == variant_type.0 {
                (*variant_type).1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            variant_types.push((type_.clone(),1));
        }
    }

    fn wrap_minimal(type_counts: &mut Vec<(Type,usize)>,variant_types: &Vec<(Type,usize)>) {
        for variant_type in variant_types.iter() {
            let mut found = false;
            for type_count in type_counts.iter_mut() {
                if type_count.0 == variant_type.0 {
                    found = true;
                    if variant_type.1 > type_count.1 {
                        (*type_count).1 = variant_type.1;
                    }
                    break;
                }
            }
            if !found {
                type_counts.push((variant_type.0.clone(),variant_type.1));
            }
        }
    }

    fn fit_index(type_: &Type,prev_type: &mut Option<Type>,prev_index: &mut usize,type_counts: &Vec<(Type,usize)>) -> usize {
        if let Some(inner_prev_type) = prev_type {
            if type_ == inner_prev_type {
                let index = *prev_index;
                *prev_index += 1;
                index
            }
            else {
                let mut index = 0usize;
                for type_count in type_counts.iter() {
                    if *type_ == type_count.0 {
                        *prev_type = Some(type_.clone());
                        *prev_index = index;
                        break;
                    }
                    index += type_count.1;
                }
                index
            }
        }
        else {
            let mut index = 0usize;
            for type_count in type_counts.iter() {
                if *type_ == type_count.0 {
                    *prev_type = Some(type_.clone());
                    *prev_index = index;
                    break;
                }
                index += type_count.1;
            }
            index
        }
    }

    fn convert_enum(enum_: &Enum,structs: &mut HashMap<String,Struct>,indices: &mut HashMap<String,Vec<Vec<usize>>>) {
        let mut type_counts: Vec<(Type,usize)> = Vec::new();
        for i in 0..enum_.variants.len() {
            
            // count how many of each type exist in this variant
            let mut variant_types: Vec<(Type,usize)> = Vec::new();
            match enum_.variants[i] {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for type_ in types.iter() {
                        Resolver::count_variant_types(&mut variant_types,type_);
                    }
                },
                Variant::Struct(_,fields) => {
                    for field in fields.iter() {
                        Resolver::count_variant_types(&mut variant_types,&field.type_);
                    }
                },
            }

            // wrap into minimal list
            Resolver::wrap_minimal(&mut variant_types,&type_counts);
        }

        // create the fields
        let mut fields: Vec<Symbol> = Vec::new();
        fields.push(Symbol { ident: "discr".to_string(),type_: Type::U32, });
        let mut count = 0usize;
        for type_count in type_counts {
            let ident = format!("_{}",count);
            fields.push(Symbol { ident,type_: type_count.0.clone(), });
            count += type_count.1;
        }

        // map each variant into the the struct
        let mut enum_indices: Vec<Vec<usize>> = Vec::new();
        let mut cur = 0usize;
        for i in 0..enum_.variants.len() {
            match enum_.variants[i] {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    let mut prev_type: Option<Type> = None;
                    let mut prev_index = 0usize;
                    let mut variant_indices: Vec<usize> = Vec::new();
                    for type_ in types.iter() {
                        variant_indices.push(Resolver::fit_index(type_,&mut prev_type,&mut prev_index,&type_counts));
                    }
                    enum_indices.push(variant_indices);
                },
                Variant::Struct(_,fields) => {
                    let mut prev_type: Option<Type> = None;
                    let mut prev_index = 0usize;
                    let mut variant_indices: Vec<usize> = Vec::new();
                    for field in fields.iter() {
                        variant_indices.push(Resolver::fit_index(&field.type_,&mut prev_type,&mut prev_index,&type_counts));
                    }
                    enum_indices.push(variant_indices);
                },
            }
        }
        structs.insert(enum_.ident.clone(),Struct { ident: enum_.ident.clone(),fields, });
        indices.insert(enum_.ident.clone(),enum_indices);
    }
*/

impl ResolveSymbols for Type {
    fn resolve_symbols(self,context: &Context) -> Self {
        match self {
            Type::Inferred => Type::Inferred,
            Type::Void => Type::Void,
            Type::Integer => Type::Integer,
            Type::Float => Type::Float,
            Type::Bool => Type::Bool,
            Type::U8 => Type::U8,
            Type::I8 => Type::I8,
            Type::U16 => Type::U16,
            Type::I16 => Type::I16,
            Type::U32 => Type::U32,
            Type::I32 => Type::I32,
            Type::U64 => Type::U64,
            Type::I64 => Type::I64,
            Type::USize => Type::USize,
            Type::ISize => Type::ISize,
            Type::F16 => Type::F16,
            Type::F32 => Type::F32,
            Type::F64 => Type::F64,
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(type_.resolve_symbols(context));
                }
                Type::AnonTuple(new_types)
            },
            Type::Array(type_,expr) => {
                let new_type = type_.resolve_symbols(context);
                let new_expr = expr.resolve_symbols(context);
                Type::Array(Box::new(new_type),Box::new(new_expr))
            },
            Type::UnknownIdent(ident) => {
                if context.stdlib.tuples.contains_key(&ident) {
                    Type::Tuple(ident)
                }
                else if context.stdlib.structs.contains_key(&ident) {
                    Type::Struct(ident)
                }
                else if context.stdlib.enums.contains_key(&ident) {
                    Type::Enum(ident)
                }
                else if context.stdlib.aliases.contains_key(&ident) {
                    Type::Alias(ident)
                }
                else if context.tuples.contains_key(&ident) {
                    Type::Tuple(ident)
                }
                else if context.structs.contains_key(&ident) {
                    Type::Struct(ident)
                }
                else if context.enums.contains_key(&ident) {
                    Type::Enum(ident)
                }
                else if context.aliases.contains_key(&ident) {
                    Type::Alias(ident)
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },
            Type::Tuple(ident) => Type::Tuple(ident),
            Type::Struct(ident) => Type::Struct(ident),
            Type::Enum(ident) => Type::Enum(ident),
            Type::Alias(ident) => Type::Alias(ident),
        }
    }
}

impl ResolveSymbols for Block {
    fn resolve_symbols(self,context: &Context) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in self.stats.iter() {
            new_stats.push(stat.resolve_symbols(context));
        }
        let new_expr = if let Some(expr) = self.expr {
            Some(Box::new(expr.resolve_symbols(context)))
        }
        else {
            None
        };
        Block { stats: new_stats,expr: new_expr, }
    }
}

impl ResolveSymbols for Range {
    fn resolve_symbols(self,context: &Context) -> Range {
        match self {
            Range::Only(expr) => Range::Only(Box::new(expr.resolve_symbols(context))),
            Range::FromTo(expr,expr2) => Range::FromTo(Box::new(expr.resolve_symbols(context)),Box::new(expr2.resolve_symbols(context))),
            Range::FromToIncl(expr,expr2) => Range::FromToIncl(Box::new(expr.resolve_symbols(context)),Box::new(expr2.resolve_symbols(context))),
            Range::From(expr) => Range::From(Box::new(expr.resolve_symbols(context))),
            Range::To(expr) => Range::To(Box::new(expr.resolve_symbols(context))),
            Range::ToIncl(expr) => Range::ToIncl(Box::new(expr.resolve_symbols(context))),
            Range::All => Range::All,
        }
    }
}

impl ResolveSymbols for Expr {
    fn resolve_symbols(self,context: &Context) -> Expr {
        match self {
            Expr::Boolean(value) => Expr::Boolean(value),
            Expr::Integer(value) => Expr::Integer(value),
            Expr::Float(value) => Expr::Float(value),
            Expr::Array(exprs) => Expr::Array(exprs.resolve_symbols(context)),
            Expr::Cloned(expr,expr2) => Expr::Cloned(Box::new(expr.resolve_symbols(context)),Box::new(expr2.resolve_symbols(context))),
            Expr::Index(expr,expr2) => Expr::Index(Box::new(expr.resolve_symbols(context)),Box::new(expr2.resolve_symbols(context))),
            Expr::Cast(expr,type_) => Expr::Cast(Box::new(expr.resolve_symbols(context)),Box::new(type_.resolve_symbols(context))),
            Expr::AnonTuple(exprs) => Expr::AnonTuple(exprs.resolve_symbols(context)),
            Expr::Unary(op,expr) => Expr::Unary(op,Box::new(expr.resolve_symbols(context))),
            Expr::Binary(expr,op,expr2) => Expr::Binary(Box::new(expr.resolve_symbols(context)),op,Box::new(expr2.resolve_symbols(context))),
            Expr::Continue => Expr::Continue,
            Expr::Break(expr) => if let Some(expr) = expr { Expr::Break(Some(Box::new(expr.resolve_symbols(context)))) } else { Expr::Break(None) },
            Expr::Return(expr) => if let Some(expr) = expr { Expr::Return(Some(Box::new(expr.resolve_symbols(context)))) } else { Expr::Return(None) },
            Expr::Block(block) => Expr::Block(block.resolve_symbols(context)),
            Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr { Expr::If(Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context),Some(Box::new(else_expr.resolve_symbols(context)))) } else { Expr::If(Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context),None) },
            Expr::While(expr,block) => Expr::While(Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context)),
            Expr::Loop(block) => Expr::Loop(block.resolve_symbols(context)),
            Expr::IfLet(pats,expr,block,else_expr) => if let Some(else_expr) = else_expr { Expr::IfLet(pats,Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context),Some(Box::new(else_expr.resolve_symbols(context)))) } else { Expr::IfLet(pats,Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context),None) },
            Expr::For(pats,range,block) => Expr::For(pats,range.resolve_symbols(context),block.resolve_symbols(context)),
            Expr::WhileLet(pats,expr,block) => Expr::WhileLet(pats,Box::new(expr.resolve_symbols(context)),block.resolve_symbols(context)),
            Expr::Match(expr,arms) => {
                let mut new_arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
                for (pats,if_expr,expr) in arms.iter() {
                    if let Some(if_expr) = if_expr {
                        new_arms.push((*pats,Some(Box::new(if_expr.resolve_symbols(context))),Box::new(expr.resolve_symbols(context))));
                    }
                    else {
                        new_arms.push((*pats,None,Box::new(expr.resolve_symbols(context))));
                    }
                }
                Expr::Match(Box::new(expr.resolve_symbols(context)),new_arms)
            },
            Expr::UnknownIdent(ident) => if context.stdlib.consts.contains_key(&ident) {
                Expr::Const(ident)
            }
            else if context.consts.contains_key(&ident) {
                Expr::Const(ident)
            }
            else if context.locals.contains_key(&ident) {
                Expr::Local(ident)
            }
            else if context.params.contains_key(&ident) {
                Expr::Param(ident)
            }
            else {
                panic!("unknown identifier {}",ident);
            },
            Expr::TupleOrCall(ident,exprs) => if context.stdlib.tuples.contains_key(&ident) {
                Expr::Tuple(ident,exprs.resolve_symbols(context))
            }
            else if context.stdlib.functions.contains_key(&ident) {
                Expr::Call(ident,exprs.resolve_symbols(context))
            }
            else if context.tuples.contains_key(&ident) {
                Expr::Tuple(ident,exprs.resolve_symbols(context))
            }
            else if context.functions.contains_key(&ident) {
                Expr::Call(ident,exprs.resolve_symbols(context))
            }
            else {
                panic!("unknown tuple or function {}",ident);
            },
            Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for (ident,expr) in fields.iter() {
                    new_fields.push((*ident,expr.resolve_symbols(context)));
                }
                Expr::Struct(ident,new_fields)
            },
            Expr::Variant(enum_ident,variant) => match variant {
                VariantExpr::Naked(ident) => Expr::Variant(enum_ident,VariantExpr::Naked(ident)),
                VariantExpr::Tuple(ident,exprs) => Expr::Variant(enum_ident,VariantExpr::Tuple(ident,exprs.resolve_symbols(context))),
                VariantExpr::Struct(ident,fields) => {
                    let mut new_fields: Vec<(String,Expr)> = Vec::new();
                    for (ident,expr) in fields.iter() {
                        new_fields.push((*ident,expr.resolve_symbols(context)));
                    }
                    Expr::Variant(enum_ident,VariantExpr::Struct(ident,new_fields))
                },
            },
            Expr::Method(expr,ident,exprs) => Expr::Method(Box::new(expr.resolve_symbols(context)),ident,exprs.resolve_symbols(context)),
            Expr::Field(expr,ident) => Expr::Field(Box::new(expr.resolve_symbols(context)),ident),
            Expr::TupleIndex(expr,index) => Expr::TupleIndex(Box::new(expr.resolve_symbols(context)),index),
            Expr::Param(ident) => Expr::Param(ident),
            Expr::Local(ident) => Expr::Local(ident),
            Expr::Const(ident) => Expr::Const(ident),
            Expr::Tuple(ident,exprs) => Expr::Tuple(ident,exprs.resolve_symbols(context)),
            Expr::Call(ident,exprs) => Expr::Call(ident,exprs.resolve_symbols(context)),
            Expr::Discriminant(expr) => Expr::Discriminant(Box::new(expr.resolve_symbols(context))),
            Expr::Destructure(expr,variant_index,field_index) => Expr::Destructure(Box::new(expr.resolve_symbols(context)),variant_index,field_index),        
        }
    }
}

impl ResolveSymbols for Vec<Expr> {
    fn resolve_symbols(self,context: &Context) -> Vec<Expr> {
        let mut new_exprs: Vec<Expr> = Vec::new();
        for expr in self.iter() {
            new_exprs.push(expr.resolve_symbols(context));
        }
        new_exprs
    }
}

impl ResolveSymbols for Stat {
    fn resolve_symbols(self,context: &Context) -> Stat {
        match self {
            Stat::Let(pat,type_,expr) => Stat::Let(pat,Box::new(type_.resolve_symbols(context)),Box::new(expr.resolve_symbols(context))),
            Stat::Expr(expr) => Stat::Expr(Box::new(expr.resolve_symbols(context))),
            Stat::Local(ident,type_,expr) => Stat::Local(ident,Box::new(type_.resolve_symbols(context)),Box::new(expr.resolve_symbols(context))),
        }
    }
}

impl ResolveSymbols for Method {
    fn resolve_symbols(self,context: &Context) -> Method {
        let mut new_params: Vec<Symbol> = Vec::new();
        for param in self.params.iter() {
            new_params.push(Symbol { ident: param.ident,type_: param.type_.resolve_symbols(context), });
        }
        Method {
            from_type: self.from_type.resolve_symbols(context),
            ident: self.ident,
            params: new_params,
            type_: self.type_.resolve_symbols(context),
        }
    }
}

impl ResolveSymbols for Function {
    fn resolve_symbols(self,context: &Context) -> Function {
        let mut new_params: Vec<Symbol> = Vec::new();
        for param in self.params.iter() {
            new_params.push(Symbol { ident: param.ident,type_: param.type_.resolve_symbols(context), });
        }
        Function {
            ident: self.ident,
            params: new_params,
            type_: self.type_.resolve_symbols(context),
            block: self.block.resolve_symbols(context),
        }
    }
}

impl ResolveSymbols for Tuple {
    fn resolve_symbols(self,context: &Context) -> Tuple {
        let mut new_types: Vec<Type> = Vec::new();
        for type_ in self.types.iter() {
            new_types.push(type_.resolve_symbols(context));
        }
        Tuple {
            ident: self.ident,
            types: new_types,
        }
    }
}

impl ResolveSymbols for Struct {
    fn resolve_symbols(self,context: &Context) -> Struct {
        let mut new_fields: Vec<Symbol> = Vec::new();
        for field in self.fields.iter() {
            new_fields.push(Symbol { ident: field.ident,type_: field.type_.resolve_symbols(context), });
        }
        Struct {
            ident: self.ident,
            fields: new_fields,
        }
    }
}

impl ResolveSymbols for Variant {
    fn resolve_symbols(self,context: &Context) -> Variant {
        match self {
            Variant::Naked(ident) => Variant::Naked(ident),
            Variant::Tuple(ident,types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(type_.resolve_symbols(context));
                }
                Variant::Tuple(ident,new_types)
            },
            Variant::Struct(ident,fields) => {
                let mut new_fields: Vec<Symbol> = Vec::new();
                for field in fields.iter() {
                    new_fields.push(Symbol { ident: field.ident,type_: field.type_.resolve_symbols(context), });
                }
                Variant::Struct(ident,new_fields)
            },
        }
    }
}

impl ResolveSymbols for Enum {
    fn resolve_symbols(self,context: &Context) -> Enum {
        let mut new_variants: Vec<Variant> = Vec::new();
        for variant in self.variants.iter() {
            new_variants.push(variant.resolve_symbols(context));
        }
        Enum { ident: self.ident,variants: new_variants, }
    }
}

impl ResolveSymbols for Const {
    fn resolve_symbols(self,context: &Context) -> Const {
        Const {
            ident: self.ident,
            type_: self.type_.resolve_symbols(context),
            expr: self.expr.resolve_symbols(context),
        }
    }
}

impl ResolveSymbols for Alias {
    fn resolve_symbols(self,context: &Context) -> Alias {
        Alias {
            ident: self.ident,
            type_: self.type_.resolve_symbols(context),
        }
    }
}
