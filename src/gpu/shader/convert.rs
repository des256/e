use {
    super::*,
    std::collections::HashMap,
};

use ast::*;

struct Context {
    tuple_structs: HashMap<String,Struct>,
    structs: HashMap<String,Struct>,
    extern_structs: HashMap<String,Struct>,
    enums: HashMap<String,Enum>,
    enum_structs: HashMap<String,Struct>,
    enum_variants: HashMap<String,HashMap<String,usize>>,
    enum_indices: HashMap<String,Vec<Vec<usize>>>,
    aliases: HashMap<String,Alias>,
    consts: HashMap<String,Const>,
    functions: HashMap<String,Function>,
    params: HashMap<String,Symbol>,
    locals: HashMap<String,Symbol>,
    stdlib: StandardLib,
}

impl Context {

    fn empty_struct(&self,struct_: &Struct) -> Expr {
        let mut fields: Vec<(String,Expr)> = Vec::new();
        for field in struct_.fields.iter() {
            fields.push((field.ident.clone(),self.empty_type(&field.type_)));
        }
        Expr::Struct(struct_.ident.clone(),fields)
    }

    fn empty_type(&self,type_: &Type) -> Expr {
        match type_ {
            Type::Inferred => panic!("{}","no empty specified for {inferred}"),
            Type::Void => panic!("no empty specified for ()"),
            Type::Integer => panic!("{}","no empty specified for {integer}"),
            Type::Float => panic!("{}","no empty specified for {float}"),
            Type::Bool => Expr::Boolean(false),
            Type::U8 |
            Type::I8 |
            Type::U16 |
            Type::I16 |
            Type::U32 |
            Type::I32 |
            Type::U64 |
            Type::I64 |
            Type::USize |
            Type::ISize => Expr::Integer(0),
            Type::F16 |
            Type::F32 |
            Type::F64 => Expr::Float(0.0),
            Type::AnonTuple(types) => {
                let mut exprs: Vec<Expr> = Vec::new();
                for type_ in types.iter() {
                    exprs.push(self.empty_type(type_));
                }
                Expr::AnonTuple(exprs)
            },
            Type::Array(type_,expr) => Expr::Cloned(Box::new(self.empty_type(type_)),expr.clone()),
            Type::UnknownIdent(ident) => {
                if self.tuple_structs.contains_key(ident) {
                    self.empty_struct(&self.tuple_structs[ident])
                }
                else if self.structs.contains_key(ident) {
                    self.empty_struct(&self.structs[ident])
                }
                else if self.extern_structs.contains_key(ident) {
                    self.empty_struct(&self.extern_structs[ident])
                }
                else if self.enum_structs.contains_key(ident) {
                    self.empty_struct(&self.enum_structs[ident])
                }
                else if self.aliases.contains_key(ident) {
                    self.empty_type(&self.aliases[ident].type_)
                }
                else if self.stdlib.tuple_structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.tuple_structs[ident])
                }
                else if self.stdlib.structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.structs[ident])
                }
                else if self.stdlib.enum_structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.enum_structs[ident])
                }
                else if self.stdlib.aliases.contains_key(ident) {
                    self.empty_type(&self.stdlib.aliases[ident].type_)
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            }
            Type::Tuple(ident) => {
                if self.tuple_structs.contains_key(ident) {
                    self.empty_struct(&self.tuple_structs[ident])
                }
                else if self.stdlib.tuple_structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.tuple_structs[ident])
                }
                else {
                    panic!("unknown tuple {}",ident);
                }
            },
            Type::Struct(ident) => {
                if self.structs.contains_key(ident) {
                    self.empty_struct(&self.structs[ident])
                }
                else if self.extern_structs.contains_key(ident) {
                    self.empty_struct(&self.extern_structs[ident])
                }
                else if self.stdlib.structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.structs[ident])
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },
            Type::Enum(ident) => {
                if self.enum_structs.contains_key(ident) {
                    self.empty_struct(&self.enum_structs[ident])
                }
                else if self.stdlib.enum_structs.contains_key(ident) {
                    self.empty_struct(&self.stdlib.enum_structs[ident])
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },
            Type::Alias(ident) => {
                if self.aliases.contains_key(ident) {
                    self.empty_type(&self.aliases[ident].type_)
                }
                else if self.stdlib.aliases.contains_key(ident) {
                    self.empty_type(&self.stdlib.aliases[ident].type_)
                }
                else {
                    panic!("unknown identifier {}",ident);
                }

            },
        }
    }

    fn process_type(&mut self,type_: &Type) -> Type {
        match type_ {
            Type::Inferred |
            Type::Void |
            Type::Integer |
            Type::Float |
            Type::Bool |
            Type::U8 |
            Type::I8 |
            Type::U16 |
            Type::I16 |
            Type::U32 |
            Type::I32 |
            Type::U64 |
            Type::I64 |
            Type::USize |
            Type::ISize |
            Type::F16 |
            Type::F32 |
            Type::F64 => type_.clone(),
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types.iter() {
                    new_types.push(self.process_type(type_));
                }
                Type::AnonTuple(new_types)
            },
            Type::Array(type_,expr) => {
                let new_type = self.process_type(type_);
                let new_expr = self.process_expr(expr);
                Type::Array(Box::new(new_type),Box::new(new_expr))
            },
            Type::UnknownIdent(ident) => {
                if self.tuple_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.extern_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.enum_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.aliases.contains_key(ident) {
                    self.aliases[ident].type_.clone()
                }
                else if self.stdlib.tuple_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.enum_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.aliases.contains_key(ident) {
                    self.stdlib.aliases[ident].type_.clone()
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },
            Type::Tuple(ident) => {
                if self.tuple_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.tuple_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else {
                    panic!("unknown tuple {}",ident);
                }
            },
            Type::Struct(ident) => {
                if self.structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.extern_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },
            Type::Enum(ident) => {
                if self.enum_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else if self.stdlib.enum_structs.contains_key(ident) {
                    Type::Struct(ident.clone())
                }
                else {
                    panic!("unknown enum {}",ident);
                }
            },
            Type::Alias(ident) => {
                if self.aliases.contains_key(ident) {
                    self.aliases[ident].type_.clone()
                }
                else if self.stdlib.aliases.contains_key(ident) {
                    self.stdlib.aliases[ident].type_.clone()
                }
                else {
                    panic!("unknown alias {}",ident);
                }
            },
        }
    }

    fn process_expr(&mut self,expr: &Expr) -> Expr {
        match expr {
            Expr::Boolean(value) => Expr::Boolean(*value),
            Expr::Integer(value) => Expr::Integer(*value),
            Expr::Float(value) => Expr::Float(*value),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Array(new_exprs)
            },
            Expr::Cloned(expr,expr2) => {
                let new_expr = self.process_expr(expr);
                let new_expr2 = self.process_expr(expr2);
                Expr::Cloned(Box::new(new_expr),Box::new(new_expr2))
            },
            Expr::Index(expr,expr2) => {
                let new_expr = self.process_expr(expr);
                let new_expr2 = self.process_expr(expr2);
                Expr::Index(Box::new(new_expr),Box::new(new_expr2))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.process_expr(expr);
                let new_type = self.process_type(type_);
                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },
            Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::AnonTuple(new_exprs)            
            },
            Expr::Unary(op,expr) => {
                let new_expr = self.process_expr(expr);
                Expr::Unary(
                    op.clone(),
                    Box::new(new_expr)
                )
            },
            Expr::Binary(expr,op,expr2) => {
                let new_expr = self.process_expr(expr);
                let new_expr2 = self.process_expr(expr2);
                Expr::Binary(
                    Box::new(new_expr),
                    op.clone(),
                    Box::new(new_expr2)
                )
            },
            Expr::Continue => Expr::Continue,
            Expr::Break(expr) => if let Some(expr) = expr {
                Expr::Break(Some(Box::new(self.process_expr(expr))))
            }
            else {
                Expr::Break(None)
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                Expr::Return(Some(Box::new(self.process_expr(expr))))
            }
            else {
                Expr::Return(None)
            },
            Expr::Block(block) => Expr::Block(self.process_block(block)),
            Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr {
                Expr::If(
                    Box::new(self.process_expr(expr)),
                    self.process_block(block),
                    Some(Box::new(self.process_expr(else_expr)))
                )
            }
            else {
                Expr::If(
                    Box::new(self.process_expr(expr)),
                    self.process_block(block),
                    None
                )
            },
            Expr::While(expr,block) => Expr::While(
                Box::new(self.process_expr(expr)),
                self.process_block(block),
            ),
            Expr::Loop(block) => Expr::Loop(self.process_block(block)),
            Expr::UnknownIdent(ident) => if self.params.contains_key(ident) {
                Expr::Param(ident.clone())
            }
            else if self.locals.contains_key(ident) {
                Expr::Local(ident.clone())
            }
            else if self.consts.contains_key(ident) {
                Expr::Const(ident.clone())
            }
            else if self.stdlib.consts.contains_key(ident) {
                Expr::Const(ident.clone())
            }
            else {
                panic!("unknown identifier {}",ident);
            },
            Expr::TupleOrCall(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for i in 0..exprs.len() {
                    let new_expr = self.process_expr(&exprs[i]);
                    new_exprs.push(new_expr.clone());
                    new_fields.push((format!("_{}",i),new_expr));
                }
                if self.tuple_structs.contains_key(ident) {
                    Expr::Struct(ident.clone(),new_fields)
                }
                else if self.functions.contains_key(ident) {
                    Expr::Call(ident.clone(),new_exprs)
                }
                else if self.stdlib.tuple_structs.contains_key(ident) {
                    Expr::Struct(ident.clone(),new_fields)
                }
                else if self.stdlib.functions.contains_key(ident) {
                    Expr::Call(ident.clone(),new_exprs)
                }
                else {
                    panic!("unknown tuple or function {}",ident);
                }
            },
            Expr::Struct(ident,fields) => {
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for i in 0..fields.len() {
                    let new_expr = self.process_expr(&fields[i].1);
                    new_fields.push((format!("_{}",i),new_expr));
                }
                Expr::Struct(ident.clone(),new_fields)
            },
            Expr::Variant(ident,variant_expr) => {
                let (struct_,variants,indices) = if self.enums.contains_key(ident) {
                    (&self.enum_structs[ident],&self.enum_variants[ident],&self.enum_indices[ident])
                }
                else if self.stdlib.enums.contains_key(ident) {
                    (&self.stdlib.structs[ident],&self.stdlib.enum_variants[ident],&self.stdlib.enum_indices[ident])
                }
                else {
                    panic!("unknown enum {}",ident)
                };
                let variant_ident = match variant_expr {
                    VariantExpr::Naked(ident) |
                    VariantExpr::Tuple(ident,_) |
                    VariantExpr::Struct(ident,_) => ident,
                };
                if variants.contains_key(variant_ident) {
                    let variant_index = variants[variant_ident];
                    let mut new_fields: Vec<(String,Expr)> = Vec::new();
                    for field in struct_.fields.iter() {
                        new_fields.push((field.ident.clone(),self.empty_type(&field.type_)));
                    }
                    match variant_expr {
                        VariantExpr::Naked(_) => { },
                        VariantExpr::Tuple(_,exprs) => {
                            for i in 0..exprs.len() {
                                new_fields[indices[variant_index][i]].1 = exprs[i].clone();
                            }
                        },
                        VariantExpr::Struct(_,fields) => {
                            for i in 0..fields.len() {
                                new_fields[indices[variant_index][i]].1 = fields[i].1.clone();
                            }
                        },
                    }
                    Expr::Struct(ident.clone(),new_fields)
                }
                else {
                    panic!("unknown variant {}::{}",ident,variant_ident);
                }
            },
            Expr::Method(expr,ident,exprs) => {
                let new_expr = self.process_expr(expr);
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Method(Box::new(new_expr),ident.clone(),new_exprs)
            },
            Expr::Field(expr,ident) => {
                let new_expr = self.process_expr(expr);
                Expr::Field(Box::new(new_expr),ident.clone())
            },
            Expr::TupleIndex(expr,index) => {
                let new_expr = self.process_expr(expr);
                Expr::Field(Box::new(new_expr),format!("_{}",index))
            },
            Expr::Param(ident) => Expr::Param(ident.clone()),
            Expr::Local(ident) => Expr::Local(ident.clone()),
            Expr::Const(ident) => Expr::Const(ident.clone()),
            Expr::Tuple(ident,exprs) => {
                let mut new_fields: Vec<(String,Expr)> = Vec::new();
                for i in 0..exprs.len() {
                    new_fields.push((format!("_{}",i),self.process_expr(&exprs[i])));
                }
                Expr::Struct(ident.clone(),new_fields)
            },
            Expr::Call(ident,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.process_expr(expr));
                }
                Expr::Call(ident.clone(),new_exprs)
            },
            Expr::Discriminant(expr,variant_ident) => {
                let new_expr = self.process_expr(expr);
                Expr::Discriminant(Box::new(new_expr),variant_ident.clone())
            },
            Expr::DestructTuple(expr,variant_ident,index) => {
                let new_expr = self.process_expr(expr);
                Expr::DestructTuple(Box::new(new_expr),variant_ident.clone(),*index)
            },
            Expr::DestructStruct(expr,variant_ident,ident) => {
                let new_expr = self.process_expr(expr);
                Expr::DestructStruct(Box::new(new_expr),variant_ident.clone(),ident.clone())
            },
            _ => panic!("illegal expression at this stage: {}",expr),
        }
    }

    fn process_block(&mut self,block: &Block) -> Block {
        let prev_locals = self.locals.clone();
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            new_stats.push(match stat {
                Stat::Expr(expr) => {
                    let new_expr = self.process_expr(expr);
                    Stat::Expr(Box::new(new_expr))
                },
                Stat::Local(ident,type_,expr) => {
                    let new_type = self.process_type(type_);
                    let new_expr = self.process_expr(expr);
                    self.locals.insert(ident.clone(),Symbol { ident: ident.clone(),type_: new_type.clone(), });
                    Stat::Local(ident.clone(),Box::new(new_type),Box::new(new_expr))
                },
                _ => {
                    panic!("illegal statement at this stage: {}",stat);
                },
            });
        }
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.process_expr(expr)))
        }
        else {
            None
        };
        self.locals = prev_locals;
        Block {
            stats: new_stats,
            expr: new_expr,
        }
    }

    fn count_variant_types(type_counts: &mut Vec<(Type,usize)>,type_: &Type) {
        let mut found = false;
        for type_count in type_counts.iter_mut() {
            if *type_ == type_count.0 {
                (*type_count).1 += 1;
                found = true;
                break;
            }
        }
        if !found {
            type_counts.push((type_.clone(),1));
        }
    }

    fn wrap_minimal(total_type_counts: &mut Vec<(Type,usize)>,type_counts: &Vec<(Type,usize)>) {
        for type_count in type_counts.iter() {
            let mut found = false;
            for total_type_count in total_type_counts.iter_mut() {
                if total_type_count.0 == type_count.0 {
                    found = true;
                    if type_count.1 > total_type_count.1 {
                        (*total_type_count).1 = type_count.1;
                    }
                    break;
                }
            }
            if !found {
                total_type_counts.push((type_count.0.clone(),type_count.1.clone()));
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

    pub fn convert_enum(enum_: &Enum) -> (Struct,HashMap<String,usize>,Vec<Vec<usize>>) {
        let mut total_type_counts: Vec<(Type,usize)> = Vec::new();
        for i in 0..enum_.variants.len() {
            let mut type_counts: Vec<(Type,usize)> = Vec::new();
            match &enum_.variants[i] {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for type_ in types.iter() {
                        Self::count_variant_types(&mut type_counts,type_);
                    }
                },
                Variant::Struct(_,fields) => {
                    for field in fields.iter() {
                        Self::count_variant_types(&mut type_counts,&field.type_);
                    }
                },
            }
            Self::wrap_minimal(&mut total_type_counts,&type_counts);
        }
        let mut fields: Vec<Symbol> = Vec::new();
        fields.push(Symbol { ident: "discr".to_string(),type_: Type::U32, });
        let mut count = 0usize;
        for total_type_count in total_type_counts.iter() {
            let ident = format!("_{}",count);
            fields.push(Symbol { ident,type_: total_type_count.0.clone(), });
            count += total_type_count.1;
        }
        let mut enum_variants: HashMap<String,usize> = HashMap::new();
        let mut enum_indices: Vec<Vec<usize>> = Vec::new();
        for i in 0..enum_.variants.len() {
            match &enum_.variants[i] {
                Variant::Naked(variant_ident) => {
                    enum_variants.insert(variant_ident.clone(),i);
                },
                Variant::Tuple(variant_ident,types) => {
                    enum_variants.insert(variant_ident.clone(),i);
                    let mut prev_type: Option<Type> = None;
                    let mut prev_index = 0usize;
                    let mut variant_indices: Vec<usize> = Vec::new();
                    for type_ in types.iter() {
                        variant_indices.push(Self::fit_index(type_,&mut prev_type,&mut prev_index,&total_type_counts));
                    }
                    enum_indices.push(variant_indices);
                },
                Variant::Struct(variant_ident,fields) => {
                    enum_variants.insert(variant_ident.clone(),i);
                    let mut prev_type: Option<Type> = None;
                    let mut prev_index = 0usize;
                    let mut variant_indices: Vec<usize> = Vec::new();
                    for field in fields.iter() {
                        variant_indices.push(Self::fit_index(&field.type_,&mut prev_type,&mut prev_index,&total_type_counts));                            
                    }
                    enum_indices.push(variant_indices);
                },
            }
        }
        (Struct { ident: enum_.ident.clone(),fields, },enum_variants,enum_indices)
    }

    pub fn process_module(module: DestructuredModule) -> ConvertedModule {

        let mut context = Context {
            tuple_structs: HashMap::new(),
            structs: HashMap::new(),
            extern_structs: HashMap::new(),
            enums: HashMap::new(),
            enum_structs: HashMap::new(),
            enum_variants: HashMap::new(),
            enum_indices: HashMap::new(),
            aliases: HashMap::new(),
            consts: HashMap::new(),
            functions: HashMap::new(),
            params: HashMap::new(),
            locals: HashMap::new(),
            stdlib: StandardLib::new(),
        };

        // prepare the context
        for tuple in module.tuples.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for i in 0..tuple.types.len() {
                new_fields.push(Symbol { ident: format!("_{}",i),type_: tuple.types[i].clone(), });
            }
            context.tuple_structs.insert(tuple.ident.clone(),Struct { ident: tuple.ident.clone(),fields: new_fields, });
        }
        for struct_ in module.structs.values() {
            context.structs.insert(struct_.ident.clone(),struct_.clone());
        }
        for struct_ in module.extern_structs.values() {
            context.extern_structs.insert(struct_.ident.clone(),struct_.clone());
        }
        for enum_ in module.enums.values() {
            let (struct_,variants,indices) = convert_enum(enum_);
            context.enum_structs.insert(enum_.ident.clone(),struct_);
            context.enum_variants.insert(enum_.ident.clone(),variants);
            context.enum_indices.insert(enum_.ident.clone(),indices);
        }
        for alias in module.aliases.values() {
            let mut type_ = alias.type_.clone();
            while let Type::Alias(ident) = type_ {
                if module.aliases.contains_key(&ident) {
                    type_ = module.aliases[&ident].type_.clone();
                }
                else if context.stdlib.aliases.contains_key(&ident) {
                    type_ = context.stdlib.aliases[&ident].type_.clone();
                }
                else {
                    panic!("unknown type alias {}",ident);
                }
            }
            context.aliases.insert(alias.ident.clone(),Alias { ident: alias.ident.clone(),type_, });
        }
        for const_ in module.consts.values() {
            context.consts.insert(const_.ident.clone(),const_.clone());
        }
        for function in module.functions.values() {
            context.functions.insert(function.ident.clone(),function.clone());
        }

        // process all types and expressions
        let mut new_structs: HashMap<String,Struct> = HashMap::new();
        for struct_ in context.tuple_structs.clone().values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(&field.type_),
                });
            }
            new_structs.insert(struct_.ident.clone(),Struct { ident: struct_.ident.clone(),fields: new_fields, });
        }
        for struct_ in context.structs.clone().values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(&field.type_),
                });
            }
            new_structs.insert(struct_.ident.clone(),Struct { ident: struct_.ident.clone(),fields: new_fields, });
        }
        for struct_ in context.extern_structs.clone().values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(&field.type_),
                });
            }
            new_structs.insert(struct_.ident.clone(),Struct { ident: struct_.ident.clone(),fields: new_fields, });
        }
        for struct_ in context.enum_structs.clone().values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.fields.iter() {
                new_fields.push(Symbol {
                    ident: field.ident.clone(),
                    type_: context.process_type(&field.type_),
                });
            }
            new_structs.insert(struct_.ident.clone(),Struct { ident: struct_.ident.clone(),fields: new_fields, });
        }
        let mut new_consts: HashMap<String,Const> = HashMap::new();
        for const_ in context.consts.clone().values() {
            let new_type = context.process_type(&const_.type_);
            let new_expr = context.process_expr(&const_.expr);            
            new_consts.insert(const_.ident.clone(),Const { ident: const_.ident.clone(),type_: new_type,expr: new_expr, });
        }
        let mut new_functions: HashMap<String,Function> = HashMap::new();
        for function in context.functions.clone().values() {
            let mut new_params: Vec<Symbol> = Vec::new();
            context.params.clear();
            for param in function.params.iter() {
                let param = Symbol {
                    ident: param.ident.clone(),
                    type_: context.process_type(&param.type_),
                };
                context.params.insert(param.ident.clone(),param.clone());
                new_params.push(param);
            }
            let new_type = context.process_type(&function.type_);
            let new_block = context.process_block(&function.block);
            new_functions.insert(function.ident.clone(),Function {
                ident: function.ident.clone(),
                params: new_params,
                type_: new_type,
                block: new_block,
            });
        }

        ConvertedModule {
            ident: module.ident.clone(),
            structs: new_structs,
            enums: context.enums,
            enum_variants: context.enum_variants,
            enum_indices: context.enum_indices,
            consts: new_consts,
            functions: new_functions,
        }
    }
}

pub fn convert_module(module: DestructuredModule) -> ConvertedModule {
    Context::process_module(module)
}

pub fn convert_enum(enum_: &Enum) -> (Struct,HashMap<String,usize>,Vec<Vec<usize>>) {
    Context::convert_enum(enum_)   
}
