use {
    super::*,
    std::{
        rc::Rc,
        collections::HashMap,
        cell::RefCell,
    }
};

use ast::*;

pub struct Resolver {
    pub stdlib_structs: HashMap<String,Struct>,
    pub stdlib_enum_indices: HashMap<String,Vec<Vec<usize>>>,
    pub stdlib_consts: HashMap<String,Const>,
    pub stdlib_functions: HashMap<String,Vec<Function>>,
    pub stdlib_methods: HashMap<String,Vec<Method>>,
    pub source_structs: HashMap<String,Struct>,
    pub source_enum_indices: HashMap<String,Vec<Vec<usize>>>,
    pub source_consts: HashMap<String,Const>,
    pub source_functions: HashMap<String,Function>,
    pub locals: HashMap<String,Symbol>,
    pub params: HashMap<String,Symbol>,
}

impl Resolver {

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

    pub fn new(source: &Source) -> Resolver {

        let stdlib = StandardLib::new();

        let stdlib_structs: HashMap<String,Struct> = HashMap::new();
        let stdlib_enum_indices: HashMap<String,Vec<Vec<usize>>> = HashMap::new();

        // convert stdlib tuples to structs
        for tuple in stdlib.tuples.values() {
            stdlib_structs.insert(tuple.ident.clone(),Resolver::convert_tuple(tuple));
        }

        // insert stdlib structs
        for struct_ in stdlib.structs.values() {
            stdlib_structs.insert(struct_.ident.clone(),*struct_.clone());
        }

        // convert stdlib enums to structs
        for enum_ in stdlib.enums.values() {
            Resolver::convert_enum(&enum_,&mut stdlib_structs,&mut stdlib_enum_indices);
        }

        // insert stdlib consts
        let stdlib_consts: HashMap<String,Const> = HashMap::new();
        for const_ in stdlib.consts.values() {
            stdlib_consts.insert(const_.ident.clone(),*const_.clone());
        }

        // insert stdlib functions
        let stdlib_functions: HashMap<String,Vec<Function>> = HashMap::new();
        for functions in stdlib.functions.values() {
            let overloaded_functions: Vec<Function> = Vec::new();
            let mut ident = String::new();
            for function in functions.iter() {
                ident = function.ident.clone();
                overloaded_functions.push(*function.clone());
            }
            stdlib_functions.insert(ident.clone(),overloaded_functions);
        }

        // insert stdlib methods
        let stdlib_methods: HashMap<String,Vec<Method>> = HashMap::new();
        for methods in stdlib.methods.values() {
            let overloaded_methods: Vec<Method> = Vec::new();
            let mut ident = String::new();
            for method in methods.iter() {
                ident = method.ident.clone();
                overloaded_methods.push(*method.clone());
            }
            stdlib_methods.insert(ident.clone(),overloaded_methods);
        }

        let source_structs: HashMap<String,Struct> = HashMap::new();
        let source_enum_indices: HashMap<String,Vec<Vec<usize>>> = HashMap::new();

        // convert tuples to structs
        for tuple in source.tuples.iter() {
            source_structs.insert(tuple.ident.clone(),Resolver::convert_tuple(tuple));
        }

        // insert structs
        for struct_ in source.structs.iter() {
            source_structs.insert(struct_.ident.clone(),struct_.clone());
        }

        // convert enums to structs
        for enum_ in source.enums.iter() {
            Resolver::convert_enum(&enum_,&mut source_structs,&mut source_enum_indices);
        }

        // insert consts
        let source_consts: HashMap<String,Const> = HashMap::new();
        for const_ in source.consts.iter() {
            source_consts.insert(const_.ident.clone(),const_.clone());
        }

        // insert functions
        let source_functions: HashMap<String,Function> = HashMap::new();
        for function in source.functions.iter() {
            source_functions.insert(function.ident.clone(),function.clone());
        }
        
        Resolver {
            stdlib_structs,
            stdlib_enum_indices,
            stdlib_consts,
            stdlib_functions,
            stdlib_methods,
            source_structs,
            source_enum_indices,
            source_consts,
            source_functions,
            locals: HashMap::new(),
            params: HashMap::new(),
        }
    }

    pub fn resolve_type(&mut self,type_: Type) -> Type {

        // Type::UnknownIdent => Type::Tuple, Type::Struct, Type::Enum or Type::Alias
        // Type::Tuple => Type::Struct
        // Type::Enum => Type::Struct
        
        match type_ {

            // inferred, void, literal and base types can't refer to symbols
            Type::Inferred | Type::Void |
            Type::Integer | Type::Float |
            Type::Bool |
            Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::USize | Type::ISize |
            Type::F16 | Type::F32 | Type::F64 => type_,

            // anonymous tuple type: resolve symbols on all elements
            Type::AnonTuple(types) => {
                let mut new_types: Vec<Type> = Vec::new();
                for type_ in types {
                    new_types.push(self.resolve_type(type_));
                }
                Type::AnonTuple(new_types)
            },

            // array type: resolve symbols on type and expression
            Type::Array(type_,expr) => {
                let new_type = self.resolve_type(*type_);
                let new_expr = self.resolve_expr(*expr);
                Type::Array(Box::new(new_type),Box::new(new_expr))
            },

            // unknown identifier type: resolve to Tuple, Struct, Enum or Alias references
            Type::UnknownIdent(ident) => {
                if self.tuples.contains_key(&ident) {
                    let tuple = Rc::clone(&self.tuples[&ident]);
                    Type::Tuple(tuple)
                }
                else if self.structs.contains_key(&ident) {
                    let struct_ = Rc::clone(&self.structs[&ident]);
                    Type::Struct(struct_)
                }
                else if self.enums.contains_key(&ident) {
                    let enum_ = Rc::clone(&self.enums[&ident]);
                    Type::Enum(enum_)
                }
                else if self.aliases.contains_key(&ident) {
                    let alias = Rc::clone(&self.aliases[&ident]);
                    Type::Alias(alias)
                }
                else {
                    panic!("unknown type {}",ident);
                }
            },

            // tuple, struct, enum or alias references: don't contain symbols any more
            Type::Tuple(_) |
            Type::Struct(_) |
            Type::Enum(_) |
            Type::Alias(_) => type_,
        }
    }

    pub fn resolve_pat(&mut self,pat: Pat) -> Pat {

        match pat {

            Pat::Wildcard |
            Pat::Rest |
            Pat::Boolean(_) |
            Pat::Integer(_) |
            Pat::Float(_) => pat,

            Pat::AnonTuple(elements) => {
                let mut new_elements: Vec<Pat> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_pat(element));
                }
                Pat::AnonTuple(new_elements)
            },

            Pat::Array(elements) => {
                let mut new_elements: Vec<Pat> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_pat(element));
                }
                Pat::Array(new_elements)
            },

            Pat::Range(from,to) => {
                let new_from = self.resolve_pat(*from);
                let new_to = self.resolve_pat(*to);
                Pat::Range(Box::new(new_from),Box::new(new_to))
            },

            Pat::UnknownIdent(_) => pat,

            Pat::UnknownTuple(ident,elements) => {
                if self.tuples.contains_key(&ident) {
                    let tuple = Rc::clone(&self.tuples[&ident]);
                    let mut new_elements: Vec<Pat> = Vec::new();
                    for element in elements {
                        new_elements.push(self.resolve_pat(element));
                    }
                    Pat::Tuple(tuple,new_elements)
                }
                else {
                    panic!("unknown tuple {}",ident);
                }
            },

            Pat::UnknownStruct(ident,elements) => {
                if self.structs.contains_key(&ident) {
                    let struct_ = Rc::clone(&self.structs[&ident]);
                    let mut new_elements: Vec<FieldPat> = Vec::new();
                    for element in elements {
                        new_elements.push(match element {
                            UnknownFieldPat::Wildcard => FieldPat::Wildcard,
                            UnknownFieldPat::Rest => FieldPat::Rest,
                            UnknownFieldPat::Ident(ident) => {
                                if let Some(index) = struct_.borrow().find_field(&ident) {
                                    FieldPat::Index(index)
                                }
                                else {
                                    panic!("field {} not found in struct {}",ident,struct_.borrow().ident);
                                }
                            },
                            UnknownFieldPat::IdentPat(ident,pat) => {
                                if let Some(index) = struct_.borrow().find_field(&ident) {
                                    let new_pat = self.resolve_pat(pat);
                                    FieldPat::IndexPat(index,new_pat)
                                }
                                else {
                                    panic!("struct {} does not contain field {}",struct_.borrow().ident,ident);
                                }
                            },
                        });
                    }
                    Pat::Struct(struct_,new_elements)
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },

            Pat::UnknownVariant(enum_ident,variant) => {
                if self.enums.contains_key(&enum_ident) {
                    let enum_ = Rc::clone(&self.enums[&enum_ident]);
                    let new_variant = match variant {
                        UnknownVariantPat::Naked(ident) => {
                            if let Some(index) = enum_.borrow().find_naked_variant(&ident) {
                                VariantPat::Naked(index)
                            }
                            else {
                                panic!("enum {} does not contain naked variant {}",enum_.borrow().ident,ident);
                            }
                        },
                        UnknownVariantPat::Tuple(ident,elements) => {
                            if let Some(index) = enum_.borrow().find_tuple_variant(&ident) {
                                let mut new_elements: Vec<Pat> = Vec::new();
                                for element in elements {
                                    new_elements.push(self.resolve_pat(element));
                                }
                                VariantPat::Tuple(index,new_elements)
                            }
                            else {
                                panic!("enum {} does not contain tuple variant {}",enum_.borrow().ident,ident);
                            }
                        },
                        UnknownVariantPat::Struct(variant_ident,elements) => {
                            if let Some(variant_index) = enum_.borrow().find_struct_variant(&variant_ident) {
                                let mut new_elements: Vec<FieldPat> = Vec::new();
                                for element in elements {
                                    new_elements.push(match element {
                                        UnknownFieldPat::Wildcard => FieldPat::Wildcard,
                                        UnknownFieldPat::Rest => FieldPat::Rest,
                                        UnknownFieldPat::Ident(ident) => {
                                            if let Some(index) = enum_.borrow().variants[variant_index].find_struct_element(&ident) {
                                                FieldPat::Index(index)
                                            }
                                            else {
                                                panic!("struct variant {}::{} does not contain field {}",enum_.borrow().ident,variant_ident,ident);
                                            }
                                        },
                                        UnknownFieldPat::IdentPat(ident,pat) => {
                                            if let Some(index) = enum_.borrow().variants[variant_index].find_struct_element(&ident) {
                                                let new_pat = self.resolve_pat(pat);
                                                FieldPat::IndexPat(index,new_pat)
                                            }
                                            else {
                                                panic!("struct variant {}::{} does not contain field {}",enum_.borrow().ident,variant_ident,ident);
                                            }
                                        },
                                    });
                                }
                                VariantPat::Struct(variant_index,new_elements)
                            }
                            else {
                                panic!("enum {} does not contain struct variant {}",enum_.borrow().ident,variant_ident);
                            }
                        }
                    };
                    Pat::Variant(enum_,new_variant)
                }
                else {
                    panic!("unknown enum {}",enum_ident);
                }
            },

            Pat::Tuple(ident,elements) => {
                let mut new_elements: Vec<Pat> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_pat(element));
                }
                Pat::Tuple(ident,new_elements)
            },

            Pat::Struct(ident,elements) => {
                let mut new_elements: Vec<FieldPat> = Vec::new();
                for element in elements {
                    new_elements.push(match element {
                        FieldPat::Wildcard |
                        FieldPat::Rest |
                        FieldPat::Index(_) => element,
                        FieldPat::IndexPat(index,pat) => FieldPat::IndexPat(index,self.resolve_pat(pat)),
                    });
                }
                Pat::Struct(ident,new_elements)
            },

            Pat::Variant(enum_,variantpat) => {
                let new_variantpat = match variantpat {
                    VariantPat::Naked(_) => variantpat,
                    VariantPat::Tuple(index,elements) => {
                        let mut new_elements: Vec<Pat> = Vec::new();
                        for element in elements {
                            new_elements.push(self.resolve_pat(element));
                        }
                        VariantPat::Tuple(index,new_elements)
                    },
                    VariantPat::Struct(index,elements) => {
                        let mut new_elements: Vec<FieldPat> = Vec::new();
                        for element in elements {
                            new_elements.push(match element {
                                FieldPat::Wildcard |
                                FieldPat::Rest |
                                FieldPat::Index(_) => element,
                                FieldPat::IndexPat(index,pat) => FieldPat::IndexPat(index,self.resolve_pat(pat)),
                            });
                        }
                        VariantPat::Struct(index,new_elements)
                    },
                };
                Pat::Variant(enum_,new_variantpat)
            },
        }
    }

    pub fn resolve_expr(&mut self,expr: Expr) -> Expr {

        match expr {

            Expr::Boolean(_) |
            Expr::Integer(_) |
            Expr::Float(_) => expr,

            Expr::Array(elements) => {
                let mut new_elements: Vec<Expr> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_expr(element));
                }
                Expr::Array(new_elements)
            },

            Expr::Cloned(expr,count) => {
                let new_expr = self.resolve_expr(*expr);
                let new_count = self.resolve_expr(*count);
                Expr::Cloned(Box::new(new_expr),Box::new(new_count))
            },

            Expr::Index(expr,index) => {
                let new_expr = self.resolve_expr(*expr);
                let new_index = self.resolve_expr(*index);
                Expr::Index(Box::new(new_expr),Box::new(new_index))
            },

            Expr::Cast(expr,type_) => {
                let new_expr = self.resolve_expr(*expr);
                let new_type = self.resolve_type(*type_);
                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },

            Expr::AnonTuple(elements) => {
                let mut new_elements: Vec<Expr> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_expr(element));
                }
                Expr::AnonTuple(new_elements)
            },

            Expr::Unary(op,expr) => {
                let new_expr = self.resolve_expr(*expr);
                Expr::Unary(op,Box::new(new_expr))
            },

            Expr::Binary(lhs,op,rhs) => {
                let lhs = self.resolve_expr(*lhs);
                let rhs = self.resolve_expr(*rhs);
                Expr::Binary(Box::new(lhs),op,Box::new(rhs))
            },

            Expr::Continue => Expr::Continue,

            Expr::Break(expr) => if let Some(expr) = expr {
                let new_expr = self.resolve_expr(*expr);
                Expr::Break(Some(Box::new(new_expr)))
            }
            else {
                Expr::Break(None)
            },

            Expr::Return(expr) => if let Some(expr) = expr {
                let new_expr = self.resolve_expr(*expr);
                Expr::Return(Some(Box::new(new_expr)))
            }
            else {
                Expr::Return(None)
            },

            Expr::Block(block) => {
                let new_block = self.resolve_block(block);
                Expr::Block(new_block)
            },

            Expr::If(expr,block,else_expr) => {
                let new_expr = self.resolve_expr(*expr);
                let new_block = self.resolve_block(block);
                if let Some(else_expr) = else_expr {
                    let new_else_expr = self.resolve_expr(*else_expr);
                    Expr::If(Box::new(new_expr),new_block,Some(Box::new(new_else_expr)))
                }
                else {
                    Expr::If(Box::new(new_expr),new_block,None)
                }
            },

            Expr::Loop(block) => {
                let new_block = self.resolve_block(block);
                Expr::Loop(new_block)
            },

            Expr::While(expr,block) => {
                let new_expr = self.resolve_expr(*expr);
                let new_block = self.resolve_block(block);
                Expr::While(Box::new(new_expr),new_block)
            },

            Expr::IfLet(pats,expr,block,else_expr) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.resolve_pat(pat));
                }
                let new_expr = self.resolve_expr(*expr);
                let new_block = self.resolve_block(block);
                if let Some(else_expr) = else_expr {
                    let new_else_expr = self.resolve_expr(*else_expr);
                    Expr::IfLet(new_pats,Box::new(new_expr),new_block,Some(Box::new(new_else_expr)))
                }
                else {
                    Expr::IfLet(new_pats,Box::new(new_expr),new_block,None)
                }
            },

            Expr::For(pats,range,block) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.resolve_pat(pat));
                }
                let new_range = self.resolve_range(range);
                let new_block = self.resolve_block(block);
                Expr::For(new_pats,new_range,new_block)
            },

            Expr::WhileLet(pats,expr,block) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats {
                    new_pats.push(self.resolve_pat(pat));
                }
                let new_expr = self.resolve_expr(*expr);
                let new_block = self.resolve_block(block);
                Expr::WhileLet(new_pats,Box::new(new_expr),new_block)
            },

            Expr::Match(expr,arms) => {
                let new_expr = self.resolve_expr(*expr);
                let mut new_arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
                for (pats,if_expr,expr) in arms {
                    let mut new_pats: Vec<Pat> = Vec::new();
                    for pat in pats {
                        new_pats.push(self.resolve_pat(pat));
                    }
                    let new_if_expr = if let Some(if_expr) = if_expr {
                        let new_if_expr = self.resolve_expr(*if_expr);
                        Some(Box::new(new_if_expr))
                    }
                    else {
                        None
                    };
                    let new_expr = self.resolve_expr(*expr);
                    new_arms.push((new_pats,new_if_expr,Box::new(new_expr)));
                }
                Expr::Match(Box::new(new_expr),new_arms)
            },

            Expr::UnknownIdent(ident) => {
                if self.consts.contains_key(&ident) {
                    let const_ = Rc::clone(&self.consts[&ident]);
                    Expr::Const(const_)
                }
                else if self.locals.contains_key(&ident) {
                    let local = Rc::clone(&self.locals[&ident]);
                    Expr::Local(local)
                }
                else if self.params.contains_key(&ident) {
                    let param = Rc::clone(&self.params[&ident]);
                    Expr::Param(param)
                }
                else {
                    panic!("unknown identifier {}",ident);
                }
            },

            Expr::UnknownTupleOrCall(ident,elements) => {
                if self.tuples.contains_key(&ident) {
                    let tuple = Rc::clone(&self.tuples[&ident]);
                    let mut new_elements: Vec<Expr> = Vec::new();
                    for element in elements {
                        new_elements.push(self.resolve_expr(element));
                    }
                    Expr::Tuple(tuple,new_elements)
                }
                else if self.functions.contains_key(&ident) {
                    let mut new_elements: Vec<Expr> = Vec::new();
                    for element in elements {
                        new_elements.push(self.resolve_expr(element));
                    }
                    let mut function: Option<Rc<RefCell<Function>>> = None;
                    let functions = &self.functions[&ident];
                    for f in functions.iter() {
                        if f.borrow().params.len() == new_elements.len() {
                            let mut found = true;
                            for i in 0..new_elements.len() {
                                if new_elements[i].find_type() != f.borrow().params[i].borrow().type_ {
                                    found = false;
                                    break;
                                }
                            }
                            if found {
                                function = Some(Rc::clone(&f));
                                break;
                            }
                        }
                    }
                    if let Some(function) = function {
                        Expr::Call(function,new_elements)
                    }
                    else {
                        panic!("unknown function {} for given parameters",ident);
                    }
                }
                else {
                    panic!("unknown tuple or function {}",ident);
                }
            },

            Expr::UnknownStruct(ident,elements) => {
                if self.structs.contains_key(&ident) {
                    let struct_ = Rc::clone(&self.structs[&ident]);
                    let mut new_elements: Vec<Expr> = Vec::new();
                    for element in elements {
                        let new_expr = self.resolve_expr(element.1);
                        new_elements.push(new_expr);
                    }
                    Expr::Struct(struct_,new_elements)
                }
                else {
                    panic!("unknown struct {}",ident);
                }
            },

            Expr::UnknownVariant(ident,variant) => {
                if self.enums.contains_key(&ident) {
                    let enum_ = Rc::clone(&self.enums[&ident]);
                    let new_variant = match variant {
                        UnknownVariantExpr::Naked(ident) => {
                            if let Some(index) = enum_.borrow().find_naked_variant(&ident) {
                                VariantExpr::Naked(index)
                            }
                            else {
                                panic!("enum {} does not contain naked variant {}",enum_.borrow().ident,ident);
                            }
                        },
                        UnknownVariantExpr::Tuple(ident,elements) => {
                            if let Some(index) = enum_.borrow().find_tuple_variant(&ident) {
                                let mut new_elements: Vec<Expr> = Vec::new();
                                for element in elements {
                                    new_elements.push(self.resolve_expr(element));
                                }
                                VariantExpr::Tuple(index,new_elements)
                            }
                            else {
                                panic!("enum {} does not contain tuple variant {}",enum_.borrow().ident,ident);
                            }
                        },
                        UnknownVariantExpr::Struct(variant_ident,elements) => {
                            if let Some(variant_index) = enum_.borrow().find_struct_variant(&variant_ident) {
                                let mut new_elements: Vec<Expr> = vec![Expr::Continue; elements.len()];
                                for element in elements {
                                    if let Some(index) = enum_.borrow().variants[variant_index].find_struct_element(&element.0) {
                                        let new_expr = self.resolve_expr(element.1);
                                        new_elements[index] = new_expr;
                                    }
                                }
                                VariantExpr::Struct(variant_index,new_elements)
                            }
                            else {
                                panic!("enum {} does not contain struct variant {}",enum_.borrow().ident,ident);
                            }
                        },
                    };
                    Expr::Variant(enum_,new_variant)
                }
                else {
                    panic!("unknown enum {}",ident);
                }
            },

            Expr::UnknownMethod(from_expr,ident,elements) => {
                if self.methods.contains_key(&ident) {
                    let mut new_elements: Vec<Expr> = Vec::new();
                    for element in elements {
                        new_elements.push(self.resolve_expr(element));
                    }
                    let new_from_expr = self.resolve_expr(*from_expr);
                    let mut method: Option<Rc<RefCell<Method>>> = None;
                    let methods = &self.methods[&ident];
                    for m in methods.iter() {
                        if new_from_expr.find_type() == m.borrow().from_type {
                            if m.borrow().params.len() == new_elements.len() {
                                let mut found = true;
                                for i in 0..new_elements.len() {
                                    if new_elements[i].find_type() != m.borrow().params[i].type_ {
                                        found = false;
                                        break;
                                    }
                                }
                                if found {
                                    method = Some(Rc::clone(&m));
                                    break;
                                }
                            }
                        }
                    }
                    if let Some(method) = method {
                        Expr::Method(Box::new(new_from_expr),method,new_elements)
                    }
                    else {
                        panic!("unknown method {} for given parameters",ident);
                    }
                }
                else {
                    panic!("unknown method {}",ident);
                }
            },

            Expr::UnknownField(expr,ident) => {
                let new_expr = self.resolve_expr(*expr);
                if let Type::Struct(struct_) = new_expr.find_type() {
                    if let Some(index) = struct_.borrow().find_field(&ident) {
                        Expr::Field(Rc::clone(&struct_),Box::new(new_expr),index)
                    }
                    else {
                        panic!("struct {} has no field {}",struct_.borrow().ident,ident);
                    }
                }
                else {
                    panic!("type of {} not a known struct",new_expr);
                }
            },

            Expr::UnknownTupleIndex(expr,index) => {
                let new_expr = self.resolve_expr(*expr);
                if let Type::Tuple(tuple) = new_expr.find_type() {
                    Expr::TupleIndex(tuple,Box::new(new_expr),index)
                }
                else {
                    panic!("type of {} not a known tuple",new_expr);
                }
            }


            Expr::Param(_) |
            Expr::Local(_) |
            Expr::Const(_) => expr,

            Expr::Call(function,elements) => {
                let mut new_elements: Vec<Expr> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_expr(element));
                }
                Expr::Call(function,new_elements)
            },

            Expr::Tuple(tuple,elements) => {
                let mut new_elements: Vec<Expr> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_expr(element));
                }
                Expr::Tuple(tuple,new_elements)
            },

            Expr::Struct(struct_,elements) => {
                let mut new_elements: Vec<Expr> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_expr(element));
                }
                Expr::Struct(struct_,new_elements)
            },

            Expr::Variant(enum_,variant) => {
                let new_variant = match variant {
                    VariantExpr::Naked(index) => VariantExpr::Naked(index),
                    VariantExpr::Tuple(index,elements) => {
                        let mut new_elements: Vec<Expr> = Vec::new();
                        for element in elements {
                            new_elements.push(self.resolve_expr(element));
                        }
                        VariantExpr::Tuple(index,new_elements)
                    },
                    VariantExpr::Struct(index,elements) => {
                        let mut new_elements: Vec<Expr> = Vec::new();
                        for element in elements {
                            new_elements.push(self.resolve_expr(element));
                        }
                        VariantExpr::Struct(index,new_elements)
                    },
                };
                Expr::Variant(enum_,new_variant)
            },

            Expr::Method(expr,method,arguments) => {
                let new_expr = self.resolve_expr(*expr);
                let mut new_arguments: Vec<Expr> = Vec::new();
                for argument in arguments {
                    new_arguments.push(self.resolve_expr(argument));
                }
                Expr::Method(Box::new(new_expr),method,new_arguments)
            },

            Expr::Field(struct_,expr,index) => Expr::Field(struct_,Box::new(self.resolve_expr(*expr)),index),

            Expr::TupleIndex(tuple,expr,index) => Expr::TupleIndex(tuple,Box::new(self.resolve_expr(*expr)),index),

            Expr::Discriminant(expr) => Expr::Discriminant(Box::new(self.resolve_expr(*expr))),

            Expr::Destructure(expr,variant_index,index) => Expr::Destructure(Box::new(self.resolve_expr(*expr)),variant_index,index),
        }
    }

    pub fn resolve_range(&mut self,range: Range) -> Range {
        match range {
            Range::Only(expr) => Range::Only(Box::new(self.resolve_expr(*expr))),
            Range::From(expr) => Range::From(Box::new(self.resolve_expr(*expr))),
            Range::To(expr) => Range::To(Box::new(self.resolve_expr(*expr))),
            Range::ToIncl(expr) => Range::ToIncl(Box::new(self.resolve_expr(*expr))),
            Range::FromTo(from,to) => Range::FromTo(Box::new(self.resolve_expr(*from)),Box::new(self.resolve_expr(*to))),
            Range::FromToIncl(from,to) => Range::FromToIncl(Box::new(self.resolve_expr(*from)),Box::new(self.resolve_expr(*to))),
            Range::All => Range::All,
        }
    }

    pub fn resolve_stat(&mut self,stat: Stat) -> Stat {
        match stat {
            // expression statement: resolve symbols in the expression
            Stat::Expr(expr) => Stat::Expr(Box::new(self.resolve_expr(*expr))),

            // let statement: resolve pattern, type and expression
            Stat::Let(pat,type_,expr) => {
                println!("Stat::Let({},{},{})",pat,type_,expr);
                let new_pat = self.resolve_pat(*pat);
                println!("new pattern = {}",new_pat);
                let new_type = self.resolve_type(*type_);
                println!("new type = {}",new_type);
                let new_expr = self.resolve_expr(*expr);
                println!("new expr = {}",new_expr);
                Stat::Let(Box::new(new_pat),Box::new(new_type),Box::new(new_expr))
            },

            // local variable let statement: this statement cannot appear
            // before later phases, because local variables are generated from
            // destructuring patterns, which can only occur after symbols in
            // other nodes are resolved
            Stat::Local(symbol,expr) => Stat::Local(symbol,expr),
        }
    }

    pub fn resolve_block(&mut self,block: Block) -> Block {

        let shadowed_locals = self.locals.clone();

        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats {
            new_stats.push(self.resolve_stat(stat));
        }
        let new_expr = if let Some(expr) = block.expr {
            Some(Box::new(self.resolve_expr(*expr)))
        }
        else {
            None
        };

        self.locals = shadowed_locals;

        Block { stats: new_stats,expr: new_expr, }
    }

    pub fn resolve_module(&mut self,module: Module) -> Module {

        for tuple in module.tuples.values() {
            let mut new_elements: Vec<Type> = Vec::new();
            for element in &tuple.borrow().types {
                new_elements.push(self.resolve_type(element.clone()));
            }
            tuple.borrow_mut().types = new_elements;
        }
        for struct_ in module.structs.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.borrow().fields.iter() {
                new_fields.push(Symbol { ident: field.ident.clone(),type_: self.resolve_type(field.type_.clone()), });
            }
            struct_.borrow_mut().fields = new_fields;
        }
        for enum_ in module.enums.values() {
            let mut new_variants: Vec<Variant> = Vec::new();
            for variant in enum_.borrow().variants.iter() {
                new_variants.push(match variant {
                    Variant::Naked(ident) => Variant::Naked(ident.clone()),
                    Variant::Tuple(ident,elements) => {
                        let mut new_elements: Vec<Type> = Vec::new();
                        for element in elements {
                            new_elements.push(self.resolve_type(element.clone()));
                        }
                        Variant::Tuple(ident.clone(),new_elements)
                    },
                    Variant::Struct(ident,elements) => {
                        let mut new_elements: Vec<Symbol> = Vec::new();
                        for element in elements {
                            new_elements.push(Symbol { ident: element.ident.clone(),type_: self.resolve_type(element.type_.clone()), });
                        }
                        Variant::Struct(ident.clone(),new_elements)
                    },
                });
            }
            enum_.borrow_mut().variants = new_variants;
        }
        for alias in module.aliases.values() {
            let new_type = self.resolve_type(alias.borrow().type_.clone());
            alias.borrow_mut().type_ = new_type;
        }
        for const_ in module.consts.values() {
            let new_type = self.resolve_type(const_.borrow().type_.clone());
            let new_expr = self.resolve_expr(const_.borrow().expr.clone());
            const_.borrow_mut().type_ = new_type;
            const_.borrow_mut().expr = new_expr;
        }
        for function in module.functions.values() {
            let mut new_params: Vec<Rc<RefCell<Symbol>>> = Vec::new();
            for param in function.borrow().params.iter() {
                let new_type = self.resolve_type(param.borrow().type_.clone());
                let new_param = Rc::new(RefCell::new(Symbol { ident: param.borrow().ident.clone(),type_: new_type, }));
                self.params.insert(param.borrow().ident.clone(),Rc::clone(&new_param));
                new_params.push(new_param);
            }
            let new_type = self.resolve_type(function.borrow().type_.clone());
            let new_block = self.resolve_block(function.borrow().block.clone());
            self.params.clear();
            function.borrow_mut().params = new_params;
            function.borrow_mut().type_ = new_type;
            function.borrow_mut().block = new_block;
        }
        module
    }
}
