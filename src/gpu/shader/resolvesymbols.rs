use {
    crate::*,
    sr::*,
    std::{
        rc::Rc,
        collections::HashMap,
        cell::RefCell,
    }
};

use ast::*;

pub struct SymbolResolver {
    pub tuples: HashMap<String,Rc<RefCell<Tuple>>>,
    pub structs: HashMap<String,Rc<RefCell<Struct>>>,
    pub enums: HashMap<String,Rc<RefCell<Enum>>>,
    pub aliases: HashMap<String,Rc<RefCell<Alias>>>,
    pub consts: HashMap<String,Rc<RefCell<Const>>>,
    pub functions: HashMap<String,Vec<Rc<RefCell<Function>>>>,
    pub methods: HashMap<String,Vec<Rc<RefCell<Method>>>>,
    pub locals: HashMap<String,Rc<RefCell<Symbol>>>,
    pub params: HashMap<String,Rc<RefCell<Symbol>>>,
}

impl SymbolResolver {

    pub fn new(module: &Module) -> SymbolResolver {
        let stdlib = StandardLib::new();
        let mut tuples = stdlib.tuples.clone();
        for (key,value) in module.tuples.iter() {
            tuples.insert(key.clone(),value.clone());
        }
        let mut structs = stdlib.structs.clone();
        for (key,value) in module.structs.iter() {
            structs.insert(key.clone(),value.clone());
        }
        let mut enums = stdlib.enums.clone();
        for (key,value) in module.enums.iter() {
            enums.insert(key.clone(),value.clone());
        }
        let mut aliases = stdlib.aliases.clone();
        for (key,value) in module.aliases.iter() {
            aliases.insert(key.clone(),value.clone());
        }
        let mut consts = stdlib.consts.clone();
        for (key,value) in module.consts.iter() {
            consts.insert(key.clone(),value.clone());
        }
        let mut functions = stdlib.functions.clone();
        for (key,value) in module.functions.iter() {
            functions.insert(key.clone(),vec![value.clone()]);
        }
        let methods = stdlib.methods.clone();
        SymbolResolver {
            tuples,
            structs,
            enums,
            aliases,
            consts,
            functions,
            methods,
            locals: HashMap::new(),
            params: HashMap::new(),
        }
    }

    pub fn resolve_type(&mut self,type_: Type) -> Type {

        match type_ {

            // inferred, void, literal and base types can't refer to symbols
            Type::Inferred | Type::Void |
            Type::Integer | Type::Float |
            Type::Bool |
            Type::U8 | Type::I8 | Type::U16 | Type::I16 | Type::U32 | Type::I32 | Type::U64 | Type::I64 | Type::USize | Type::ISize |
            Type::F16 | Type::F32 | Type::F64 => type_,

            // anonymous tuple type: resolve symbols on all elements
            Type::AnonTuple(elements) => {
                let mut new_elements: Vec<Type> = Vec::new();
                for element in elements {
                    new_elements.push(self.resolve_type(element));
                }
                Type::AnonTuple(new_elements)
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
                let new_pat = self.resolve_pat(*pat);
                let new_type = self.resolve_type(*type_);
                let new_expr = self.resolve_expr(*expr);
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
