use {
    crate::*,
    sr::*,
    std::{
        collections::HashMap,
        rc::Rc,
        cell::RefCell,
    },
};

use ast::*;

pub struct Detuplifier {
    pub tuple_structs: HashMap<String,Rc<RefCell<Struct>>>,
    pub anon_tuple_structs: HashMap<String,Rc<RefCell<Struct>>>,
}

impl Detuplifier {
    pub fn new() -> Detuplifier {
        Detuplifier { 
            tuple_structs: HashMap::new(),
            anon_tuple_structs: HashMap::new(),
        }
    }

    pub fn detuplify_type(&mut self,type_: &Type) -> Type {
        match type_ {
            Type::Array(type_,expr) => {
                let new_type = self.detuplify_type(type_);
                let new_expr = self.detuplify_expr(expr);
                Type::Array(Box::new(new_type),Box::new(new_expr))
            },
            Type::AnonTuple(types) => {
                let mut found: Option<Rc<RefCell<Struct>>> = None;
                for struct_ in self.anon_tuple_structs.values() {
                    let mut matches: Option<Rc<RefCell<Struct>>> = Some(Rc::clone(&struct_));
                    if struct_.borrow().fields.len() == types.len() {
                        for i in 0..types.len() {
                            if struct_.borrow().fields[i].type_ != types[i] {
                                matches = None;
                                break;
                            }
                        }
                        if let Some(matches) = matches {
                            found = Some(matches);
                            break;
                        }
                    }
                }
                if let Some(struct_) = found {
                    Type::Struct(struct_)
                }
                else {
                    let ident = format!("AnonTuple{}",self.anon_tuple_structs.len());
                    let mut fields: Vec<Symbol> = Vec::new();
                    for i in 0..types.len() {
                        let ident = format!("f{}",i);
                        fields.push(Symbol { ident,type_: types[i].clone(), });
                    }
                    let struct_ = Rc::new(RefCell::new(Struct { ident: ident.clone(),fields, }));
                    self.anon_tuple_structs.insert(ident,Rc::clone(&struct_));
                    Type::Struct(struct_)
                }
            },
            Type::Tuple(tuple) => {
                if self.tuple_structs.contains_key(&tuple.borrow().ident) {
                    Type::Struct(Rc::clone(&self.tuple_structs[&tuple.borrow().ident]))
                }
                else {
                    panic!("ERROR: tuple {} cannot be found in tuple struct list",tuple.borrow().ident);
                }
            },
            _ => type_.clone()
        }
    }

    pub fn detuplify_pat(&mut self,pat: &Pat) -> Pat {
        match pat {
            Pat::AnonTuple(pats) => {
                // anonymous tuples in patterns are only used to guide the
                // identifiers (local variables) for later destructuring, so
                // don't need to be mapped to corresponding structs
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats.iter() {
                    new_pats.push(self.detuplify_pat(pat));
                }
                Pat::AnonTuple(new_pats)
            },
            Pat::Array(pats) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats.iter() {
                    new_pats.push(self.detuplify_pat(pat));
                }
                Pat::Array(new_pats)
            },
            Pat::Range(lo,hi) => {
                let new_lo = self.detuplify_pat(lo);
                let new_hi = self.detuplify_pat(hi);
                Pat::Range(Box::new(new_lo),Box::new(new_hi))
            },
            Pat::Tuple(tuple,pats) => {
                if self.tuple_structs.contains_key(&tuple.borrow().ident) {
                    let mut new_fieldpats: Vec<FieldPat> = Vec::new();
                    for i in 0..pats.len() {
                        match pats[i] {
                            Pat::Wildcard => new_fieldpats.push(FieldPat::Wildcard),
                            Pat::Rest => new_fieldpats.push(FieldPat::Rest),
                            _ => new_fieldpats.push(FieldPat::IndexPat(i,self.detuplify_pat(&pats[i]))),
                        }
                    }
                    Pat::Struct(Rc::clone(&self.tuple_structs[&tuple.borrow().ident]),new_fieldpats)
                }
                else {
                    panic!("ERROR: tuple {} cannot be found in tuple struct list",tuple.borrow().ident);
                }
            },
            Pat::Struct(struct_,fieldpats) => {
                let mut new_fieldpats: Vec<FieldPat> = Vec::new();
                for fieldpat in fieldpats.iter() {
                    new_fieldpats.push(if let FieldPat::IndexPat(index,pat) = fieldpat {
                        FieldPat::IndexPat(*index,self.detuplify_pat(pat))
                    }
                    else {
                        fieldpat.clone()
                    });
                }
                Pat::Struct(Rc::clone(&struct_),new_fieldpats)
            },
            Pat::Variant(enum_,variant) => {
                let new_variant = match variant {
                    VariantPat::Naked(index) => VariantPat::Naked(*index),
                    VariantPat::Tuple(index,pats) => {
                        let mut new_pats: Vec<Pat> = Vec::new();
                        for pat in pats.iter() {
                            new_pats.push(self.detuplify_pat(pat));
                        }
                        VariantPat::Tuple(*index,new_pats)
                    },
                    VariantPat::Struct(index,fieldpats) => {
                        let mut new_fieldpats: Vec<FieldPat> = Vec::new();
                        for fieldpat in fieldpats.iter() {
                            new_fieldpats.push(if let FieldPat::IndexPat(index,pat) = fieldpat {
                                FieldPat::IndexPat(*index,self.detuplify_pat(pat))
                            }
                            else {
                                fieldpat.clone()
                            });
                        }
                        VariantPat::Struct(*index,new_fieldpats)
                    }
                };
                Pat::Variant(Rc::clone(&enum_),new_variant)
            },
            _ => pat.clone(),
        }
    }

    pub fn detuplify_expr(&mut self,expr: &Expr) -> Expr {
        match expr {
            Expr::Boolean(value) => Expr::Boolean(*value),
            Expr::Integer(value) => Expr::Integer(*value),
            Expr::Float(value) => Expr::Float(*value),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                Expr::Array(new_exprs)
            },
            Expr::Cloned(expr,count) => {
                let new_expr = self.detuplify_expr(expr);
                let new_count = self.detuplify_expr(count);
                Expr::Cloned(Box::new(new_expr),Box::new(new_count))
            },
            Expr::Index(expr,index) => {
                let new_expr = self.detuplify_expr(expr);
                let new_index = self.detuplify_expr(index);
                Expr::Index(Box::new(new_expr),Box::new(new_index))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.detuplify_expr(expr);
                let new_type = self.detuplify_type(type_);
                Expr::Cast(Box::new(new_expr),Box::new(new_type))
            },
            Expr::AnonTuple(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                let mut found: Option<Rc<RefCell<Struct>>> = None;
                for struct_ in self.anon_tuple_structs.values() {
                    let mut matches: Option<Rc<RefCell<Struct>>> = Some(Rc::clone(&struct_));
                    if struct_.borrow().fields.len() == exprs.len() {
                        for i in 0..new_exprs.len() {
                            let type_ = new_exprs[i].find_type();
                            if let None = tightest(&type_,&struct_.borrow().fields[i].type_) {
                                matches = None;
                                break;
                            }
                        }
                        if let Some(matches) = matches {
                            found = Some(matches);
                            break;
                        }
                    }
                }
                if let Some(struct_) = found {
                    Expr::Struct(struct_,new_exprs)
                }
                else {
                    let ident = format!("AnonTuple{}",self.anon_tuple_structs.len());
                    let mut fields: Vec<Symbol> = Vec::new();
                    for i in 0..new_exprs.len() {
                        let ident = format!("f{}",i);
                        let type_ = new_exprs[i].find_type();
                        fields.push(Symbol { ident,type_, });
                    }
                    let struct_ = Rc::new(RefCell::new(Struct { ident: ident.clone(),fields, }));
                    self.anon_tuple_structs.insert(ident,Rc::clone(&struct_));
                    Expr::Struct(struct_,new_exprs)
                }
            },
            Expr::Unary(op,expr) => {
                let new_expr = self.detuplify_expr(expr);
                Expr::Unary(op.clone(),Box::new(new_expr))
            },
            Expr::Binary(expr,op,expr2) => {
                let new_expr = self.detuplify_expr(expr);
                let new_expr2 = self.detuplify_expr(expr2);
                Expr::Binary(Box::new(new_expr),op.clone(),Box::new(new_expr2))
            },
            Expr::Continue => Expr::Continue,
            Expr::Break(expr) => {
                if let Some(expr) = expr {
                    Expr::Break(Some(Box::new(self.detuplify_expr(expr))))
                }
                else {
                    Expr::Break(None)
                }
            },
            Expr::Return(expr) => {
                if let Some(expr) = expr {
                    Expr::Return(Some(Box::new(self.detuplify_expr(expr))))
                }
                else {
                    Expr::Return(None)
                }
            },
            Expr::Block(block) => {
                let new_block = self.detuplify_block(block);
                Expr::Block(new_block)
            },
            Expr::If(expr,block,else_expr) => {
                let new_expr = self.detuplify_expr(expr);
                let new_block = self.detuplify_block(block);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.detuplify_expr(else_expr)))
                }
                else {
                    None
                };
                Expr::If(Box::new(new_expr),new_block,new_else_expr)
            },
            Expr::While(expr,block) => {
                let new_expr = self.detuplify_expr(expr);
                let new_block = self.detuplify_block(block);
                Expr::While(Box::new(new_expr),new_block)
            },
            Expr::Loop(block) => {
                let new_block = self.detuplify_block(block);
                Expr::Loop(new_block)
            },
            Expr::IfLet(pats,expr,block,else_expr) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats.iter() {
                    new_pats.push(self.detuplify_pat(pat));
                }
                let new_expr = self.detuplify_expr(expr);
                let new_block = self.detuplify_block(block);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.detuplify_expr(else_expr)))
                }
                else {
                    None
                };
                Expr::IfLet(new_pats,Box::new(new_expr),new_block,new_else_expr)
            },
            Expr::For(pats,range,block) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats.iter() {
                    new_pats.push(self.detuplify_pat(pat));
                }
                let new_range = self.detuplify_range(range);
                let new_block = self.detuplify_block(block);
                Expr::For(new_pats,new_range,new_block)
            },
            Expr::WhileLet(pats,expr,block) => {
                let mut new_pats: Vec<Pat> = Vec::new();
                for pat in pats.iter() {
                    new_pats.push(self.detuplify_pat(pat));
                }
                let new_expr = self.detuplify_expr(expr);
                let new_block = self.detuplify_block(block);
                Expr::WhileLet(new_pats,Box::new(new_expr),new_block)
            },
            Expr::Match(expr,arms) => {
                let new_expr = self.detuplify_expr(expr);
                let mut new_arms: Vec<(Vec<Pat>,Option<Box<Expr>>,Box<Expr>)> = Vec::new();
                for (pats,if_expr,expr) in arms.iter() {
                    let mut new_pats: Vec<Pat> = Vec::new();
                    for pat in pats.iter() {
                        new_pats.push(self.detuplify_pat(pat));
                    }
                    let if_expr = if let Some(if_expr) = if_expr {
                        Some(Box::new(self.detuplify_expr(if_expr)))
                    }
                    else {
                        None
                    };
                    let new_expr = self.detuplify_expr(expr);
                    new_arms.push((new_pats,if_expr,Box::new(new_expr)));
                }
                Expr::Match(Box::new(new_expr),new_arms)
            },
            Expr::UnknownIdent(_) => panic!("ERROR: Expr::UnknownIdent cannot occur in detuplify stage"),
            Expr::UnknownTupleOrCall(_,_) => panic!("ERROR: Expr::UnknownTupleOrCall cannot occur in detuplify stage"),
            Expr::UnknownStruct(_,_) => panic!("ERROR: Expr::UnknownStruct cannot occur in detuplify stage"),
            Expr::UnknownVariant(_,_) => panic!("ERROR: Expr::UnknownVariant cannot occur in detuplify stage"),
            Expr::UnknownMethod(_,_,_) => panic!("ERROR: Expr::UnknownMethod cannot occur in detuplify stage"),
            Expr::UnknownField(_,_) => panic!("ERROR: Expr::UnknownField cannot occur in detuplify stage"),
            Expr::UnknownTupleIndex(_,_) => panic!("ERROR: Expr::UnknownTupleIndex cannot occur in detuplify stage"),        
            Expr::Param(param) => Expr::Param(Rc::clone(&param)),
            Expr::Local(local) => Expr::Local(Rc::clone(&local)),
            Expr::Const(const_) => Expr::Const(Rc::clone(&const_)),
            Expr::Tuple(tuple,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                if self.tuple_structs.contains_key(&tuple.borrow().ident) {
                    Expr::Struct(Rc::clone(&self.tuple_structs[&tuple.borrow().ident]),new_exprs)
                }
                else {
                    panic!("ERROR: tuple {} cannot be found in tuple struct list",tuple.borrow().ident);
                }
            },
            Expr::Call(function,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                Expr::Call(Rc::clone(&function),new_exprs)
            },
            Expr::Struct(struct_,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                Expr::Struct(Rc::clone(&struct_),new_exprs)
            },
            Expr::Variant(enum_,variant) => {
                let new_variant = match variant {
                    VariantExpr::Naked(index) => VariantExpr::Naked(*index),
                    VariantExpr::Tuple(index,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(self.detuplify_expr(expr));
                        }
                        VariantExpr::Tuple(*index,new_exprs)
                    },
                    VariantExpr::Struct(index,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(self.detuplify_expr(expr));
                        }
                        VariantExpr::Struct(*index,new_exprs)
                    },
                };
                Expr::Variant(Rc::clone(&enum_),new_variant)
            },
            Expr::Method(expr,method,exprs) => {
                let new_expr = self.detuplify_expr(expr);
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.detuplify_expr(expr));
                }
                Expr::Method(Box::new(new_expr),Rc::clone(&method),new_exprs)
            },
            Expr::Field(struct_,expr,index) => {
                let new_expr = self.detuplify_expr(expr);
                Expr::Field(Rc::clone(&struct_),Box::new(new_expr),*index)
            },
            Expr::TupleIndex(tuple,expr,index) => {
                let new_expr = self.detuplify_expr(expr);
                if self.tuple_structs.contains_key(&tuple.borrow().ident) {
                    Expr::Field(Rc::clone(&self.tuple_structs[&tuple.borrow().ident]),Box::new(new_expr),*index)
                }
                else {
                    panic!("ERROR: tuple {} cannot be found in tuple struct list",tuple.borrow().ident);
                }
            }
        }
    }

    pub fn detuplify_range(&mut self,range: &Range) -> Range {
        match range {
            Range::Only(expr) => {
                let new_expr = self.detuplify_expr(expr);
                Range::Only(Box::new(new_expr))
            },
            Range::FromTo(expr,expr2) => {
                let new_expr = self.detuplify_expr(expr);
                let new_expr2 = self.detuplify_expr(expr2);
                Range::FromTo(Box::new(new_expr),Box::new(new_expr2))
            },
            Range::FromToIncl(expr,expr2) => {
                let new_expr = self.detuplify_expr(expr);
                let new_expr2 = self.detuplify_expr(expr2);
                Range::FromToIncl(Box::new(new_expr),Box::new(new_expr2))
            },
            Range::From(expr) => {
                let new_expr = self.detuplify_expr(expr);
                Range::From(Box::new(new_expr))
            },
            Range::To(expr) => {
                let new_expr = self.detuplify_expr(expr);
                Range::To(Box::new(new_expr))
            },
            Range::ToIncl(expr) => {
                let new_expr = self.detuplify_expr(expr);
                Range::Only(Box::new(new_expr))
            },
            Range::All => Range::All,
        }
    }

    pub fn detuplify_stat(&mut self,stat: &Stat) -> Stat {
        match stat {
            Stat::Let(pat,type_,expr) => {
                let new_pat = self.detuplify_pat(pat);
                let new_type = self.detuplify_type(type_);
                let new_expr = self.detuplify_expr(expr);
                Stat::Let(Box::new(new_pat),Box::new(new_type),Box::new(new_expr))
            },
            Stat::Expr(expr) => {
                let new_expr = self.detuplify_expr(expr);
                Stat::Expr(Box::new(new_expr))
            },
            Stat::Local(local,expr) => {
                let new_expr = self.detuplify_expr(expr);
                Stat::Local(Rc::clone(&local),Box::new(new_expr))
            },
        }
    }

    pub fn detuplify_block(&mut self,block: &Block) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            new_stats.push(self.detuplify_stat(stat));
        }
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.detuplify_expr(&expr)))
        }
        else {
            None
        };
        Block { stats: new_stats,expr: new_expr, }
    }

    pub fn detuplify_module(&mut self,mut module: Module) -> Module {

        // convert tuples to structs
        for tuple in module.tuples.values() {
            let mut fields: Vec<Symbol> = Vec::new();
            for i in 0..tuple.borrow().types.len() {
                let ident = format!("f{}",i);
                let type_ = self.detuplify_type(&tuple.borrow().types[i]);
                fields.push(Symbol { ident,type_, });
            }
            self.tuple_structs.insert(tuple.borrow().ident.clone(),Rc::new(RefCell::new(Struct { ident: tuple.borrow().ident.clone(),fields, })));
        }

        for struct_ in module.structs.values() {
            let mut new_fields: Vec<Symbol> = Vec::new();
            for field in struct_.borrow().fields.iter() {
                let type_ = self.detuplify_type(&field.type_);
                new_fields.push(Symbol { ident: field.ident.clone(),type_, });
            }
            struct_.borrow_mut().fields = new_fields;
        }

        for enum_ in module.enums.values() {
            let mut new_variants: Vec<Variant> = Vec::new();
            for variant in enum_.borrow().variants.iter() {
                new_variants.push(match variant {
                    Variant::Naked(ident) => Variant::Naked(ident.clone()),
                    Variant::Tuple(ident,types) => {
                        let mut new_types: Vec<Type> = Vec::new();
                        for type_ in types.iter() {
                            new_types.push(self.detuplify_type(type_));
                        }
                        Variant::Tuple(ident.clone(),new_types)
                    },
                    Variant::Struct(ident,fields) => {
                        let mut new_fields: Vec<Symbol> = Vec::new();
                        for field in fields.iter() {
                            new_fields.push(Symbol { ident: field.ident.clone(),type_: self.detuplify_type(&field.type_), });
                        }
                        Variant::Struct(ident.clone(),new_fields)
                    },
                });
            }
            enum_.borrow_mut().variants = new_variants;
        }

        for alias in module.aliases.values() {
            let new_type = self.detuplify_type(&alias.borrow().type_);
            alias.borrow_mut().type_ = new_type;
        }

        for const_ in module.consts.values() {
            let new_type = self.detuplify_type(&const_.borrow().type_);
            let new_expr = self.detuplify_expr(&const_.borrow().expr);
            const_.borrow_mut().type_ = new_type;
            const_.borrow_mut().expr = new_expr;
        }

        for function in module.functions.values() {
            for param in function.borrow().params.iter() {
                let new_type = self.detuplify_type(&param.borrow().type_);
                param.borrow_mut().type_ = new_type;
            }
            let new_type = self.detuplify_type(&function.borrow().type_);
            function.borrow_mut().type_ = new_type;
            let new_block = self.detuplify_block(&function.borrow().block);
            function.borrow_mut().block = new_block;
        }

        for struct_ in self.tuple_structs.values() {
            module.structs.insert(struct_.borrow().ident.clone(),Rc::clone(&struct_));
        }

        for struct_ in self.anon_tuple_structs.values() {
            module.structs.insert(struct_.borrow().ident.clone(),Rc::clone(&struct_));
        }

        module
    }
}
