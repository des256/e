use {
    crate::*,
    sr::*,
    std::{
        rc::Rc,
        cell::RefCell,
    },
};

use ast::*;

pub struct Destructurer { }

impl Destructurer {
    pub fn new() -> Destructurer {
        Destructurer { }
    }

    fn build_matcher(&mut self,pat: &Pat,scrut: &Expr) -> Option<Expr> {
        match pat {
            Pat::Wildcard |
            Pat::Rest |
            Pat::UnknownIdent(_) => None,
            Pat::Boolean(value) => Some(if *value { scrut.clone() } else { Expr::Unary(UnaryOp::Not,Box::new(scrut.clone())) }),
            Pat::Integer(value) => Some(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Integer(*value)))),
            Pat::Float(value) => Some(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Eq,Box::new(Expr::Float(*value)))),
            Pat::AnonTuple(pats) => {
                let struct_ = if let Type::Struct(struct_) = scrut.find_type() {
                    struct_
                }
                else {
                    panic!("pattern cannot be matched for {}",scrut);
                };
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.build_matcher(&pats[i],&Expr::Field(Rc::clone(&struct_),Box::new(scrut.clone()),i)) {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                accum
            },
            Pat::Array(pats) => {
                let mut accum: Option<Expr> = None;
                for i in 0..pats.len() {
                    if let Some(expr) = self.build_matcher(&pats[i],&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64)))) {
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                }
                accum
            },
            Pat::Range(lo,hi) =>  {
                match **lo {
                    Pat::Integer(lo) => {
                        if let Pat::Integer(hi) = **hi {
                            Some(Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Integer(lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Integer(hi)))),
                            ))
                        }
                        else {
                            panic!("pattern range can only be integer or float");
                        }
                    },
                    Pat::Float(lo) => {
                        if let Pat::Float(hi) = **hi {
                            Some(Expr::Binary(
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::GreaterEq,Box::new(Expr::Float(lo)))),
                                BinaryOp::LogAnd,
                                Box::new(Expr::Binary(Box::new(scrut.clone()),BinaryOp::Less,Box::new(Expr::Float(hi)))),
                            ))
                        }
                        else {
                            panic!("pattern range can only be integer or float");
                        }
                    },
                    _ => panic!("pattern range can only be integer or float"),
                }
            },
            Pat::UnknownTuple(..) => panic!("ERROR: Pat::UnknownTuple cannot exist in destructure phase"),
            Pat::UnknownStruct(..) => panic!("ERROR: Pat::UnknownStruct cannot exist in destructure phase"),
            Pat::UnknownVariant(..) => panic!("ERROR: Pat::UnknownVariant cannot exist in destructure phase"),
            Pat::Tuple(..) => panic!("ERROR: Pat::Tuple cannot exist in destructure phase"),
            Pat::Struct(struct_,fields) => {
                let mut accum: Option<Expr> = None;
                for field in fields.iter() {
                    if let FieldPat::IndexPat(index,pat) = field {
                        if let Some(expr) = self.build_matcher(pat,&Expr::Field(Rc::clone(&struct_),Box::new(scrut.clone()),*index)) {
                            if let Some(accum_inner) = accum {
                                accum = Some(Expr::Binary(Box::new(accum_inner.clone()),BinaryOp::LogAnd,Box::new(expr)));
                            }
                            else {
                                accum = Some(expr);
                            }
                        }
                    }
                }
                accum
            },
            Pat::Variant(_,variant) => {
                match variant {
                    VariantPat::Naked(variant_index) => {
                        Some(Expr::Binary(
                            Box::new(Expr::Discriminant(Box::new(scrut.clone()))),
                            BinaryOp::Eq,
                            Box::new(Expr::Integer(*variant_index as i64)),
                        ))
                    },
                    VariantPat::Tuple(variant_index,pats) => {
                        let mut accum = Expr::Binary(
                            Box::new(Expr::Discriminant(Box::new(scrut.clone()))),
                            BinaryOp::Eq,
                            Box::new(Expr::Integer(*variant_index as i64)),
                        );
                        for i in 0..pats.len() {
                            if let Some(expr) = self.build_matcher(&pats[i],&Expr::Destructure(Box::new(scrut.clone()),*variant_index,i)) {
                                accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                            }
                        }
                        Some(accum)
                    },
                    VariantPat::Struct(variant_index,fields) => {
                        let mut accum = Expr::Binary(
                            Box::new(Expr::Discriminant(Box::new(scrut.clone()))),
                            BinaryOp::Eq,
                            Box::new(Expr::Integer(*variant_index as i64)),
                        );
                        for i in 0..fields.len() {
                            if let FieldPat::IndexPat(index,pat) = &fields[i] {
                                if let Some(expr) = self.build_matcher(&pat,&Expr::Destructure(Box::new(scrut.clone()),*variant_index,*index)) {
                                    accum = Expr::Binary(Box::new(accum),BinaryOp::LogAnd,Box::new(expr));
                                }
                            }
                        }
                        Some(accum)
                    },
                }
            },
        }
    }

    fn build_matchers(&mut self,pats: &Vec<Pat>,local: &Rc<RefCell<Symbol>>) -> Option<Expr> {
        let mut accum: Option<Expr> = None;
        for pat in pats.iter() {
            if let Some(expr) = self.build_matcher(pat,&Expr::Local(Rc::clone(&local))) {
                if let Some(accum_inner) = accum {
                    accum = Some(Expr::Binary(Box::new(accum_inner),BinaryOp::LogOr,Box::new(expr)));
                }
                else {
                    accum = Some(expr);
                }
            }
        }
        accum
    }

    pub fn destructure_pat(&mut self,pat: &Pat,scrut: &Expr) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        match pat {
            Pat::Wildcard |
            Pat::Rest |
            Pat::Boolean(_) |
            Pat::Integer(_) |
            Pat::Float(_) |
            Pat::Range(_,_) => { },
            Pat::UnknownIdent(ident) => {
                let expr = scrut.clone();
                let type_ = expr.find_type();
                let local = Rc::new(RefCell::new(Symbol { ident: ident.clone(),type_, }));
                stats.push(Stat::Local(local,Box::new(expr)));
            },
            Pat::Struct(struct_,fields) => {
                for field in fields.iter() {
                    match field {
                        FieldPat::Wildcard |
                        FieldPat::Rest => { },
                        FieldPat::Index(index) => {
                            let expr = Expr::Field(Rc::clone(&struct_),Box::new(scrut.clone()),*index);
                            let type_ = struct_.borrow().fields[*index].type_.clone();
                            let local = Rc::new(RefCell::new(Symbol { ident: struct_.borrow().fields[*index].ident.clone(),type_, }));
                            stats.push(Stat::Local(local,Box::new(expr)));
                        },
                        FieldPat::IndexPat(index,pat) => stats.append(&mut self.destructure_pat(pat,&Expr::Field(Rc::clone(&struct_),Box::new(scrut.clone()),*index))),
                    }
                }
            },
            Pat::Array(pats) => {
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(pat,&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64)))));
                }
            },
            Pat::Variant(_,variant) => {
                match variant {
                    VariantPat::Naked(_) => { },
                    VariantPat::Tuple(variant_index,pats) => {
                        for i in 0..pats.len() {
                            stats.append(&mut self.destructure_pat(&pats[i],&Expr::Destructure(Box::new(scrut.clone()),*variant_index,i)));
                        }
                    },
                    VariantPat::Struct(variant_index,fields) => {
                        for field in fields.iter() {
                            if let FieldPat::IndexPat(index,pat) = field {
                                stats.append(&mut self.destructure_pat(pat,&Expr::Destructure(Box::new(scrut.clone()),*variant_index,*index)));
                            }
                        }
                    },
                }
            },
            Pat::AnonTuple(pats) => {
                let struct_ = if let Type::Struct(struct_) = scrut.find_type() {
                    struct_
                }
                else {
                    panic!("pattern cannot be matched for {}",scrut);
                };
                for i in 0..pats.len() {
                    stats.append(&mut self.destructure_pat(pat,&Expr::Field(Rc::clone(&struct_),Box::new(scrut.clone()),i)));
                }
            },
            Pat::UnknownTuple(..) => panic!("ERROR: Pat::UnknownTuple cannot occur in destructure phase"),
            Pat::UnknownStruct(..) => panic!("ERROR: Pat::UnknownStruct cannot occur in destructure phase"),
            Pat::UnknownVariant(..) => panic!("ERROR: Pat::UnknownVariant cannot occur in destructure phase"),
            Pat::Tuple(..) => panic!("ERROR: Pat::Tuple cannot occur in destructure phase"),
        }
        stats
    }

    pub fn destructure_pats(&mut self,pats: &Vec<Pat>,local: &Rc<RefCell<Symbol>>) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        for pat in pats.iter() {
            stats.append(&mut self.destructure_pat(pat,&Expr::Local(Rc::clone(&local))));
        }
        stats
    }

    pub fn destructure_expr(&mut self,expr: &Expr) -> Expr {
        match expr {
            Expr::Boolean(value) => Expr::Boolean(*value),
            Expr::Integer(value) => Expr::Integer(*value),
            Expr::Float(value) => Expr::Float(*value),
            Expr::Array(exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs.iter() {
                    new_exprs.push(self.destructure_expr(expr));
                }
                Expr::Array(new_exprs)
            },
            Expr::Cloned(expr,count) => {
                let new_expr = self.destructure_expr(expr);
                let new_count = self.destructure_expr(count);
                Expr::Cloned(Box::new(new_expr),Box::new(new_count))
            },
            Expr::Index(expr,index) => {
                let new_expr = self.destructure_expr(expr);
                let new_index = self.destructure_expr(index);
                Expr::Index(Box::new(new_expr),Box::new(new_index))
            },
            Expr::Cast(expr,type_) => {
                let new_expr = self.destructure_expr(expr);
                Expr::Cast(Box::new(new_expr),type_.clone())
            },
            Expr::AnonTuple(_) => panic!("ERROR: Expr::AnonTuple cannot exist in destructure phase"),
            Expr::Unary(op,expr) => {
                let new_expr = self.destructure_expr(expr);
                Expr::Unary(op.clone(),Box::new(new_expr))
            }
            Expr::Binary(expr,op,expr2) => {
                let new_expr = self.destructure_expr(expr);
                let new_expr2 = self.destructure_expr(expr2);
                Expr::Binary(Box::new(new_expr),op.clone(),Box::new(new_expr2))
            },
            Expr::Continue => Expr::Continue,
            Expr::Break(expr) => if let Some(expr) = expr {
                let new_expr = self.destructure_expr(expr);
                Expr::Break(Some(Box::new(new_expr)))
            }
            else {
                Expr::Break(None)
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                let new_expr = self.destructure_expr(expr);
                Expr::Return(Some(Box::new(new_expr)))
            }
            else {
                Expr::Return(None)
            },
            Expr::Block(block) => {
                let new_block = self.destructure_block(block);
                Expr::Block(new_block)
            },
            Expr::If(expr,block,else_expr) => {
                let new_expr = self.destructure_expr(expr);
                let new_block = self.destructure_block(block);
                let else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.destructure_expr(else_expr)))
                }
                else {
                    None
                };
                Expr::If(Box::new(new_expr),new_block,else_expr)
            },
            Expr::While(expr,block) => {
                let new_expr = self.destructure_expr(expr);
                let new_block = self.destructure_block(block);
                Expr::While(Box::new(new_expr),new_block)
            },
            Expr::Loop(block) => {
                let new_block = self.destructure_block(block);
                Expr::Loop(new_block)
            },
            Expr::IfLet(pats,expr,block,else_expr) => {
                let new_expr = self.destructure_expr(expr);
                let new_block = self.destructure_block(block);
                let new_else_expr = if let Some(else_expr) = else_expr {
                    Some(Box::new(self.destructure_expr(else_expr)))
                }
                else {
                    None
                };
                let mut if_block = Block { stats: Vec::new(),expr: None, };
                let type_ = new_expr.find_type();
                let local = Rc::new(RefCell::new(Symbol { ident: "scrut".to_string(),type_, }));
                if_block.stats.push(Stat::Local(Rc::clone(&local),Box::new(new_expr)));
                let condition = self.build_matchers(pats,&local).expect("unable to create condition from if let patterns");
                let mut then_block = Block { stats: Vec::new(),expr: new_block.expr.clone(), };
                then_block.stats.append(&mut self.destructure_pats(pats,&local));
                for stat in new_block.stats.iter() {
                    then_block.stats.push(stat.clone());
                }
                if_block.expr = Some(Box::new(Expr::If(Box::new(condition),then_block,new_else_expr)));
                Expr::Block(if_block)
            },
            Expr::For(pats,range,block) => {
                let mut new_block = self.destructure_block(block);
                if pats.len() > 1 {
                    panic!("only one pattern supported for for-loops");
                }
                if let Pat::UnknownIdent(ident) = &pats[0] {
                    match range {
                        Range::Only(expr) => {
                            let new_expr = self.destructure_expr(expr);
                            let mut for_block = Block { stats: Vec::new(),expr: None, };
                            let local = Rc::new(RefCell::new(Symbol { ident: ident.clone(),type_: Type::I64, }));
                            for_block.stats.push(Stat::Local(Rc::clone(&local),Box::new(new_expr)));
                            for_block.stats.append(&mut new_block.stats);
                            Expr::Block(for_block)
                        },
                        Range::FromTo(expr,expr2) => {
                            let new_expr = self.destructure_expr(expr);
                            let new_expr2 = self.destructure_expr(expr2);
                            let mut while_block = Block { stats: Vec::new(),expr: None, };
                            let local = Rc::new(RefCell::new(Symbol { ident: ident.clone(),type_: Type::I64, }));
                            while_block.stats.push(
                                Stat::Local(
                                    Rc::clone(&local),
                                    Box::new(new_expr)
                                )
                            );
                            new_block.stats.push(
                                Stat::Expr(
                                    Box::new(Expr::Binary(
                                        Box::new(Expr::Local(Rc::clone(&local))),
                                        BinaryOp::AddAssign,
                                        Box::new(Expr::Integer(1))
                                    ))
                                )
                            );
                            let while_expr = Expr::While(
                                Box::new(Expr::Binary(
                                    Box::new(Expr::Local(Rc::clone(&local))),
                                    BinaryOp::Less,
                                    Box::new(new_expr2)
                                )),
                                new_block
                            );
                            while_block.stats.push(Stat::Expr(Box::new(while_expr)));
                            Expr::Block(while_block)
                        },
                        Range::FromToIncl(expr,expr2) => {
                            let new_expr = self.destructure_expr(expr);
                            let new_expr2 = self.destructure_expr(expr2);
                            let mut while_block = Block { stats: Vec::new(),expr: None, };
                            let local = Rc::new(RefCell::new(Symbol { ident: ident.clone(),type_: Type::I64, }));
                            while_block.stats.push(
                                Stat::Local(
                                    Rc::clone(&local),
                                    Box::new(new_expr)
                                )
                            );
                            new_block.stats.push(
                                Stat::Expr(
                                    Box::new(Expr::Binary(
                                        Box::new(Expr::Local(Rc::clone(&local))),
                                        BinaryOp::AddAssign,
                                        Box::new(Expr::Integer(1))
                                    ))
                                )
                            );
                            let while_expr = Expr::While(
                                Box::new(Expr::Binary(
                                    Box::new(Expr::Local(Rc::clone(&local))),
                                    BinaryOp::LessEq,
                                    Box::new(new_expr2)
                                )),
                                new_block
                            );
                            while_block.stats.push(Stat::Expr(Box::new(while_expr)));
                            Expr::Block(while_block)
                        },
                        _ => panic!("invalid for-loop range (should have start and end)"),
                    }
                }
                else {
                    panic!("only one count variable supported for for-loops");
                }
            },
            Expr::WhileLet(pats,expr,block) => {
                let new_expr = self.destructure_expr(expr);
                let new_block = self.destructure_block(block);
                let mut while_block = Block { stats: Vec::new(),expr: None, };
                let type_ = new_expr.find_type();
                let local = Rc::new(RefCell::new(Symbol { ident: "scrut".to_string(),type_, }));
                while_block.stats.push(Stat::Local(Rc::clone(&local),Box::new(new_expr)));
                let condition = self.build_matchers(pats,&local).expect("unable to create condition from if let patterns");
                let mut then_block = Block { stats: Vec::new(),expr: new_block.expr.clone(), };
                then_block.stats.append(&mut self.destructure_pats(pats,&local));
                for stat in new_block.stats.iter() {
                    then_block.stats.push(stat.clone());
                }
                while_block.expr = Some(Box::new(Expr::While(Box::new(condition),then_block)));
                Expr::Block(while_block)
            },
            Expr::Match(scrut,arms) => {
                let new_scrut = self.destructure_expr(scrut);
                let mut match_block = Block { stats: Vec::new(),expr: None, };
                let local = Rc::new(RefCell::new(Symbol { ident: "scrut".to_string(),type_: new_scrut.find_type(), }));
                match_block.stats.push(Stat::Local(Rc::clone(&local),Box::new(new_scrut)));
                let mut exprs: Vec<Expr> = Vec::new();
                let mut else_expr: Option<Box<Expr>> = None;
                for (pats,_,expr) in arms {
                    // TODO: if_expr does what exactly?
                    let new_expr = self.destructure_expr(expr);
                    if let Some(condition) = self.build_matchers(pats,&Rc::clone(&local)) {
                        let arm_block = Block {
                            stats: self.destructure_pats(pats,&Rc::clone(&local)),
                            expr: Some(Box::new(new_expr)),
                        };
                        exprs.push(Expr::If(Box::new(condition),arm_block,None));
                    }
                    else if let None = else_expr {
                        let arm_block = Block {
                            stats: Vec::new(),
                            expr: Some(Box::new(Expr::Local(Rc::clone(&local)))),
                        };
                        else_expr = Some(Box::new(Expr::Block(arm_block)));
                    }
                    else {
                        panic!("match-expression can only have one catch-all arm");
                    };
                }
                let mut result_expr: Option<Box<Expr>> = else_expr;
                for i in 0..exprs.len() {
                    if let Expr::If(condition,block,_) = &exprs[exprs.len() - i - 1] {
                        result_expr = Some(Box::new(Expr::If(condition.clone(),block.clone(),result_expr)));
                    }
                }
                *result_expr.unwrap()
            },
            Expr::UnknownIdent(..) => panic!("ERROR: Expr::UnknownIdent cannot occur in destructure phase"),
            Expr::UnknownTupleOrCall(..) => panic!("ERROR: Expr::UnknownIdent cannot occur in destructure phase"),
            Expr::UnknownStruct(..) => panic!("ERROR: Expr::UnknownStruct cannot occur in destructure phase"),
            Expr::UnknownVariant(..) => panic!("ERROR: Expr::UnknownVariant cannot occur in destructure phase"),
            Expr::UnknownMethod(..) => panic!("ERROR: Expr::UnknownMethod cannot occur in destructure phase"),
            Expr::UnknownField(..) => panic!("ERROR: Expr::UnknownField cannot occur in destructure phase"),
            Expr::UnknownTupleIndex(..) => panic!("ERROR: Expr::UnknownTupleIndex cannot occur in destructure phase"),
            Expr::Param(param) => Expr::Param(Rc::clone(&param)),
            Expr::Local(local) => Expr::Local(Rc::clone(&local)),
            Expr::Const(const_) => Expr::Const(Rc::clone(&const_)),
            Expr::Tuple(..) => panic!("Expr::Tuple cannot occur in destructure phase"),
            Expr::Call(function,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..exprs.len() {
                    new_exprs.push(self.destructure_expr(&exprs[i]))
                }
                Expr::Call(Rc::clone(&function),new_exprs)
            },
            Expr::Struct(struct_,exprs) => {
                let mut new_exprs: Vec<Expr> = Vec::new();
                for i in 0..exprs.len() {
                    new_exprs.push(self.destructure_expr(&exprs[i]));
                }
                Expr::Struct(Rc::clone(&struct_),new_exprs)
            },
            Expr::Variant(enum_,variant) => {
                match variant {
                    VariantExpr::Naked(index) => Expr::Variant(Rc::clone(&enum_),VariantExpr::Naked(*index)),
                    VariantExpr::Tuple(index,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(self.destructure_expr(expr));
                        }
                        Expr::Variant(Rc::clone(&enum_),VariantExpr::Tuple(*index,new_exprs))
                    },
                    VariantExpr::Struct(index,exprs) => {
                        let mut new_exprs: Vec<Expr> = Vec::new();
                        for expr in exprs.iter() {
                            new_exprs.push(self.destructure_expr(expr));
                        }
                        Expr::Variant(Rc::clone(&enum_),VariantExpr::Struct(*index,new_exprs))
                    },
                }
            },
            Expr::Method(expr,method,exprs) => {
                let new_expr = self.destructure_expr(expr);
                let mut new_exprs: Vec<Expr> = Vec::new();
                for expr in exprs {
                    new_exprs.push(self.destructure_expr(expr));
                }
                Expr::Method(Box::new(new_expr),Rc::clone(&method),new_exprs)
            },
            Expr::Field(struct_,expr,index) => {
                let new_expr = self.destructure_expr(expr);
                Expr::Field(Rc::clone(&struct_),Box::new(new_expr),*index)
            },
            Expr::TupleIndex(..) => panic!("ERROR: Expr::TupleIndex cannot occur in destructure phase"),
            Expr::Discriminant(expr) => {
                let new_expr = self.destructure_expr(expr);
                Expr::Discriminant(Box::new(new_expr))
            },
            Expr::Destructure(expr,variant_index,index) => {
                let new_expr = self.destructure_expr(expr);
                Expr::Destructure(Box::new(new_expr),*variant_index,*index)
            },
        }
    }

    pub fn destructure_range(&mut self,range: &Range) -> Range {
        match range {
            Range::Only(expr) => {
                let new_expr = self.destructure_expr(expr);
                Range::Only(Box::new(new_expr))
            },
            Range::FromTo(expr,expr2) => {
                let new_expr = self.destructure_expr(expr);
                let new_expr2 = self.destructure_expr(expr2);
                Range::FromTo(Box::new(new_expr),Box::new(new_expr2))
            },
            Range::FromToIncl(expr,expr2) => {
                let new_expr = self.destructure_expr(expr);
                let new_expr2 = self.destructure_expr(expr2);
                Range::FromToIncl(Box::new(new_expr),Box::new(new_expr2))
            },
            Range::From(expr) => {
                let new_expr = self.destructure_expr(expr);
                Range::From(Box::new(new_expr))
            },
            Range::To(expr) => {
                let new_expr = self.destructure_expr(expr);
                Range::To(Box::new(new_expr))
            },
            Range::ToIncl(expr) => {
                let new_expr = self.destructure_expr(expr);
                Range::ToIncl(Box::new(new_expr))
            },
            Range::All => Range::All,
        }
    }

    pub fn destructure_block(&mut self,block: &Block) -> Block {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter() {
            match stat {
                Stat::Let(pat,type_,expr) => {
                    let new_expr = Expr::Cast(Box::new(self.destructure_expr(expr)),type_.clone());
                    new_stats.append(&mut self.destructure_pat(pat,&new_expr));
                },
                Stat::Expr(expr) => {
                    let new_expr = self.destructure_expr(expr);
                    new_stats.push(Stat::Expr(Box::new(new_expr)));
                },
                Stat::Local(local,expr) => {
                    let new_expr = self.destructure_expr(expr);
                    new_stats.push(Stat::Local(Rc::clone(&local),Box::new(new_expr)));
                },
            }
        }
        let new_expr = if let Some(expr) = &block.expr {
            Some(Box::new(self.destructure_expr(expr)))
        }
        else {
            None
        };
        Block { stats: new_stats,expr: new_expr, }
    }

    pub fn destructure_module(&mut self,module: Module) -> Module {
        for function in module.functions.values() {
            let new_block = self.destructure_block(&function.borrow().block);
            function.borrow_mut().block = new_block;
        }
        module
    }
}
