// unfold all pattern matching and resolve enum references

use {
    crate::*,
    std::collections::HashMap,
};

struct Unfolder {
    pub enums: HashMap<String,(Vec<Variant>,Vec<Vec<usize>>)>,
}

impl Unfolder {

    // construct boolean equation to see if pattern fits scrutinee
    fn make_pat_boolean(&self,pat: &Pat,scrut: &Expr) -> Option<Expr> {
        
        match pat {
            // no influence on the result
            Pat::Wildcard |
            Pat::Rest |
            Pat::Ident(_) => None,

            // scrut == boolean
            Pat::Boolean(value) => Some(if *value { scrut.clone() } else { Expr::Not(Box::new(scrut.clone())) }),

            // scrut == integer
            Pat::Integer(value) => Some(Expr::Eq(Box::new(scrut.clone()),Box::new(Expr::Integer(*value)))),

            // scrut == float
            Pat::Float(value) => Some(Expr::Eq(Box::new(scrut.clone()),Box::new(Expr::Float(*value)))),

            // scrut == const
            Pat::Const(ident) => Some(Expr::Eq(Box::new(scrut.clone()),Box::new(Expr::Const(ident.clone())))),

            // verify all fields: (scrut.a == ...) && ... && (scrut.z == ...)
            Pat::UnknownStruct(_,identpats) => {
                // LogAnd together the necessary fields
                let mut accum: Option<Expr> = None;
                for identpat in identpats.iter() {
                    // only look at fields that actually specify a pattern
                    if let IdentPat::IdentPat(ident,pat) = identpat {
                        // recurse the pattern and create (scrut.ident == ...)
                        if let Some(expr) = self.make_pat_boolean(pat,&Expr::Field(Box::new(scrut.clone()),ident.clone())) {
                            // replace accum or LogAnd them together
                            if let Some(accum_inner) = accum {
                                accum = Some(Expr::LogAnd(Box::new(accum_inner.clone()),Box::new(expr)));
                            }
                            else {
                                accum = Some(expr);
                            }
                        }
                    }
                }
                accum
            },

            // verify all elements: (scrut[0] == ...) && ... && (scrut[N] == ...)
            Pat::Array(pats) => {
                let mut accum: Option<Expr> = None;
                let mut i = 0usize;
                for pat in pats.iter() {
                    // recurse the pattern and create (scrut[i] == ...)
                    if let Some(expr) = self.make_pat_boolean(pat,&Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64)))) {
                        // replace accum or LogAnd them together
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::LogAnd(Box::new(accum_inner.clone()),Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                    i += 1;
                }
                accum
            }

            // verify the variant and elements
            Pat::UnknownVariant(enum_ident,variantpat) => {
                if self.enums.contains_key(enum_ident) {
                    let enum_variants = &self.enums[enum_ident].0;
                    let enum_indices = &self.enums[enum_ident].1;

                    // find variant ID
                    let variant_ident = match variantpat {
                        VariantPat::Naked(ident) => ident,
                        VariantPat::Tuple(ident,_) => ident,
                        VariantPat::Struct(ident,_) => ident,
                    };
                    let mut variant_id: Option<usize> = None;
                    let mut i = 0usize;
                    for variant in enum_variants {
                        match variant {
                            Variant::Naked(ident) |
                            Variant::Tuple(ident,_) |
                            Variant::Struct(ident,_) => if *variant_ident == *ident {
                                variant_id = Some(i);
                                break;
                            }
                        }
                        i += 1;
                    }
                    let variant_id = variant_id.expect(&format!("unknown variant {} of enum {}",variant_ident,enum_ident));

                    // build boolean equation to check if this is the right variant
                    let mut accum = Expr::Eq(
                        Box::new(Expr::Field(Box::new(scrut.clone()),"id".to_string())),
                        Box::new(Expr::Integer(variant_id as i64)),
                    );

                    // and LogAnd anything else
                    match variantpat {
                        VariantPat::Naked(_) => { },
                        VariantPat::Tuple(_,pats) => {
                            let mut i = 0usize;
                            for pat in pats.iter() {
                                // recurse the pattern and create (scrut.ident == ...)
                                if let Some(expr) = self.make_pat_boolean(pat,&Expr::Field(
                                    Box::new(scrut.clone()),
                                    format!("_{}",enum_indices[variant_id][i])
                                )) {
                                    // LogAnd them together
                                    accum = Expr::LogAnd(Box::new(accum),Box::new(expr));
                                }
                                i += 1;
                            }
                        },
                        VariantPat::Struct(_,identpats) => {
                            for identpat in identpats.iter() {
                                // only look at fields that actually specify a pattern
                                if let IdentPat::IdentPat(field_ident,pat) = identpat {
                                    // lookup which field this is
                                    let mut field_index = 0usize;
                                    let mut found = false;
                                    for variant in enum_variants.iter() {
                                        match variant {
                                            Variant::Struct(ident,fields) => {
                                                if ident == variant_ident {
                                                    let mut i = 0usize;
                                                    for field in fields.iter() {
                                                        if field.0 == *field_ident {
                                                            field_index = i;
                                                            found = true;
                                                            break;
                                                        }
                                                        i += 1;
                                                    }
                                                }
                                            },
                                            _ => { },
                                        }
                                    }
                                    if !found {
                                        panic!("variant {}::{} has no field {}",enum_ident,variant_ident,field_ident);
                                    }

                                    // recurse the pattern and create (scrut.ident == ...)
                                    if let Some(expr) = self.make_pat_boolean(&pat,&Expr::Field(
                                        Box::new(scrut.clone()),
                                        format!("_{}",enum_indices[variant_id][field_index]),
                                    )) {
                                        // LogAnd them together
                                        accum = Expr::LogAnd(Box::new(accum),Box::new(expr));
                                    }
                                }
                            }
                        },
                    }
                    Some(accum)
                }
                else {
                    panic!("unknown enum {}",enum_ident);
                }
            },

            // verify all elements
            Pat::AnonTuple(pats) => {
                let mut accum: Option<Expr> = None;
                let mut i = 0usize;
                for pat in pats.iter() {
                    // recurse the pattern and create (scrut._i == ...)
                    if let Some(expr) = self.make_pat_boolean(pat,&Expr::Field(Box::new(scrut.clone()),format!("_{}",i))) {
                        // replace accum or LogAnd them together
                        if let Some(accum_inner) = accum {
                            accum = Some(Expr::LogAnd(Box::new(accum_inner.clone()),Box::new(expr)));
                        }
                        else {
                            accum = Some(expr);
                        }
                    }
                    i += 1;
                }
                accum
            },

            // (scrut > pat) && (scrut <= pat2)
            Pat::Range(pat,pat2) => {
                // for now only accept Pat::Integer and Pat::Float for the range
                match **pat {
                    Pat::Integer(low) => {
                        if let Pat::Integer(high) = **pat2 {
                            Some(Expr::LogAnd(
                                Box::new(Expr::Greater(Box::new(scrut.clone()),Box::new(Expr::Integer(low)))),
                                Box::new(Expr::LessEq(Box::new(scrut.clone()),Box::new(Expr::Integer(high))))
                            ))
                        }
                        else {
                            panic!("pattern range can only be integers or floats");
                        }
                    },
                    Pat::Float(low) => {
                        if let Pat::Float(high) = **pat2 {
                            Some(Expr::LogAnd(
                                Box::new(Expr::Greater(Box::new(scrut.clone()),Box::new(Expr::Float(low)))),
                                Box::new(Expr::LessEq(Box::new(scrut.clone()),Box::new(Expr::Float(high))))
                            ))
                        }
                        else {
                            panic!("pattern range can only be integers or floats");
                        }
                    },
                    _ => panic!("pattern range can only be integers or floats"),
                }
            }
        }
    }

    fn make_pats_boolean(&self,pats: &Vec<Pat>,temp: &str) -> Option<Expr> {
        let mut accum: Option<Expr> = None;
        for pat in pats.iter() {
            if let Some(expr) = self.make_pat_boolean(pat,&Expr::UnknownIdent(temp.to_string())) {
                if let Some(accum_inner) = accum {
                    accum = Some(Expr::LogOr(Box::new(accum_inner),Box::new(expr)));
                }
                else {
                    accum = Some(expr);
                }
            }
        }
        accum
    }

    fn destructure_pat(&self,pat: &Pat,scrut: &Expr) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        match pat {
            // no effect
            Pat::Wildcard |
            Pat::Rest |
            Pat::Boolean(_) |
            Pat::Integer(_) |
            Pat::Float(_) |
            Pat::Const(_) |
            Pat::Range(_,_) => { },

            // create LetIdent statement
            Pat::Ident(ident) => stats.push(Stat::LetIdent(
                ident.clone(),
                Box::new(scrut.clone())
            )),

            // recurse into all fields
            Pat::UnknownStruct(_,identpats) => {
                for identpat in identpats.iter() {
                    match identpat {
                        IdentPat::Wildcard |
                        IdentPat::Rest => { },
                        IdentPat::Ident(ident) => stats.push(
                            Stat::LetIdent(
                                ident.clone(),
                                Box::new(Expr::Field(Box::new(scrut.clone()),ident.clone()))
                            )
                        ),
                        IdentPat::IdentPat(ident,pat) => stats.append(
                            &mut self.destructure_pat(
                                pat,
                                &Expr::Field(Box::new(scrut.clone()),ident.clone())
                            )
                        ),
                    }
                }
            },

            Pat::Array(pats) => {
                let mut i = 0usize;
                for pat in pats.iter() {
                    stats.append(
                        &mut self.destructure_pat(
                            pat,
                            &Expr::Index(Box::new(scrut.clone()),Box::new(Expr::Integer(i as i64)))
                        )
                    );
                    i += 1;
                }
            },

            Pat::UnknownVariant(enum_ident,variantpat) => {
                if self.enums.contains_key(enum_ident) {
                    let enum_variants = &self.enums[enum_ident].0;
                    let enum_indices = &self.enums[enum_ident].1;
                    let variant_ident = match variantpat {
                        VariantPat::Naked(ident) => ident,
                        VariantPat::Tuple(ident,_) => ident,
                        VariantPat::Struct(ident,_) => ident,
                    };
                    let mut variant_id: Option<usize> = None;
                    let mut i = 0usize;
                    for variant in enum_variants {
                        match variant {
                            Variant::Naked(ident) |
                            Variant::Tuple(ident,_) |
                            Variant::Struct(ident,_) => if *variant_ident == *ident {
                                variant_id = Some(i);
                                break;
                            }
                        }
                        i += 1;
                    }
                    let variant_id = variant_id.expect(&format!("unknown variant {} of enum {}",variant_ident,enum_ident));
                    match variantpat {
                        VariantPat::Naked(_) => { },
                        VariantPat::Tuple(_,pats) => {
                            let mut i = 0usize;
                            for pat in pats.iter() {
                                stats.append(&mut self.destructure_pat(
                                    pat,
                                    &Expr::Field(
                                        Box::new(scrut.clone()),
                                        format!("_{}",enum_indices[variant_id][i])),
                                ));
                                i += 1;
                            }
                        },
                        VariantPat::Struct(_,identpats) => {
                            for identpat in identpats.iter() {
                                if let IdentPat::IdentPat(field_ident,pat) = identpat {
                                    let mut field_index = 0usize;
                                    let mut found = false;
                                    for variant in enum_variants.iter() {
                                        match variant {
                                            Variant::Struct(ident,fields) => {
                                                if ident == variant_ident {
                                                    let mut i = 0usize;
                                                    for field in fields.iter() {
                                                        if field.0 == *field_ident {
                                                            field_index = i;
                                                            found = true;
                                                            break;
                                                        }
                                                        i += 1;
                                                    }
                                                }
                                            },
                                            _ => { },
                                        }
                                    }
                                    if !found {
                                        panic!("variant {}::{} has no field {}",enum_ident,variant_ident,field_ident);
                                    }
                                    stats.append(&mut self.destructure_pat(
                                        pat,
                                        &Expr::Field(
                                            Box::new(scrut.clone()),
                                            format!("_{}",enum_indices[variant_id][field_index]),
                                        )
                                    ));
                                }
                            }
                        },
                    }
                }
                else {
                    panic!("unknown enum {}",enum_ident);
                }
            },

            Pat::AnonTuple(pats) => {
                let mut i = 0usize;
                for pat in pats.iter() {
                    stats.append(
                        &mut self.destructure_pat(
                            pat,
                            &Expr::Field(Box::new(scrut.clone()),format!("_{}",i))
                        )
                    );
                    i += 1;
                }
            },
        }
        stats
    }

    fn destructure_pats(&self,pats: &Vec<Pat>,temp: &str) -> Vec<Stat> {
        let mut stats: Vec<Stat> = Vec::new();
        for pat in pats.iter() {
            stats.append(&mut self.destructure_pat(pat,&Expr::UnknownIdent(temp.to_string())));
        }
        stats
    }

    fn block(&self,block: &mut Block) {
        let mut new_stats: Vec<Stat> = Vec::new();
        for stat in block.stats.iter_mut() {
            match stat {
                Stat::Let(pat,type_,expr) => {
                    let expr = &mut **expr;
                    self.expr(expr);
                    *expr = Expr::Cast(Box::new(expr.clone()),Box::new(*type_.clone()));
                    new_stats.append(&mut self.destructure_pat(pat,expr));
                },
                Stat::LetIdent(ident,expr) => {
                    self.expr(expr);
                    new_stats.push(Stat::LetIdent(ident.clone(),expr.clone()));
                },
                Stat::Expr(expr) => {
                    self.expr(expr);
                    new_stats.push(Stat::Expr(expr.clone()));
                },
            }
        }
        block.stats = new_stats;
        if let Some(mut expr) = block.expr.as_mut() {
            self.expr(&mut expr);
        }
    }

    fn expr(&self,expr: &mut Expr) {
        match expr {
            Expr::Boolean(_) |
            Expr::Integer(_) |
            Expr::Float(_) |
            Expr::UnknownIdent(_) |
            Expr::Const(_) => { },
            Expr::Base(_,fields) => {
                for field in fields.iter_mut() {
                    self.expr(&mut field.1);
                }
            },
            Expr::Array(exprs) |
            Expr::AnonTuple(exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },
            Expr::Cloned(expr,expr2) => {
                self.expr(expr);
                self.expr(expr2);
            },
            Expr::UnknownStruct(_,fields) => {
                for field in fields.iter_mut() {
                    self.expr(&mut field.1);
                }
            },
            Expr::UnknownCallOrTuple(_,exprs) => {
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },
            Expr::UnknownVariant(_,variantexpr) => {
                match variantexpr {
                    VariantExpr::Naked(_) => { },
                    VariantExpr::Tuple(_,exprs) => {
                        for expr in exprs.iter_mut() {
                            self.expr(expr);
                        }
                    },
                    VariantExpr::Struct(_,fields) => {
                        for field in fields.iter_mut() {
                            self.expr(&mut field.1);
                        }
                    },
                };
            },
            Expr::Field(expr,_) |
            Expr::Cast(expr,_) |
            Expr::Neg(expr) |
            Expr::Not(expr) => self.expr(expr),
            Expr::Method(expr,_,exprs) => {
                self.expr(expr);
                for expr in exprs.iter_mut() {
                    self.expr(expr);
                }
            },
            Expr::Index(expr,expr2) |
            Expr::Mul(expr,expr2) |
            Expr::Div(expr,expr2) |
            Expr::Mod(expr,expr2) |
            Expr::Add(expr,expr2) |
            Expr::Sub(expr,expr2) |
            Expr::Shl(expr,expr2) |
            Expr::Shr(expr,expr2) |
            Expr::And(expr,expr2) |
            Expr::Or(expr,expr2) |
            Expr::Xor(expr,expr2) |
            Expr::Eq(expr,expr2) |
            Expr::NotEq(expr,expr2) |
            Expr::Greater(expr,expr2) |
            Expr::Less(expr,expr2) |
            Expr::GreaterEq(expr,expr2) |
            Expr::LessEq(expr,expr2) |
            Expr::LogAnd(expr,expr2) |
            Expr::LogOr(expr,expr2) |
            Expr::Assign(expr,expr2) |
            Expr::AddAssign(expr,expr2) |
            Expr::SubAssign(expr,expr2) |
            Expr::MulAssign(expr,expr2) |
            Expr::DivAssign(expr,expr2) |
            Expr::ModAssign(expr,expr2) |
            Expr::AndAssign(expr,expr2) |
            Expr::OrAssign(expr,expr2) |
            Expr::XorAssign(expr,expr2) |
            Expr::ShlAssign(expr,expr2) |
            Expr::ShrAssign(expr,expr2) => {
                self.expr(expr);
                self.expr(expr2);
            },
            Expr::Continue => { },
            Expr::Break(expr) |
            Expr::Return(expr) => if let Some(expr) = expr {
                self.expr(expr);
            },
            Expr::Block(block) => self.block(block),
            Expr::If(expr,block,else_expr) => {
                self.expr(expr);
                self.block(block);
                if let Some(else_expr) = else_expr {
                    self.expr(else_expr);
                }
            },
            Expr::IfLet(pats,scrut,block,else_expr) => {
                self.expr(scrut);
                self.block(block);
                if let Some(else_expr) = else_expr {
                    self.expr(else_expr);
                }

                // create if-block
                let mut if_block = Block { stats: Vec::new(),expr: None, };

                // add statement to create temporary variable for scrutinee
                if_block.stats.push(Stat::LetIdent("scrut".to_string(),scrut.clone()));

                // create boolean expression from the patterns
                let condition = self.make_pats_boolean(pats,"scrut").expect("unable to create boolean condition from if let patterns");

                // create then-block
                let mut then_block = Block { stats: Vec::new(),expr: block.expr.clone(), };

                // start then-block with destructuring statements for the scrutinee
                then_block.stats.append(&mut self.destructure_pats(pats,"scrut"));

                // copy statements from the original block into then-block
                for stat in block.stats.iter() {
                    then_block.stats.push(stat.clone());
                }

                // add if statement to if-block
                if_block.expr = Some(Box::new(
                    Expr::If(
                        Box::new(condition),
                        then_block,
                        else_expr.clone(),
                    )
                ));

                // and replace in the tree
                *expr = Expr::Block(if_block);
            },
            Expr::Loop(block) => self.block(block),
            Expr::For(pat,range,block) => {
                match range {
                    Range::Only(expr) => self.expr(expr),
                    Range::FromTo(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    Range::FromToIncl(expr,expr2) => {
                        self.expr(expr);
                        self.expr(expr2);
                    },
                    Range::From(expr) => self.expr(expr),
                    Range::To(expr) => self.expr(expr),
                    Range::ToIncl(expr) => self.expr(expr),
                    Range::All => { },
                };
                self.block(block);
            },
            Expr::While(expr,block) => {
                self.expr(expr);
                self.block(block);
            },
            Expr::WhileLet(pats,scrut,block) => {
                self.expr(scrut);
                self.block(block);

                // create while-block
                let mut while_block = Block { stats: Vec::new(),expr: None, };

                // add statement to create temporary variable for scrutinee
                while_block.stats.push(Stat::LetIdent("scrut".to_string(),scrut.clone()));

                // create boolean expression from the patterns
                let condition = self.make_pats_boolean(pats,"scrut").expect("unable to create boolean condition from while let patterns");

                // create then-block
                let mut then_block = Block { stats: Vec::new(),expr: block.expr.clone(), };

                // start then-block with destructuring statements for the scrutinee
                then_block.stats.append(&mut self.destructure_pats(pats,"scrut"));

                // copy statements from the original block into then-block
                for stat in block.stats.iter() {
                    then_block.stats.push(stat.clone());
                }

                // add while statement to while-block
                while_block.expr = Some(Box::new(
                    Expr::While(
                        Box::new(condition),
                        then_block,
                    )
                ));

                // and replace the WhileLet expression
                *expr = Expr::Block(while_block);
            },            
            Expr::Match(scrut,arms) => {
                self.expr(scrut);

                // create match-block
                let mut match_block = Block { stats: Vec::new(),expr: None, };

                // add statement to create temporary variable for scrutinee
                match_block.stats.push(Stat::LetIdent("scrut".to_string(),scrut.clone()));

                // treat all arms as if-else chain, first collect separate if expressions
                let mut exprs: Vec<Expr> = Vec::new();
                let mut else_expr: Option<Box<Expr>> = None;
                for (pats,if_expr,expr) in arms {
                    // TODO: if_expr does what exactly?
                    if let Some(if_expr) = if_expr {
                        self.expr(if_expr);
                    }
                    self.expr(expr);

                    // if this condition evaluates to something, add as if-expression
                    if let Some(condition) = self.make_pats_boolean(pats,"scrut") {
                        let arm_block = Block {
                            stats: self.destructure_pats(pats,"scrut"),
                            expr: Some(Box::new(*expr.clone())),
                        };
                        exprs.push(Expr::If(Box::new(condition),arm_block,None));
                    }

                    // otherwise this is the else-expression of the match
                    else if let None = else_expr {
                        let arm_block = Block {
                            stats: Vec::new(),
                            expr: Some(Box::new(*expr.clone())),
                        };
                        else_expr = Some(Box::new(Expr::Block(arm_block)));
                    }
                    else {
                        panic!("match expression can only have one catch-all arm");
                    }

                }

                // create one big if-else chain from the individual expressions
                let mut result_expr: Option<Box<Expr>> = else_expr;
                for i in 0..exprs.len() {
                    // peel off starting at the back
                    if let Expr::If(condition,block,_) = &exprs[exprs.len() - i - 1] {
                        result_expr = Some(Box::new(Expr::If(condition.clone(),block.clone(),result_expr)));
                    }
                }

                // and replace the match expression
                *expr = *result_expr.unwrap();
            },
        }
    }
}

pub fn unfold_patterns(module: &mut Module) {
    let unfolder = Unfolder {
        enums: module.enums.clone(),
    };
    for (_,(_,_,block)) in module.functions.iter_mut() {
        unfolder.block(block);
    }
}
