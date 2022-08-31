use {
    crate::*,
    std::collections::HashMap,
};

trait ResolveIdents {
    fn resolve_consts(&mut self,consts: &HashMap<String,(Type,Expr)>);
}

impl ResolveIdents for Expr {
    fn resolve_consts(&mut self,consts: &HashMap<String,(Type,Expr)>) {
        match self {
            Expr::Ident(ident) => {
                if consts.contains_key(ident) {
                    *self = Expr::Const(ident.clone());
                }
            },
            Expr::Struct(_,fields) => {
                for (_,expr) in fields {
                    expr.resolve_consts(consts);
                }
            },
            Expr::Tuple(_,exprs) | Expr::Call(_,exprs) => {
                for expr in exprs {
                    expr.resolve_consts(consts);
                }
            },
            Expr::Array(exprs) | Expr::AnonTuple(exprs) => {
                for expr in exprs {
                    expr.resolve_consts(consts);
                }
            },
            Expr::Variant(_,variant_expr) => {
                match variant_expr {
                    VariantExpr::Naked(_) => { },
                    VariantExpr::Tuple(_,exprs) => {
                        for expr in exprs {
                            expr.resolve_consts(consts);
                        }
                    },
                    VariantExpr::Struct(_,fields) => {
                        for (_,r#type) in fields {
                            r#type.resolve_consts(consts);
                        }
                    },
                }
            },
            Expr::Field(expr,_) | Expr::TupleIndex(expr,_) => {
                expr.resolve_consts(consts);
            },
            Expr::Cloned(expr1,expr2) |
            Expr::Index(expr1,expr2) |
            Expr::Mul(expr1,expr2) |
            Expr::Div(expr1,expr2) |
            Expr::Mod(expr1,expr2) |
            Expr::Add(expr1,expr2) |
            Expr::Sub(expr1,expr2) |
            Expr::Shl(expr1,expr2) |
            Expr::Shr(expr1,expr2) |
            Expr::And(expr1,expr2) |
            Expr::Or(expr1,expr2) |
            Expr::Xor(expr1,expr2) |
            Expr::Eq(expr1,expr2) |
            Expr::NotEq(expr1,expr2) |
            Expr::Greater(expr1,expr2) |
            Expr::Less(expr1,expr2) |
            Expr::GreaterEq(expr1,expr2) |
            Expr::LessEq(expr1,expr2) |
            Expr::LogAnd(expr1,expr2) |
            Expr::LogOr(expr1,expr2) |
            Expr::Assign(expr1,expr2) |
            Expr::AddAssign(expr1,expr2) |
            Expr::SubAssign(expr1,expr2) |
            Expr::MulAssign(expr1,expr2) |
            Expr::DivAssign(expr1,expr2) |
            Expr::ModAssign(expr1,expr2) |
            Expr::AndAssign(expr1,expr2) |
            Expr::OrAssign(expr1,expr2) |
            Expr::XorAssign(expr1,expr2) |
            Expr::ShlAssign(expr1,expr2) |
            Expr::ShrAssign(expr1,expr2) => {
                expr1.resolve_consts(consts);
                expr2.resolve_consts(consts);
            },
            Expr::Cast(expr,r#type) => {
                expr.resolve_consts(consts);
                r#type.resolve_consts(consts);
            },
            Expr::Neg(expr) | Expr::Not(expr) => {
                expr.resolve_consts(consts);
            },
            Expr::Break(Some(expr)) | Expr::Return(Some(expr)) => {
                expr.resolve_consts(consts);
            },
            Expr::Block(block) => {
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
            },
            Expr::If(expr,block,else_expr) => {
                expr.resolve_consts(consts);
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
                if let Some(else_expr) = else_expr {
                    else_expr.resolve_consts(consts);
                }
            },
            Expr::IfLet(_,expr,block,else_expr) => {
                expr.resolve_consts(consts);
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
                if let Some(else_expr) = else_expr {
                    else_expr.resolve_consts(consts);
                }
            },
            Expr::Loop(block) => {
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
            },
            Expr::For(_,range,block) => {
                match range {
                    Range::Only(expr) | Range::From(expr) | Range::To(expr) | Range::ToIncl(expr) => {
                        expr.resolve_consts(consts);
                    },
                    Range::FromTo(expr1,expr2) | Range::FromToIncl(expr1,expr2) => {
                        expr1.resolve_consts(consts);
                        expr2.resolve_consts(consts);
                    },
                    Range::All => { },
                }
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
            },
            Expr::While(expr,block) => {
                expr.resolve_consts(consts);
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
            },
            Expr::WhileLet(_,expr,block) => {
                expr.resolve_consts(consts);
                for stat in block.stats {
                    stat.resolve_consts(consts);
                }
                if let Some(expr) = block.expr {
                    expr.resolve_consts(consts);
                }
            },
            Expr::Match(expr,arms) => {
                expr.resolve_consts(consts);
                for (_,if_expr,expr) in arms {
                    if let Some(if_expr) = if_expr {
                        if_expr.resolve_consts(consts);
                    }
                    expr.resolve_consts(consts);
                }
            },
            _ => { },
        }
    }
}

impl ResolveIdents for Type {
    fn resolve_consts(&mut self,consts: &HashMap<String,(Type,Expr)>) {
        match self {
            Type::Array(r#type,expr) => {
                r#type.resolve_consts(consts);
                expr.resolve_consts(consts);
            },
            Type::AnonTuple(types) => {
                for r#type in types {
                    r#type.resolve_consts(consts);
                }
            },
            _ => { },
        }
    }
}

impl ResolveIdents for Stat {
    fn resolve_consts(&mut self,consts: &HashMap<String,(Type,Expr)>) {
        match self {
            Stat::Let(_,r#type,expr) => {
                if let Some(r#type) = r#type {
                    r#type.resolve_consts(consts);
                }
            },
            Stat::Expr(expr) => {
                expr.resolve_consts(consts);
            },
        }
    }
}

fn resolve_identifiers(module: &mut Module) {
    for (_,(params,return_type,block)) in module.functions {
        for (_,r#type) in params {
            r#type.resolve_consts(&module.consts);
        }
        if let Some(return_type) = return_type {
            return_type.resolve_consts(&module.consts);
        }
        for stat in block.stats {
            stat.resolve_consts(&module.consts);
        }
        if let Some(expr) = block.expr {
            expr.resolve_consts(&module.consts);
        }
    }
    for (_,fields) in module.structs {
        for (_,r#type) in fields {
            r#type.resolve_consts(&module.consts);
        }
    }
    for (_,types) in module.tuples {
        for r#type in types {
            r#type.resolve_consts(&module.consts);
        }
    }
    for (_,variants) in module.enums {
        for variant in variants {
            match variant {
                Variant::Naked(_) => { },
                Variant::Tuple(_,types) => {
                    for r#type in types {
                        r#type.resolve_consts(&module.consts);
                    }
                },
                Variant::Struct(_,fields) => {
                    for (_,r#type) in fields {
                        r#type.resolve_consts(&module.consts);
                    }
                },
            }
        }
    }
    for (_,(r#type,expr)) in module.consts {
        r#type.resolve_consts(&module.consts);
        expr.resolve_consts(&module.consts);
    }
}

pub(crate) fn render_vertex_trait(ident: &str,fields: &Vec<(String,Type)>) -> String {

    // make sure all fields are base types
    let mut out_fields: Vec<(String,BaseType)> = Vec::new();
    for (ident,r#type) in fields {
        if let Type::Base(base_type) = *r#type {
            out_fields.push((*ident,base_type));
        }
    }

    let mut r = format!("impl Vertex for {} {{\n",ident);
    r += "    fn get_fields() -> Vec<(String,BaseType)> {{\n";
    r += "        vec![\n";
    let mut first_type = true;
    for (ident,base_type) in out_fields {
        if !first_type {
            r += ",\n";
        }
        r += &format!("            (\"{}\".to_string(),BaseType::{})",ident,base_type.variant());
        first_type = false;
    }
    r += "        ]\n";
    r += "    }\n";
    r += "}";
    r
}

pub(crate) fn render_vertex_shader(module: &mut Module,vertex: &str) -> String {
    resolve_identifiers(module);
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {{\n";
    r += &format!("        compile_vertex_shader({},\"{}\",{}::get_fields())\n",render_module(module),vertex,vertex);
    r += "    }}\n";
    r += "}}\n";
    r
}

pub(crate) fn render_fragment_shader(module: &mut Module) -> String {
    resolve_identifiers(module);
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {{\n";
    r += &format!("        compile_fragment_shader({})\n",render_module(module));
    r += "    }}\n";
    r += "}}\n";
    r
}
