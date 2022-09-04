use crate::*;

fn render_range(range: &Range) -> String {
    match range {
        Range::Only(expr) => format!("sr::Range::Only({})",render_expr(expr)),
        Range::FromTo(expr,expr2) => format!("sr::Range::FromTo({},{})",render_expr(expr),render_expr(expr2)),
        Range::FromToIncl(expr,expr2) => format!("sr::Range::FromToIncl({},{})",render_expr(expr),render_expr(expr2)),
        Range::From(expr) => format!("sr::Range::From({})",render_expr(expr)),
        Range::To(expr) => format!("sr::Range::To({})",render_expr(expr)),
        Range::ToIncl(expr) => format!("sr::Range::ToIncl({})",render_expr(expr)),
        Range::All => "sr::Range::All".to_string(),
    }
}

fn render_expr(expr: &Expr) -> String {
    match expr {
        Expr::Boolean(value) => format!("sr::Expr::Boolean({})",if *value { "true" } else { "false" }),
        Expr::Integer(value) => format!("sr::Expr::Integer({})",value),
        Expr::Float(value) => format!("sr::Expr::Float({})",value),
        Expr::Base(base_type,fields) => {
            let mut r = format!("sr::Expr::Base(sr::BaseType::{},[",base_type.variant());
            for (ident,expr) in fields {
                r += &format!("(\"{}\",{}),",ident,render_expr(expr));
            }
            r
        },
        Expr::Ident(_) => panic!("render_expr: Expr::Ident shouldn't exist"),
        Expr::Local(ident,_) => format!("sr::Expr::Local(\"{}\")",ident),
        Expr::Param(ident,_) => format!("sr::Expr::Param(\"{}\")",ident),
        Expr::Const(ident,_) => format!("sr::Expr::Const(\"{}\")",ident),
        Expr::Array(exprs) => {
            let mut r = "sr::Expr::Array([".to_string();
            for expr in exprs {
                r += &render_expr(expr);
                r += ",";
            }
            r += "])";
            r
        },
        Expr::Cloned(expr,expr2) => format!("sr::Expr::Cloned({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Struct(ident,fields) => {
            let mut r = format!("sr::Expr::Struct(\"{}\",vec![",ident);
            for (ident,expr) in fields {
                r += &format!("(\"{}\",{}),",ident,render_expr(expr));
            }
            r
        },
        Expr::Tuple(ident,exprs) => {
            let mut r = format!("sr::Expr::Tuple(\"{}\",vec![",ident);
            for expr in exprs {
                r += &format!("{},",expr);
            }
            r += "])";
            r
        },
        Expr::AnonTuple(exprs) => {
            let mut r = "sr::Expr::AnonTuple(vec![".to_string();
            for expr in exprs {
                r += &format!("{},",expr);
            }
            r += "])";
            r
        },
        Expr::Variant(_,_) => panic!("render_expr: TODO: Expr::Variant"),
        Expr::Call(ident,exprs) => {
            let mut r = format!("sr::Expr::Call(\"{}\",vec![",ident);
            for expr in exprs {
                r += &render_expr(expr);
                r += ",";
            }
            r += ",])";
            r
        },
        Expr::Field(expr,ident) => format!("sr::Expr::Field({},\"{}\")",render_expr(expr),ident),
        Expr::TupleIndex(expr,index) => format!("sr::Expr::TupleIndex({},{})",render_expr(expr),index),
        Expr::Index(expr,expr2) => format!("sr::Expr::Index({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Cast(expr,ty) => format!("sr::Expr::Cast({},{})",render_expr(expr),render_type(ty)),
        Expr::Neg(expr) => format!("sr::Expr::Neg({})",render_expr(expr)),
        Expr::Not(expr) => format!("sr::Expr::Not({})",render_expr(expr)),
        Expr::Mul(expr,expr2) => format!("sr::Expr::Mul({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Div(expr,expr2) => format!("sr::Expr::Div({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Mod(expr,expr2) => format!("sr::Expr::Mod({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Add(expr,expr2) => format!("sr::Expr::Add({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Sub(expr,expr2) => format!("sr::Expr::Sub({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Shl(expr,expr2) => format!("sr::Expr::Shl({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Shr(expr,expr2) => format!("sr::Expr::Shr({},{})",render_expr(expr),render_expr(expr2)),
        Expr::And(expr,expr2) => format!("sr::Expr::And({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Or(expr,expr2) => format!("sr::Expr::Or({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Xor(expr,expr2) => format!("sr::Expr::Xor({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Eq(expr,expr2) => format!("sr::Expr::Eq({},{})",render_expr(expr),render_expr(expr2)),
        Expr::NotEq(expr,expr2) => format!("sr::Expr::NotEq({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Greater(expr,expr2) => format!("sr::Expr::Greater({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Less(expr,expr2) => format!("sr::Expr::Less({},{})",render_expr(expr),render_expr(expr2)),
        Expr::GreaterEq(expr,expr2) => format!("sr::Expr::GreaterEq({},{})",render_expr(expr),render_expr(expr2)),
        Expr::LessEq(expr,expr2) => format!("sr::Expr::LessEq({},{})",render_expr(expr),render_expr(expr2)),
        Expr::LogAnd(expr,expr2) => format!("sr::Expr::LogAnd({},{})",render_expr(expr),render_expr(expr2)),
        Expr::LogOr(expr,expr2) => format!("sr::Expr::LogOr({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Assign(expr,expr2) => format!("sr::Expr::Assign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::AddAssign(expr,expr2) => format!("sr::Expr::AddAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::SubAssign(expr,expr2) => format!("sr::Expr::SubAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::MulAssign(expr,expr2) => format!("sr::Expr::MulAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::DivAssign(expr,expr2) => format!("sr::Expr::DivAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::ModAssign(expr,expr2) => format!("sr::Expr::ModAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::AndAssign(expr,expr2) => format!("sr::Expr::AndAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::OrAssign(expr,expr2) => format!("sr::Expr::OrAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::XorAssign(expr,expr2) => format!("sr::Expr::XorAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::ShlAssign(expr,expr2) => format!("sr::Expr::ShlAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::ShrAssign(expr,expr2) => format!("sr::Expr::ShrAssign({},{})",render_expr(expr),render_expr(expr2)),
        Expr::Continue => format!("sr::Expr::Continue"),
        Expr::Break(expr) => {
            if let Some(expr) = expr {
                format!("sr::Expr::Break(Some({}))",render_expr(expr))
            }
            else {
                "sr::Expr::Break(None)".to_string()
            }
        },
        Expr::Return(expr) => {
            if let Some(expr) = expr {
                format!("sr::Expr::Return(Some({}))",render_expr(expr))
            }
            else {
                "sr::Expr::Return(None)".to_string()
            }
        },
        Expr::Block(block) => format!("sr::Expr::Block({})",render_block(block)),
        Expr::If(expr,block,else_expr) => {
            let mut r = format!("sr::Expr::If({},{},",render_expr(expr),render_block(block));
            if let Some(else_expr) = else_expr {
                r += &format!("Some({})",render_expr(else_expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        Expr::IfLet(pats,expr,block,else_expr) => {
            let mut r = "sr::Expr::IfLet(vec![".to_string();
            for pat in pats {
                r += &render_pat(pat);
                r += ",";
            }
            r += &format!("],{},{},",render_expr(expr),render_block(block));
            if let Some(else_expr) = else_expr {
                r += &format!("Some({})",render_expr(else_expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        Expr::Loop(block) => format!("sr::Expr::Loop({})",render_block(block)),
        Expr::For(pats,range,block) => {
            let mut r = "sr::Expr::For(vec![".to_string();
            for pat in pats {
                r += &render_pat(pat);
                r += ",";
            }
            r += &format!("],{},{})",render_range(range),render_block(block));
            r
        },
        Expr::While(expr,block) => format!("sr::Expr::While({},{})",render_expr(expr),render_block(block)),
        Expr::WhileLet(pats,expr,block) => {
            let mut r = "sr::Expr::WhileLet(vec![".to_string();
            for pat in pats {
                r += &render_pat(pat);
                r += ",";
            }
            r += &format!("],{},{})",render_expr(expr),render_block(block));
            r
        },
        Expr::Match(expr,arms) => {
            let mut r = format!("sr::Expr::Match({},vec![",render_expr(expr));
            for (pats,if_expr,expr) in arms {
                r += "(vec![";
                for pat in pats {
                    r += &render_pat(pat);
                    r += ",";
                }
                r += "],";
                if let Some(if_expr) = if_expr {
                    r += &format!("Some({})",render_expr(if_expr));
                }
                else {
                    r += "None";
                }
                r += &format!(",{}),",render_expr(expr));
            }
            r += "])";
            r
        },
    }
}

fn render_type(ty: &Type) -> String {
    match ty {
        Type::Inferred => "sr::Type::Inferred".to_string(),
        Type::Integer => "sr::Type::Integer".to_string(),
        Type::Float => "sr::Type::Float".to_string(),
        Type::Void => "sr::Type::Void".to_string(),
        Type::Base(base_type) => format!("sr::Type::Base(sr::BaseType::{})",base_type.variant()),
        Type::Ident(_) => panic!("render_type: Type::Ident shouldn't exist"),
        Type::Struct(ident) => format!("sr::Type::Struct(\"{}\")",ident),
        Type::Tuple(ident) => format!("sr::Type::Tuple(\"{}\")",ident),
        Type::Enum(ident) => format!("sr::Type::Enum(\"{}\")",ident),
        Type::Array(ty,expr) => format!("sr::Type::Array({},{})",render_type(ty),render_expr(expr)),
        Type::AnonTuple(types) => {
            let mut r = "sr::Type::AnonTuple(vec![".to_string();
            for ty in types {
                r += &format!("{},",render_type(ty));
            }
            r += "])";
            r
        },
    }
}

fn render_pat(pat: &Pat) -> String {
    match pat {
        Pat::Wildcard => "sr::Pat::Wildcard".to_string(),
        Pat::Rest => "sr::Pat::Rest".to_string(),
        Pat::Boolean(value) => format!("sr::Pat::Boolean({})",if *value { "true" } else { "false" }),
        Pat::Integer(value) => format!("sr::Pat::Integer({})",*value),
        Pat::Float(value) => format!("sr::Pat::Float({})",*value),
        Pat::Ident(_) => panic!("render_pat: Pat::Ident shouldn't exist"),
        Pat::Const(ident,ty) => format!("sr::Pat::Const(\"{}\",{})",ident,render_type(ty)),
        Pat::Struct(ident,identpats) => {
            let mut r = format!("sr::Pat::Struct(\"{}\",vec![",ident);
            for identpat in identpats {
                r += &match identpat {
                    IdentPat::Wildcard => "sr::IdentPat::Wildcard,".to_string(),
                    IdentPat::Rest => "sr::IdentPat::Rest,".to_string(),
                    IdentPat::Ident(ident) => format!("sr::IdentPat::Ident(\"{}\"),",ident),
                    IdentPat::IdentPat(ident,pat) => format!("sr::IdentPat::IdentPat(\"{}\",{}),",ident,render_pat(pat)),
                }
            }
            r += "])";
            r
        },
        Pat::Tuple(ident,pats) => {
            let mut r = format!("sr::Pat::Tuple(\"{}\",vec![",ident);
            for pat in pats {
                r += &format!("{},",render_pat(pat));
            }
            r += "])";
            r
        },
        Pat::Array(pats) => {
            let mut r = "sr::Pat::Array(vec![".to_string();
            for pat in pats {
                r += &render_pat(pat);
                r += ",";
            }
            r += "])";
            r
        },
        Pat::AnonTuple(pats) => {
            let mut r = "sr::Pat::AnonTuple(vec![".to_string();
            for pat in pats {
                r += &render_pat(pat);
                r += ",";
            }
            r += "])";
            r
        },
        Pat::Variant(_,_) => panic!("render_pat: TODO: Pat::Variant"),
        Pat::Range(pat,pat2) => format!("sr::Pat::Range({},{})",render_pat(pat),render_pat(pat2)),
    }
}

fn render_stat(stat: &Stat) -> String {
    match stat {
        Stat::Let(pat,ty,expr) => {
            let mut r = format!("sr::Stat::Let({},",render_pat(pat));
            if let Some(ty) = ty {
                r += &format!("Some({})",render_type(ty));
            }
            else {
                r += "None";
            }
            r += &format!(",{})",render_expr(expr));
            r
        },
        Stat::Expr(expr) => {
            format!("sr::Stat::Expr({})",render_expr(expr))
        },
    }
}

fn render_block(block: &Block) -> String {
    let mut r = "sr::Block { stats: vec![".to_string();
    for stat in &block.stats {
        r += &render_stat(stat);
        r += ",";
    }
    r += "],expr: ";
    if let Some(expr) = &block.expr {
        r += &render_expr(expr);
    }
    else {
        r += "None";
    }
    r += ", }";
    r
}

fn render_module(module: &Module) -> String {
    let mut r = "let mut functions: HashMap<&'static str,(Vec<(&'static str,Type)>,Type,Block)> = HashMap::new();\n".to_string();
    for ident in module.functions.keys() {
        let (params,return_type,block) = &module.functions[ident];
        r += "let mut params: Vec<(&'static str,Type)> = Vec::new();\n";
        for (ident,ty) in params {
            r += &format!("params.push((\"{}\".to_string(),{}));\n",ident,render_type(ty));
        }
        r += &format!("let return_type = {};\n",render_type(&return_type));
        r += &format!("let block = {};\n",render_block(&block));
        r += &format!("functions[{}] = (params,return_type,block);\n",ident);
    }
    r += "let mut structs: HashMap<&'static str,Vec<(&'static str,Type)>> = HashMap::new();\n";
    for ident in module.structs.keys() {
        let fields = &module.structs[ident];
        r += "let mut fields: Vec<(&'static str,Type)> = Vec::new();\n";
        for (ident,ty) in fields {
            r += &format!("fields.push((\"{}\",{}));\n",ident,render_type(&ty));
        }
    }
    r += "let mut consts: HashMap<&'static str,(Type,Expr)> = HashMap::new();\n";
    for ident in module.consts.keys() {
        let (ty,expr) = &module.consts[ident];
        r += &format!("consts[\"{}\"] = ({},{});\n",ident,render_type(&ty),render_expr(&expr));
    }
    r += &format!("let module = Module {{ ident: \"{}\",functions,structs,consts, }}\n",module.ident);
    r
}

pub(crate) fn render_vertex_trait(ident: &str,fields: &Vec<(String,Type)>) -> String {

    // make sure all fields are base types
    let mut out_fields: Vec<(String,sr::BaseType)> = Vec::new();
    for (ident,r#type) in fields {
        if let Type::Base(base_type) = r#type {
            out_fields.push((ident.clone(),base_type.clone()));
        }
    }

    let mut r = format!("impl Vertex for {} {{\n",ident);
    r += "    fn get_fields() -> Vec<(&'static str,BaseType)> {\n";
    r += "        vec![\n";
    for (ident,base_type) in out_fields {
        r += &format!("            (\"{}\",sr::BaseType::{}),\n",ident,base_type.variant());
    }
    r += "        ]\n";
    r += "    }\n";
    r += "}";
    r
}

pub(crate) fn render_vertex_shader(mut module: Module,vertex: &str) -> String {
    resolve_module(&mut module);
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n";
    r += &format!("        {}\n",render_module(&module));
    r += &format!("        compile_vertex_shader(module,\"{}\",{}::get_fields())\n",vertex,vertex);
    r += "    }}\n";
    r += "}}\n";
    r
}

pub(crate) fn render_fragment_shader(mut module: Module) -> String {
    resolve_module(&mut module);
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n";
    r += &format!("        {}\n",render_module(&module));
    r += &format!("        compile_fragment_shader(module)\n");
    r += "    }}\n";
    r += "}}\n";
    r
}
