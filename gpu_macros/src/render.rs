use crate::*;

struct Renderer { }

impl Renderer {

    fn range(&self,range: &Range) -> String {
        match range {
            Range::Only(expr) => format!("sr::Range::Only(Box::new({}))",self.expr(expr)),
            Range::FromTo(expr,expr2) => format!("sr::Range::FromTo(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Range::FromToIncl(expr,expr2) => format!("sr::Range::FromToIncl(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Range::From(expr) => format!("sr::Range::From(Box::new({}))",self.expr(expr)),
            Range::To(expr) => format!("sr::Range::To(Box::new({}))",self.expr(expr)),
            Range::ToIncl(expr) => format!("sr::Range::ToIncl(Box::new({}))",self.expr(expr)),
            Range::All => "sr::Range::All".to_string(),
        }
    }

    fn expr(&self,expr: &Expr) -> String {
        match expr {
            Expr::Boolean(value) => format!("sr::Expr::Boolean({})",if *value { "true" } else { "false" }),
            Expr::Integer(value) => format!("sr::Expr::Integer({})",value),
            Expr::Float(value) => format!("sr::Expr::Float({}f64)",value),
            Expr::Base(base_type,fields) => {
                let mut r = format!("sr::Expr::Base(sr::BaseType::{},vec![",base_type.variant());
                for (ident,expr) in fields {
                    r += &format!("(\"{}\".to_string(),{}),",ident,self.expr(expr));
                }
                r += "])";
                r
            },
            Expr::Ident(ident) => format!("sr::Expr::Ident(\"{}\".to_string())",ident),
            Expr::Array(exprs) => {
                let mut r = "sr::Expr::Array([".to_string();
                for expr in exprs {
                    r += &self.expr(expr);
                    r += ",";
                }
                r += "])";
                r
            },
            Expr::Cloned(expr,expr2) => format!("sr::Expr::Cloned(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Struct(ident,fields) => {
                let mut r = format!("sr::Expr::Struct(\"{}\".to_string(),vec![",ident);
                for (ident,expr) in fields {
                    r += &format!("(\"{}\".to_string(),{}),",ident,self.expr(expr));
                }
                r += "])";
                r
            },
            Expr::Variant(ident,variantexpr) => {
                match variantexpr {
                    VariantExpr::Naked(ident2) => format!("sr::Expr::Variant(\"{}\".to_string(),VariantExpr::Naked(\"{}\".to_string()))",ident,ident2),
                    VariantExpr::Tuple(ident2,exprs) => {
                        let mut r = format!("sr::Expr::Variant(\"{}\".to_string(),VariantExpr::Tuple(\"{}\".to_string(),vec![",ident,ident2);
                        for expr in exprs {
                            r += &format!("{},",self.expr(expr));
                        }
                        r += "]))";
                        r
                    },
                    VariantExpr::Struct(ident2,fields) => {
                        let mut r = format!("sr::Expr::Variant(\"{}\".to_string(),VariantExpr::Struct(\"{}\".to_string(),vec![",ident,ident2);
                        for (ident,expr) in fields {
                            r += &format!("(\"{}\".to_string(),{})",ident,self.expr(expr));
                        }
                        r += "]))";
                        r
                    },
                }
            },
            Expr::Call(ident,exprs) => {
                let mut r = format!("sr::Expr::Call(\"{}\".to_string(),vec![",ident);
                for expr in exprs {
                    r += &self.expr(expr);
                    r += ",";
                }
                r += "])";
                r
            },
            Expr::Field(expr,ident) => format!("sr::Expr::Field(Box::new({}),\"{}\".to_string())",self.expr(expr),ident),
            Expr::Index(expr,expr2) => format!("sr::Expr::Index(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Cast(expr,ty) => format!("sr::Expr::Cast(Box::new({}),Box::new({}))",self.expr(expr),self.type_(ty)),
            Expr::AnonTuple(exprs) => {
                let mut r = "sr::Expr::AnonTuple(vec![".to_string();
                for expr in exprs {
                    r += &self.expr(expr);
                    r += ",";
                }
                r += "])";
                r
            },
            Expr::Neg(expr) => format!("sr::Expr::Neg(Box::new({}))",self.expr(expr)),
            Expr::Not(expr) => format!("sr::Expr::Not(Box::new({}))",self.expr(expr)),
            Expr::Mul(expr,expr2) => format!("sr::Expr::Mul(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Div(expr,expr2) => format!("sr::Expr::Div(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Mod(expr,expr2) => format!("sr::Expr::Mod(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Add(expr,expr2) => format!("sr::Expr::Add(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Sub(expr,expr2) => format!("sr::Expr::Sub(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Shl(expr,expr2) => format!("sr::Expr::Shl(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Shr(expr,expr2) => format!("sr::Expr::Shr(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::And(expr,expr2) => format!("sr::Expr::And(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Or(expr,expr2) => format!("sr::Expr::Or(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Xor(expr,expr2) => format!("sr::Expr::Xor(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Eq(expr,expr2) => format!("sr::Expr::Eq(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::NotEq(expr,expr2) => format!("sr::Expr::NotEq(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Greater(expr,expr2) => format!("sr::Expr::Greater(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Less(expr,expr2) => format!("sr::Expr::Less(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::GreaterEq(expr,expr2) => format!("sr::Expr::GreaterEq(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::LessEq(expr,expr2) => format!("sr::Expr::LessEq(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::LogAnd(expr,expr2) => format!("sr::Expr::LogAnd(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::LogOr(expr,expr2) => format!("sr::Expr::LogOr(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Assign(expr,expr2) => format!("sr::Expr::Assign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::AddAssign(expr,expr2) => format!("sr::Expr::AddAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::SubAssign(expr,expr2) => format!("sr::Expr::SubAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::MulAssign(expr,expr2) => format!("sr::Expr::MulAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::DivAssign(expr,expr2) => format!("sr::Expr::DivAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::ModAssign(expr,expr2) => format!("sr::Expr::ModAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::AndAssign(expr,expr2) => format!("sr::Expr::AndAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::OrAssign(expr,expr2) => format!("sr::Expr::OrAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::XorAssign(expr,expr2) => format!("sr::Expr::XorAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::ShlAssign(expr,expr2) => format!("sr::Expr::ShlAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::ShrAssign(expr,expr2) => format!("sr::Expr::ShrAssign(Box::new({}),Box::new({}))",self.expr(expr),self.expr(expr2)),
            Expr::Continue => format!("sr::Expr::Continue"),
            Expr::Break(expr) => {
                if let Some(expr) = expr {
                    format!("sr::Expr::Break(Some(Box::new({})))",self.expr(expr))
                }
                else {
                    "sr::Expr::Break(None)".to_string()
                }
            },
            Expr::Return(expr) => {
                if let Some(expr) = expr {
                    format!("sr::Expr::Return(Some(Box::new({})))",self.expr(expr))
                }
                else {
                    "sr::Expr::Return(None)".to_string()
                }
            },
            Expr::Block(block) => format!("sr::Expr::Block({})",self.block(block)),
            Expr::If(expr,block,else_expr) => {
                let mut r = format!("sr::Expr::If(Box::new({}),{},",self.expr(expr),self.block(block));
                if let Some(else_expr) = else_expr {
                    r += &format!("Some(Box::new({}))",self.expr(else_expr));
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
                    r += &self.pat(pat);
                    r += ",";
                }
                r += &format!("],Box::new({}),{},",self.expr(expr),self.block(block));
                if let Some(else_expr) = else_expr {
                    r += &format!("Some(Box::new({}))",self.expr(else_expr));
                }
                else {
                    r += "None";
                }
                r += ")";
                r
            },
            Expr::Loop(block) => format!("sr::Expr::Loop({})",self.block(block)),
            Expr::For(pats,range,block) => {
                let mut r = "sr::Expr::For(vec![".to_string();
                for pat in pats {
                    r += &self.pat(pat);
                    r += ",";
                }
                r += &format!("],{},{})",self.range(range),self.block(block));
                r
            },
            Expr::While(expr,block) => format!("sr::Expr::While(Box::new({}),{})",self.expr(expr),self.block(block)),
            Expr::WhileLet(pats,expr,block) => {
                let mut r = "sr::Expr::WhileLet(vec![".to_string();
                for pat in pats {
                    r += &self.pat(pat);
                    r += ",";
                }
                r += &format!("],Box::new({}),{})",self.expr(expr),self.block(block));
                r
            },
            Expr::Match(expr,arms) => {
                let mut r = format!("sr::Expr::Match({},vec![",self.expr(expr));
                for (pats,if_expr,expr) in arms {
                    r += "(vec![";
                    for pat in pats {
                        r += &self.pat(pat);
                        r += ",";
                    }
                    r += "],";
                    if let Some(if_expr) = if_expr {
                        r += &format!("Some(Box::new({}))",self.expr(if_expr));
                    }
                    else {
                        r += "None";
                    }
                    r += &format!(",Box::new({})),",self.expr(expr));
                }
                r += "])";
                r
            },
        }
    }

    fn type_(&self,ty: &Type) -> String {
        match ty {
            Type::Void => "sr::Type::Void".to_string(),
            Type::Base(base_type) => format!("sr::Type::Base(sr::BaseType::{})",base_type.variant()),
            Type::Ident(ident) => format!("sr::Type::Ident(\"{}\".to_string())",ident),
            Type::Array(ty,expr) => format!("sr::Type::Array(Box::new({}),Box::new({}))",self.type_(ty),self.expr(expr)),
        }
    }

    fn pat(&self,pat: &Pat) -> String {
        match pat {
            Pat::Wildcard => "sr::Pat::Wildcard".to_string(),
            Pat::Rest => "sr::Pat::Rest".to_string(),
            Pat::Boolean(value) => format!("sr::Pat::Boolean({})",if *value { "true" } else { "false" }),
            Pat::Integer(value) => format!("sr::Pat::Integer({})",*value),
            Pat::Float(value) => format!("sr::Pat::Float({})",*value),
            Pat::Ident(ident) => format!("sr::Pat::Ident(\"{}\")",ident),
            Pat::Struct(ident,identpats) => {
                let mut r = format!("sr::Pat::Struct(\"{}\".to_string(),vec![",ident);
                for identpat in identpats {
                    r += &match identpat {
                        IdentPat::Wildcard => "sr::IdentPat::Wildcard,".to_string(),
                        IdentPat::Rest => "sr::IdentPat::Rest,".to_string(),
                        IdentPat::Ident(ident) => format!("sr::IdentPat::Ident(\"{}\".to_string()),",ident),
                        IdentPat::IdentPat(ident,pat) => format!("sr::IdentPat::IdentPat(\"{}\".to_string(),{}),",ident,self.pat(pat)),
                    }
                }
                r += "])";
                r
            },
            Pat::Array(pats) => {
                let mut r = "sr::Pat::Array(vec![".to_string();
                for pat in pats {
                    r += &self.pat(pat);
                    r += ",";
                }
                r += "])";
                r
            },
            Pat::Variant(ident,variantpat) => {
                match variantpat {
                    VariantPat::Naked(ident2) => format!("sr::Pat::Variant(\"{}\".to_string(),VariantPat::Naked(\"{}\".to_string()))",ident,ident2),
                    VariantPat::Tuple(ident2,pats) => {
                        let mut r = format!("sr::Pat::Variant(\"{}\".to_string(),VariantPat::Tuple(\"{}\".to_string(),vec![",ident,ident2);
                        for pat in pats {
                            r += &format!("{},",self.pat(pat));
                        }
                        r += "]))";
                        r
                    },
                    VariantPat::Struct(ident2,identpats) => {
                        let mut r = format!("sr::Pat::Variant(\"{}\".to_string(),VariantPat::Struct(\"{}\".to_string(),vec![",ident,ident2);
                        for identpat in identpats {
                            match identpat {
                                IdentPat::Wildcard => r += "sr::IdentPat::Wildcard,",
                                IdentPat::Rest => r += "sr::IdentPat::Rest,",
                                IdentPat::Ident(ident) => r += &format!("sr::IdentPat::Ident(\"{}\".to_string()),",ident),
                                IdentPat::IdentPat(ident,pat) => r += &format!("sr::IdentPat::IdentPat(\"{}\".to_string(),{}),",ident,self.pat(pat)),
                            }
                        }
                        r += "]))";
                        r
                    },
                }
            },
            Pat::Range(pat,pat2) => format!("sr::Pat::Range(Box::new({}),Box::new({}))",self.pat(pat),self.pat(pat2)),
        }
    }

    fn stat(&self,stat: &Stat) -> String {
        match stat {
            Stat::Let(ident,ty,expr) => {
                let mut r = format!("sr::Stat::Let({},",ident);
                if let Some(ty) = ty {
                    r += &format!("Some(Box::new({}))",self.type_(ty));
                }
                else {
                    r += "None";
                }
                r += &format!(",Box::new({}))",self.expr(expr));
                r
            },
            Stat::Expr(expr) => {
                format!("sr::Stat::Expr(Box::new({}))",self.expr(expr))
            },
        }
    }

    fn block(&self,block: &Block) -> String {
        let mut r = "sr::Block { stats: vec![".to_string();
        for stat in &block.stats {
            r += &self.stat(stat);
            r += ",";
        }
        r += "],expr: ";
        if let Some(expr) = &block.expr {
            r += &format!("Some(Box::new({}))",self.expr(expr));
        }
        else {
            r += "None";
        }
        r += ", }";
        r
    }

    fn variant(&self,variant: &Variant) -> String {
        match variant {
            Variant::Naked(ident) => format!("Variant::Naked(\"{}\".to_string())",ident),
            Variant::Tuple(ident,types) => {
                let mut r = format!("Variant::Tuple(\"{}\".to_string(),vec![",ident);
                for ty in types {
                    r += &format!("{},",self.type_(ty));
                }
                r += "])";
                r
            },
            Variant::Struct(ident,fields) => {
                let mut r = format!("Variant::Struct(\"{}\".to_string(),vec![",ident);
                for (ident,ty) in fields {
                    r += &format!("(\"{}\".to_string(),{}),",ident,self.type_(ty));
                }
                r += "])";
                r
            },
        }
    }

    fn module(&self,module: &Module) -> String {
        let mut r = "let mut functions: HashMap<String,(Vec<(String,sr::Type)>,sr::Type,sr::Block)> = HashMap::new();\n\n".to_string();

        r += "        let mut consts: HashMap<String,(sr::Type,sr::Expr)> = HashMap::new();\n\n";
        for ident in module.consts.keys() {
            let (ty,expr) = &module.consts[ident];
            r += &format!("        consts.insert(\"{}\".to_string(),({},{}));\n\n",ident,self.type_(&ty),self.expr(&expr));
        }

        r += "        let mut structs: HashMap<String,Vec<(String,sr::Type)>> = HashMap::new();\n\n";
        for ident in module.structs.keys() {
            let fields = &module.structs[ident];
            r += "        let mut fields: Vec<(String,sr::Type)> = Vec::new();\n";
            for (ident,ty) in fields {
                r += &format!("        fields.push((\"{}\".to_string(),{}));\n",ident,self.type_(&ty));
            }
            r += &format!("        structs.insert(\"{}\".to_string(),fields);\n\n",ident);
        }

        r += "        let mut anon_tuple_structs: HashMap<String,Vec<(String,sr::Type)>> = HashMap::new();\n\n";
        for ident in module.anon_tuple_structs.keys() {
            let fields = &module.anon_tuple_structs[ident];
            r += "        let mut fields: Vec<(String,sr::Type)> = Vec::new();\n";
            for (ident,ty) in fields {
                r += &format!("        fields.push((\"{}\".to_string(),{}));\n",ident,self.type_(&ty));
            }
            r += &format!("        anon_tuple_structs.insert(\"{}\".to_string(),fields);\n\n",ident);
        }

        r += "        let mut enums: HashMap<String,Vec<sr::Variant>> = HashMap::new();\n\n";
        for ident in module.enums.keys() {
            let variants = &module.enums[ident];
            r += "        let mut variants: Vec<Variant> = Vec::new();\n";
            for variant in variants {
                r += &format!("        variants.push({});",self.variant(variant));
            }
            r += &format!("        enums.insert(\"{}\".to_string(),variants);\n\n",ident);
        }

        for ident in module.functions.keys() {
            let (params,return_type,block) = &module.functions[ident];
            r += "        let mut params: Vec<(String,sr::Type)> = Vec::new();\n";
            for (ident,ty) in params {
                r += &format!("        params.push((\"{}\".to_string(),{}));\n",ident,self.type_(ty));
            }
            r += &format!("        let return_type = {};\n",self.type_(&return_type));
            r += &format!("        let block = {};\n",self.block(&block));
            r += &format!("        functions.insert(\"{}\".to_string(),(params,return_type,block));\n\n",ident);
        }

        r += &format!("        let module = sr::Module {{ ident: \"{}\".to_string(),consts,structs,anon_tuple_structs,enums,functions, }};\n",module.ident);

        r
    }
}

pub(crate) fn render_vertex_trait(ident: &str,fields: &Vec<(String,Type)>) -> String {

    // make sure all fields are base types
    let mut out_fields: Vec<(String,sr::BaseType)> = Vec::new();
    for (ident,r#type) in fields {
        if let Type::Base(base_type) = r#type {
            out_fields.push((ident.clone(),base_type.clone()));
        }
        else {
            panic!("all fields of the vertex struct should be base types");
        }
    }

    let mut r = format!("impl Vertex for {} {{\n",ident);
    r += "    fn get_fields() -> Vec<(String,sr::BaseType)> {\n";
    r += "        vec![\n";
    for (ident,base_type) in out_fields {
        r += &format!("            (\"{}\".to_string(),sr::BaseType::{}),\n",ident,base_type.variant());
    }
    r += "        ]\n";
    r += "    }\n";
    r += "}";
    r
}

pub(crate) fn render_vertex_shader(module: Module,vertex: &str) -> String {
    let renderer = Renderer { };
    let mut r = format!("pub mod {} {{\n\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n\n";
    r += &format!("        {}\n\n",renderer.module(&module));
    r += &format!("        compile_vertex_shader(module,\"{}\".to_string(),{}::get_fields())\n",vertex,vertex);
    r += "    }\n";
    r += "}\n";
    r
}

pub(crate) fn render_fragment_shader(module: Module) -> String {
    let renderer = Renderer { };
    let mut r = format!("pub mod {} {{\n\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n\n";
    r += &format!("        {}\n\n",renderer.module(&module));
    r += &format!("        compile_fragment_shader(module)\n");
    r += "    }\n";
    r += "}\n";
    r
}
