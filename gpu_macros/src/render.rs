use {
    sr::*,
};

pub trait Render {
    fn render(&self) -> String;
}

impl Render for ast::Type {
    fn render(&self) -> String {
        match self {
            ast::Type::Inferred => "ast::Type::Inferred".to_string(),
            ast::Type::Void => "ast::Type::Void".to_string(),
            ast::Type::Integer => "ast::Type::Integer".to_string(),
            ast::Type::Float => "ast::Type::Float".to_string(),
            ast::Type::Bool => "ast::Type::Bool".to_string(),
            ast::Type::U8 => "ast::Type::U8".to_string(),
            ast::Type::I8 => "ast::Type::I8".to_string(),
            ast::Type::U16 => "ast::Type::U16".to_string(),
            ast::Type::I16 => "ast::Type::I16".to_string(),
            ast::Type::U32 => "ast::Type::U32".to_string(),
            ast::Type::I32 => "ast::Type::I32".to_string(),
            ast::Type::U64 => "ast::Type::U64".to_string(),
            ast::Type::I64 => "ast::Type::I64".to_string(),
            ast::Type::USize => "ast::Type::USize".to_string(),
            ast::Type::ISize => "ast::Type::ISize".to_string(),
            ast::Type::F16 => "ast::Type::F16".to_string(),
            ast::Type::F32 => "ast::Type::F32".to_string(),
            ast::Type::F64 => "ast::Type::F64".to_string(),
            ast::Type::AnonTuple(types) => {
                let mut r = "ast::Type::AnonTuple(vec![".to_string();
                for type_ in types.iter() {
                    r += &format!("{},",type_.render());
                }
                r += "])";
                r
            },
            ast::Type::Array(type_,expr) => format!("ast::Type::Array(Box::new({}),Box::new({}))",type_.render(),expr.render()),
            ast::Type::UnknownIdent(ident) => format!("ast::Type::UnknownIdent(\"{}\".to_string())",ident),
            ast::Type::Tuple(_) => panic!("unable to render ast::Type::Tuple"),
            ast::Type::Struct(_) => panic!("unable to render ast::Type::Struct"),
            ast::Type::Enum(_) => panic!("unable to render ast::Type::Enum"),
            ast::Type::Alias(_) => panic!("unable to render ast::Type::Alias"),
        }
    }
}

impl Render for ast::Pat {
    fn render(&self) -> String {
        match self {
            ast::Pat::Wildcard => "ast::Pat::Wildcard".to_string(),
            ast::Pat::Rest => "ast::Pat::Rest".to_string(),
            ast::Pat::Boolean(value) => format!("ast::Pat::Boolean({})",if *value { "true" } else { "false" }),
            ast::Pat::Integer(value) => format!("ast::Pat::Integer({})",*value),
            ast::Pat::Float(value) => format!("ast::Pat::Float({})",*value),
            ast::Pat::AnonTuple(pats) => {
                let mut r = "ast::Pat::AnonTuple(vec![".to_string();
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += "])";
                r
            },
            ast::Pat::Array(pats) => {
                let mut r = "ast::Pat::Array(vec![".to_string();
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += "])";
                r
            },
            ast::Pat::Range(pat0,pat1) => format!("ast::Pat::Range({},{})",pat0.render(),pat1.render()),
            ast::Pat::UnknownIdent(ident) => format!("ast::Pat::UnknownIdent(\"{}\".to_string()",ident),
            ast::Pat::UnknownTuple(ident,pats) => {
                let mut r = format!("ast::Pat::UnknownTuple(\"{}\".to_string(),vec![",ident);
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += "])";
                r
            },
            ast::Pat::UnknownStruct(ident,identpats) => {
                let mut r = format!("ast::Pat::UnknownStruct(\"{}\".to_string(),vec![",ident);
                for identpat in identpats.iter() {
                    match identpat {
                        ast::UnknownFieldPat::Wildcard => r += "ast::UnknownFieldPat::Wildcard,",
                        ast::UnknownFieldPat::Rest => r += "ast::UnknownFieldPat::Rest,",
                        ast::UnknownFieldPat::Ident(ident) => r += &format!("ast::UnknownFieldPat::Ident(\"{}\".to_string()),",ident),
                        ast::UnknownFieldPat::IdentPat(ident,pat) => r += &format!("ast::UnknownFieldPat::IdentPat(\"{}\".to_string(),{}),",ident,pat.render()),
                    }
                }
                r += "])";
                r
            },
            ast::Pat::UnknownVariant(ident,variant) => {
                let mut r = format!("ast::Pat::UnknownVariant(\"{}\".to_string(),",ident);
                match variant {
                    ast::UnknownVariantPat::Naked(ident) => r += &format!("ast::UnknownVariantPat::Naked(\"{}\".to_string())",ident),
                    ast::UnknownVariantPat::Tuple(ident,pats) => {
                        r += &format!("ast::UnknownPatVariant::Tuple(\"{}\".to_string(),vec![",ident);
                        for pat in pats.iter() {
                            r += &format!("{},",pat.render());
                        }
                        r += "])";
                    },
                    ast::UnknownVariantPat::Struct(ident,identpats) => {
                        r += &format!("ast::UnknownPatVariant::Struct(\"{}\".to_string(),vec![",ident);
                        for identpat in identpats.iter() {
                            match identpat {
                                ast::UnknownFieldPat::Wildcard => r += "ast::UnknownFieldPat::Wildcard,",
                                ast::UnknownFieldPat::Rest => r += "ast::UnknownFieldPat::Rest,",
                                ast::UnknownFieldPat::Ident(ident) => r += &format!("ast::UnknownFieldPat::Ident(\"{}\".to_string()),",ident),
                                ast::UnknownFieldPat::IdentPat(ident,pat) => r += &format!("ast::UnknownFieldPat::IdentPat(\"{}\".to_string(),{}),",ident,pat.render()),
                            }
                        }
                        r += "])";
                    },
                }
                r += ")";
                r
            },
            ast::Pat::Tuple(_,_) => {
                panic!("unable to render resolved tuple pattern");
            },
            ast::Pat::Struct(_,_) => {
                panic!("unable to render resolved struct pattern");
            },
            ast::Pat::Variant(_,_) => {
                panic!("unable to render resolved variant pattern");
            },
        }
    }
}

impl Render for ast::Block {
    fn render(&self) -> String {
        let mut r = "ast::Block { stats: vec![".to_string();
        for stat in self.stats.iter() {
            r += &format!("{},",stat.render());
        }
        r += "],expr: ";
        if let Some(expr) = &self.expr {
            r += &format!("Some(Box::new({}))",expr.render());
        }
        else {
            r += "None";
        }
        r += " }";
        r
    }
}

impl Render for ast::UnaryOp {
    fn render(&self) -> String {
        match self {
            ast::UnaryOp::Neg => "ast::UnaryOp::Neg".to_string(),
            ast::UnaryOp::Not => "ast::UnaryOp::Not".to_string(),
        }
    }
}

impl Render for ast::BinaryOp {
    fn render(&self) -> String {
        match self {
            ast::BinaryOp::Mul => "ast::BinaryOp::Mul".to_string(),
            ast::BinaryOp::Div => "ast::BinaryOp::Div".to_string(),
            ast::BinaryOp::Mod => "ast::BinaryOp::Mod".to_string(),
            ast::BinaryOp::Add => "ast::BinaryOp::Add".to_string(),
            ast::BinaryOp::Sub => "ast::BinaryOp::Sub".to_string(),
            ast::BinaryOp::Shl => "ast::BinaryOp::Shl".to_string(),
            ast::BinaryOp::Shr => "ast::BinaryOp::Shr".to_string(),
            ast::BinaryOp::And => "ast::BinaryOp::And".to_string(),
            ast::BinaryOp::Or => "ast::BinaryOp::Or".to_string(),
            ast::BinaryOp::Xor => "ast::BinaryOp::Xor".to_string(),
            ast::BinaryOp::Eq => "ast::BinaryOp::Eq".to_string(),
            ast::BinaryOp::NotEq => "ast::BinaryOp::NotEq".to_string(),
            ast::BinaryOp::Greater => "ast::BinaryOp::Greater".to_string(),
            ast::BinaryOp::Less => "ast::BinaryOp::Less".to_string(),
            ast::BinaryOp::GreaterEq => "ast::BinaryOp::GreaterEq".to_string(),
            ast::BinaryOp::LessEq => "ast::BinaryOp::LessEq".to_string(),
            ast::BinaryOp::LogAnd => "ast::BinaryOp::LogAnd".to_string(),
            ast::BinaryOp::LogOr => "ast::BinaryOp::LogOr".to_string(),
            ast::BinaryOp::Assign => "ast::BinaryOp::Assign".to_string(),
            ast::BinaryOp::AddAssign => "ast::BinaryOp::AddAssign".to_string(),
            ast::BinaryOp::SubAssign => "ast::BinaryOp::SubAssign".to_string(),
            ast::BinaryOp::MulAssign => "ast::BinaryOp::MulAssign".to_string(),
            ast::BinaryOp::DivAssign => "ast::BinaryOp::DivAssign".to_string(),
            ast::BinaryOp::ModAssign => "ast::BinaryOp::ModAssign".to_string(),
            ast::BinaryOp::AndAssign => "ast::BinaryOp::AndAssign".to_string(),
            ast::BinaryOp::OrAssign => "ast::BinaryOp::OrAssign".to_string(),
            ast::BinaryOp::XorAssign => "ast::BinaryOp::XorAssign".to_string(),
            ast::BinaryOp::ShlAssign => "ast::BinaryOp::ShlAssign".to_string(),
            ast::BinaryOp::ShrAssign => "ast::BinaryOp::ShrAssign".to_string(),
        }
    }
}

impl Render for ast::Range {
    fn render(&self) -> String {
        match self {
            ast::Range::Only(expr) => format!("ast::Range::Only({})",expr.render()),
            ast::Range::FromTo(expr,expr2) => format!("ast::Range::FromTo({},{})",expr.render(),expr2.render()),
            ast::Range::FromToIncl(expr,expr2) => format!("ast::Range::FromToIncl({},{})",expr.render(),expr2.render()),
            ast::Range::From(expr) => format!("ast::Range::From({})",expr.render()),
            ast::Range::To(expr) => format!("ast::Range::To({})",expr.render()),
            ast::Range::ToIncl(expr) => format!("ast::Range::ToIncl({})",expr.render()),
            ast::Range::All => "ast::Range::All".to_string(),
        }
    }
}

impl Render for ast::Expr {
    fn render(&self) -> String {
        match self {
            ast::Expr::Boolean(value) => format!("ast::Expr::Boolean({})",if *value { "true" } else { "false" }),
            ast::Expr::Integer(value) => format!("ast::Expr::Integer({} as i64)",*value),
            ast::Expr::Float(value) => format!("ast::Expr::Float({} as f64)",*value),
            ast::Expr::Array(exprs) => {
                let mut r = "ast::Expr::Array(vec![".to_string();
                for expr in exprs.iter() {
                    r += &format!("{},",expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::Cloned(expr,expr2) => format!("ast::Expr::Cloned(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            ast::Expr::Index(expr,expr2) => format!("ast::Expr::Index(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            ast::Expr::Cast(expr,type_) => format!("ast::Expr::Cast(Box::new({}),Box::new({}))",expr.render(),type_.render()),
            ast::Expr::AnonTuple(exprs) => {
                let mut r = "ast::Expr::AnonTuple(vec![".to_string();
                for expr in exprs.iter() {
                    r += &format!("{},",expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::Unary(op,expr) => format!("ast::Expr::Unary({},Box::new({}))",op.render(),expr.render()),
            ast::Expr::Binary(expr,op,expr2) => format!("ast::Expr::Binary(Box::new({}),{},Box::new({}))",expr.render(),op.render(),expr2.render()),
            ast::Expr::Continue => "ast::Expr::Continue".to_string(),
            ast::Expr::Break(expr) => if let Some(expr) = expr {
                format!("ast::Expr::Break(Some({}))",expr.render())
            }
            else {
                "ast::Expr::Break(None)".to_string()
            },
            ast::Expr::Return(expr) => if let Some(expr) = expr {
                format!("ast::Expr::Return(Some({}))",expr.render())
            }
            else {
                "ast::Expr::Return(None)".to_string()
            },
            ast::Expr::Block(block) => format!("ast::Expr::Block({})",block.render()),
            ast::Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr {
                format!("ast::Expr::If(Box::new({}),{},Some({}))",expr.render(),block.render(),else_expr.render())
            }
            else {
                format!("ast::Expr::If(Box::new({}),{},None)",expr.render(),block.render())
            },
            ast::Expr::Loop(block) => format!("ast::Expr::Loop({})",block.render()),
            ast::Expr::While(expr,block) => format!("ast::Expr::While(Box::new({}),{})",expr.render(),block.render()),
            ast::Expr::IfLet(pats,expr,block,else_expr) => {
                let mut r = "ast::Expr::IfLet(vec![".to_string();
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += &format!("],Box::new({}),{},",expr.render(),block.render());
                if let Some(else_expr) = else_expr {
                    r += &format!("Some({})",else_expr.render());
                }
                else {
                    r += "None";
                }
                r += ")";
                r
            },
            ast::Expr::For(pats,range,block) => {
                let mut r = "ast::Expr::For(vec![".to_string();
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += &format!("],{},{})",range.render(),block.render());
                r
            },
            ast::Expr::WhileLet(pats,expr,block) => {
                let mut r = "ast::Expr::WhileLet(vec![".to_string();
                for pat in pats.iter() {
                    r += &format!("{},",pat.render());
                }
                r += &format!("],Box::new({}),{})",expr.render(),block.render());
                r
            },
            ast::Expr::Match(expr,arms) => {
                let mut r = format!("ast::Expr::Match(Box::new({}),vec![",expr.render());
                for (pats,if_expr,expr) in arms.iter() {
                    r += "(vec![";
                    for pat in pats.iter() {
                        r += &format!("{},",pat.render());
                    }
                    r += "],";
                    if let Some(if_expr) = if_expr {
                        r += &format!("Some({})",if_expr.render());
                    }
                    else {
                        r += "None";
                    }
                    r += &format!(",{}),",expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::UnknownIdent(ident) => format!("ast::Expr::UnknownIdent(\"{}\".to_string())",ident),
            ast::Expr::UnknownTupleOrCall(ident,exprs) => {
                let mut r = format!("ast::Expr::UnknownTupleOrCall(\"{}\".to_string(),vec![",ident);
                for expr in exprs.iter() {
                    r += &format!("{},",expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::UnknownStruct(ident,fields) => {
                let mut r = format!("ast::Expr::UnknownStruct(\"{}\".to_string(),vec![",ident);
                for (ident,expr) in fields.iter() {
                    r += &format!("(\"{}\".to_string(),{}),",ident,expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::UnknownVariant(ident,variant) => {
                let mut r = format!("ast::Expr::UnknownVariant(\"{}\".to_string(),",ident);
                match variant {
                    ast::UnknownVariantExpr::Naked(ident) => r += &format!("ast::UnknownVariantExpr::Naked(\"{}\".to_string())",ident),
                    ast::UnknownVariantExpr::Tuple(ident,exprs) => {
                        r += &format!("ast::UnknownVariantExpr::Tuple(\"{}\".to_string(),vec![",ident);
                        for expr in exprs.iter() {
                            r += &format!("{},",expr.render());
                        }
                        r += "])";
                    },
                    ast::UnknownVariantExpr::Struct(ident,fields) => {
                        r += &format!("ast::UnknownVariantExpr::Struct(\"{}\".to_string(),vec![",ident);
                        for (ident,expr) in fields.iter() {
                            r += &format!("(\"{}\".to_string(),{}),",ident,expr.render());
                        }
                        r += "])";
                    },
                }
                r += ")";
                r
            },
            ast::Expr::UnknownMethod(expr,ident,exprs) => {
                let mut r = format!("ast::Expr::UnknownMethod(Box::new({}),\"{}\".to_string(),vec![",expr.render(),ident);
                for expr in exprs.iter() {
                    r += &format!("{},",expr.render());
                }
                r += "])";
                r
            },
            ast::Expr::UnknownField(expr,ident) => format!("ast::Expr::UnknownField(Box::new({}),\"{}\".to_string())",expr.render(),ident),
            ast::Expr::UnknownTupleIndex(expr,index) => format!("ast::Expr::UnknowTupleIndex(Box::new({}),{})",expr.render(),index),
            ast::Expr::Param(_) |
            ast::Expr::Local(_) |
            ast::Expr::Const(_) |
            ast::Expr::Tuple(_,_) |
            ast::Expr::Call(_,_) |
            ast::Expr::Struct(_,_) |
            ast::Expr::Variant(_,_) |
            ast::Expr::Method(_,_,_) |
            ast::Expr::Field(_,_,_) |
            ast::Expr::TupleIndex(_,_,_) => {
                panic!("unable to render Expr containing Rc reference");
            }
        }
    }
}

impl Render for ast::Stat {
    fn render(&self) -> String {
        match self {
            ast::Stat::Let(pat,type_,expr) => format!("ast::Stat::Let({},{},{})",pat.render(),type_.render(),expr.render()),
            ast::Stat::Expr(expr) => format!("ast::Stat::Expr({})",expr.render()),
            ast::Stat::Local(_,_) => panic!("unable to render Stat containing Rc reference"),
        }
    }
}

impl Render for ast::Struct {
    fn render(&self) -> String {
        let mut r = format!("ast::Struct {{ ident: \"{}\".to_string(),fields: vec![",self.ident);
        for field in self.fields.iter() {
            r += &format!("ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }},",field.ident,field.type_.render());
        }
        r += "], }";
        r
    }
}

impl Render for ast::Tuple {
    fn render(&self) -> String {
        let mut r = format!("ast::Tuple {{ ident: \"{}\".to_string(),vec![",self.ident);
        for type_ in self.types.iter() {
            r += &format!("{},",type_.render());
        }
        r += "], }";
        r
    }
}

impl Render for ast::Enum {
    fn render(&self) -> String {
        let mut r = format!("ast::Enum {{ ident: \"{}\".to_string(),variants: vec![",self.ident);
        for variant in self.variants.iter() {
            match variant {
                ast::Variant::Naked(ident) => r += &format!("ast::Variant::Naked(\"{}\".to_string()),",ident),
                ast::Variant::Tuple(ident,types) => {
                    r += &format!("ast::Variant::Tuple(\"{}\".to_string(),vec![",ident);
                    for type_ in types.iter() {
                        r += &format!("{},",type_.render());
                    }
                    r += "]),";
                },
                ast::Variant::Struct(ident,fields) => {
                    r += &format!("ast::Variant::Struct(\"{}\".to_string(),vec![",ident);
                    for field in fields.iter() {
                        r += &format!("ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }},",field.ident,field.type_);
                    }
                    r += "]),";
                },
            }
        }
        r += "], }";
        r
    }
}

impl Render for ast::Alias {
    fn render(&self) -> String {
        format!("ast::Alias {{ ident: \"{}\".to_string(),type_: {}, }}",self.ident,self.type_.render())
    }
}

impl Render for ast::Const {
    fn render(&self) -> String {
        format!("ast::Const {{ ident: \"{}\".to_string(),type_: {},expr: {}, }}",self.ident,self.type_.render(),self.expr.render())
    }
}

impl Render for ast::Function {
    fn render(&self) -> String {
        let mut r = format!("ast::Function {{ ident: \"{}\".to_string(),params: vec![",self.ident);
        for param in self.params.iter() {
            r += &format!("Rc::new(RefCell::new(ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }})),",param.borrow().ident,param.borrow().type_.render());
        }
        r += &format!("],type_: {},block: {}, }}",self.type_.render(),self.block.render());
        r
    }
}

impl Render for ast::Module {
    fn render(&self) -> String {

        let mut extern_struct_idents: Vec<String> = Vec::new();
        if !self.functions.contains_key("main") {
            panic!("missing main function");
        }
        for param in self.functions["main"].borrow().params.iter() {
            if let ast::Type::UnknownIdent(ident) = param.borrow().type_.clone() {
                // bit of a hack: if type identifier contains <, it's propably a Vec or Mat, and should not be added to the struct list
                if !ident.contains("<") {
                    extern_struct_idents.push(ident);
                }
            }
        }

        let mut r = "{ use { super::*,std::{ rc::Rc,collections::HashMap,cell::RefCell, }, }; ".to_string();

        if self.tuples.len() > 0 {
            r += "let mut tuples: HashMap<String,Rc<RefCell<ast::Tuple>>> = HashMap::new(); ";
            for tuple in self.tuples.values() {
                r += &format!("tuples.insert(\"{}\".to_string(),Rc::new(RefCell::new({}))); ",tuple.borrow().ident,tuple.borrow().render());
            }
        }
        else {
            r += "let tuples: HashMap<String,Rc<RefCell<ast::Tuple>>> = HashMap::new(); ";
        }

        r += "let mut structs: HashMap<String,Rc<RefCell<ast::Struct>>> = HashMap::new(); ";
        for extern_struct_ident in extern_struct_idents.iter() {
            r += &format!("structs.insert(\"{}\".to_string(),Rc::new(RefCell::new(super::{}::ast()))); ",extern_struct_ident,extern_struct_ident);
        }
        for struct_ in self.structs.values() {
            r += &format!("structs.insert(\"{}\".to_string(),Rc::new(RefCell::new({}))); ",struct_.borrow().ident,struct_.borrow().render());
        }

        if self.enums.len() > 0 {
            r += "let mut enums: HashMap<String,Rc<RefCell<ast::Enum>>> = HashMap::new(); ";
            for enum_ in self.enums.values() {
                r += &format!("enums.insert(\"{}\".to_string(),Rc::new(RefCell::new({}))); ",enum_.borrow().ident,enum_.borrow().render());
            }
        }
        else {
            r += "let enums: HashMap<String,Rc<RefCell<ast::Enum>>> = HashMap::new(); ";
        }
        if self.aliases.len() > 0 {
            r += "let mut aliases: HashMap<String,Rc<RefCell<ast::Alias>>> = HashMap::new(); ";
            for alias in self.aliases.values() {
                r += &format!("aliases.insert(\"{}\".to_string(),Rc::new(RefCell::new({}))); ",alias.borrow().ident,alias.borrow().render());
            }
        }
        else {
            r += "let aliases: HashMap<String,Rc<RefCell<ast::Alias>>> = HashMap::new(); ";
        }
        if self.consts.len() > 0 {
            r += "let mut consts: HashMap<String,Rc<RefCell<ast::Const>>> = HashMap::new(); ";
            for const_ in self.consts.values() {
                r += &format!("consts.insert(\"{}\",Rc::new(RefCell::new({}))); ",const_.borrow().ident,const_.borrow().render());
            }
        }
        else {
            r += "let consts: HashMap<String,Rc<RefCell<ast::Const>>> = HashMap::new(); ";
        }
        if self.functions.len() > 0 {
            r += "let mut functions: HashMap<String,Rc<RefCell<ast::Function>>> = HashMap::new(); ";
            for function in self.functions.values() {
                r += &format!("functions.insert(\"{}\".to_string(),Rc::new(RefCell::new({}))); ",function.borrow().ident,function.borrow().render());
            }
        }
        else {
            r += "let functions: HashMap<String,Rc<RefCell<ast::Function>>> = HashMap::new(); ";
        }
        r += &format!("ast::Module {{ ident: \"{}\".to_string(),tuples,structs,enums,aliases,consts,functions, }} }}",self.ident);
        r
    }
}

/*
pub(crate) fn render_vertex_shader(module: Module) -> String {
    let renderer = Renderer { };
    let mut r = format!("pub mod {} {{\n\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n\n";
    r += &format!("{}\n\n",renderer.module(&module));
    if !module.functions.contains_key("main") {
        panic!("main function missing from shader module");
    }
    let main = &module.functions["main"];
    let vertex_struct_ident = if let ast::Type::UnknownIdent(ident) = &main.0[0].1 {
        ident
    }
    else {
        panic!("the first parameter of main() should be the vertex structure, and not {}",main.0[0].1);
    };
    r += &format!("        compile_vertex_shader(module,\"{}\".to_string(),{}::get_fields())\n",vertex_struct_ident,vertex_struct_ident);
    r += "    }\n";
    r += "}\n";
    r
}

pub(crate) fn render_fragment_shader(module: Module) -> String {
    let renderer = Renderer { };
    let mut r = format!("pub mod {} {{\n\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {\n\n";
    r += &format!("{}\n\n",renderer.module(&module));
    r += &format!("        compile_fragment_shader(module)\n");
    r += "    }\n";
    r += "}\n";
    r
}
*/