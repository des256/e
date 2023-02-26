use super::*;

pub trait Render {
    fn render(&self) -> String;
}

impl<T: Render> Render for Vec<T> {
    fn render(&self) -> String {
        let mut r = "vec![".to_string();
        for item in self.iter() {
            r += &format!("{},",item.render());
        }
        r += "]";
        r
    }
}

impl Render for Type {
    fn render(&self) -> String {
        match self {
            Type::Inferred => "ast::Type::Inferred".to_string(),
            Type::Void => "ast::Type::Void".to_string(),
            Type::Integer => "ast::Type::Integer".to_string(),
            Type::Float => "ast::Type::Float".to_string(),
            Type::Bool => "ast::Type::Bool".to_string(),
            Type::U8 => "ast::Type::U8".to_string(),
            Type::I8 => "ast::Type::I8".to_string(),
            Type::U16 => "ast::Type::U16".to_string(),
            Type::I16 => "ast::Type::I16".to_string(),
            Type::U32 => "ast::Type::U32".to_string(),
            Type::I32 => "ast::Type::I32".to_string(),
            Type::U64 => "ast::Type::U64".to_string(),
            Type::I64 => "ast::Type::I64".to_string(),
            Type::USize => "ast::Type::USize".to_string(),
            Type::ISize => "ast::Type::ISize".to_string(),
            Type::F16 => "ast::Type::F16".to_string(),
            Type::F32 => "ast::Type::F32".to_string(),
            Type::F64 => "ast::Type::F64".to_string(),
            Type::AnonTuple(types) => format!("ast::Type::AnonTuple({})",types.render()),
            Type::Array(type_,expr) => format!("ast::Type::Array(Box::new({}),Box::new({}))",type_.render(),expr.render()),
            Type::UnknownIdent(ident) => format!("ast::Type::UnknownIdent(\"{}\".to_string())",ident),
            Type::Generic(ident,types) => format!("ast::Type::Generic(\"{}\".to_string(),{})",ident,types.render()),
        }
    }
}

impl Render for UnknownFieldPat {
    fn render(&self) -> String {
        match self {
            UnknownFieldPat::Wildcard => "ast::FieldPat::Wildcard".to_string(),
            UnknownFieldPat::Rest => "ast::FieldPat::Rest".to_string(),
            UnknownFieldPat::Ident(ident) => format!("ast::FieldPat::Ident(\"{}\".to_string())",ident),
            UnknownFieldPat::IdentPat(ident,pat) => format!("ast::FieldPat::IdentPat(\"{}\".to_string(),{})",ident,pat.render()),
        }
    }
}

impl Render for UnknownVariantPat {
    fn render(&self) -> String {
        match self {
            UnknownVariantPat::Naked(ident) => format!("ast::VariantPat::Naked(\"{}\".to_string())",ident),
            UnknownVariantPat::Tuple(ident,pats) => format!("ast::VariantPat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            UnknownVariantPat::Struct(ident,identpats) => format!("ast::VariantPat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
        }
    }
}

impl Render for Pat {
    fn render(&self) -> String {
        match self {
            Pat::Wildcard => "ast::Pat::Wildcard".to_string(),
            Pat::Rest => "ast::Pat::Rest".to_string(),
            Pat::Boolean(value) => format!("ast::Pat::Boolean({})",if *value { "true" } else { "false" }),
            Pat::Integer(value) => format!("ast::Pat::Integer({})",*value),
            Pat::Float(value) => format!("ast::Pat::Float({})",*value),
            Pat::AnonTuple(pats) => format!("ast::Pat::AnonTuple({})",pats.render()),
            Pat::Array(pats) => format!("ast::Pat::Array({})",pats.render()),
            Pat::Range(pat0,pat1) => format!("ast::Pat::Range({},{})",pat0.render(),pat1.render()),
            Pat::UnknownIdent(ident) => format!("ast::Pat::UnknownIdent(\"{}\".to_string())",ident),
            Pat::UnknownTuple(ident,pats) => format!("ast::Pat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            Pat::UnknownStruct(ident,identpats) => format!("ast::Pat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
            Pat::UnknownVariant(ident,variant) => format!("ast::Pat::Variant(\"{}\".to_string(),{})",ident,variant.render()),
        }
    }
}

impl Render for Block {
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

impl Render for UnaryOp {
    fn render(&self) -> String {
        match self {
            UnaryOp::Neg => "ast::UnaryOp::Neg".to_string(),
            UnaryOp::Not => "ast::UnaryOp::Not".to_string(),
        }
    }
}

impl Render for BinaryOp {
    fn render(&self) -> String {
        match self {
            BinaryOp::Mul => "ast::BinaryOp::Mul".to_string(),
            BinaryOp::Div => "ast::BinaryOp::Div".to_string(),
            BinaryOp::Mod => "ast::BinaryOp::Mod".to_string(),
            BinaryOp::Add => "ast::BinaryOp::Add".to_string(),
            BinaryOp::Sub => "ast::BinaryOp::Sub".to_string(),
            BinaryOp::Shl => "ast::BinaryOp::Shl".to_string(),
            BinaryOp::Shr => "ast::BinaryOp::Shr".to_string(),
            BinaryOp::And => "ast::BinaryOp::And".to_string(),
            BinaryOp::Or => "ast::BinaryOp::Or".to_string(),
            BinaryOp::Xor => "ast::BinaryOp::Xor".to_string(),
            BinaryOp::Eq => "ast::BinaryOp::Eq".to_string(),
            BinaryOp::NotEq => "ast::BinaryOp::NotEq".to_string(),
            BinaryOp::Greater => "ast::BinaryOp::Greater".to_string(),
            BinaryOp::Less => "ast::BinaryOp::Less".to_string(),
            BinaryOp::GreaterEq => "ast::BinaryOp::GreaterEq".to_string(),
            BinaryOp::LessEq => "ast::BinaryOp::LessEq".to_string(),
            BinaryOp::LogAnd => "ast::BinaryOp::LogAnd".to_string(),
            BinaryOp::LogOr => "ast::BinaryOp::LogOr".to_string(),
            BinaryOp::Assign => "ast::BinaryOp::Assign".to_string(),
            BinaryOp::AddAssign => "ast::BinaryOp::AddAssign".to_string(),
            BinaryOp::SubAssign => "ast::BinaryOp::SubAssign".to_string(),
            BinaryOp::MulAssign => "ast::BinaryOp::MulAssign".to_string(),
            BinaryOp::DivAssign => "ast::BinaryOp::DivAssign".to_string(),
            BinaryOp::ModAssign => "ast::BinaryOp::ModAssign".to_string(),
            BinaryOp::AndAssign => "ast::BinaryOp::AndAssign".to_string(),
            BinaryOp::OrAssign => "ast::BinaryOp::OrAssign".to_string(),
            BinaryOp::XorAssign => "ast::BinaryOp::XorAssign".to_string(),
            BinaryOp::ShlAssign => "ast::BinaryOp::ShlAssign".to_string(),
            BinaryOp::ShrAssign => "ast::BinaryOp::ShrAssign".to_string(),
        }
    }
}

impl Render for Range {
    fn render(&self) -> String {
        match self {
            Range::Only(expr) => format!("ast::Range::Only({})",expr.render()),
            Range::FromTo(expr,expr2) => format!("ast::Range::FromTo({},{})",expr.render(),expr2.render()),
            Range::FromToIncl(expr,expr2) => format!("ast::Range::FromToIncl({},{})",expr.render(),expr2.render()),
            Range::From(expr) => format!("ast::Range::From({})",expr.render()),
            Range::To(expr) => format!("ast::Range::To({})",expr.render()),
            Range::ToIncl(expr) => format!("ast::Range::ToIncl({})",expr.render()),
            Range::All => "ast::Range::All".to_string(),
        }
    }
}

impl<T: Render> Render for Option<Box<T>> {
    fn render(&self) -> String {
        if let Some(item) = self {
            format!("Some({})",item.render())
        }
        else {
            format!("None")
        }
    }
}

impl Render for (String,Expr) {
    fn render(&self) -> String {
        format!("(\"{}\".to_string(),{})",self.0,self.1.render())
    }
}

impl Render for (String,Type) {
    fn render(&self) -> String {
        format!("ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }}",self.0,self.1.render())
    }
}

impl Render for UnknownVariantExpr {
    fn render(&self) -> String {
        match self {
            UnknownVariantExpr::Naked(ident) => format!("ast::VariantExpr::Naked(\"{}\".to_string())",ident),
            UnknownVariantExpr::Tuple(ident,exprs) => format!("ast::VariantExpr::Tuple(\"{}\".to_string(),{})",ident,exprs.render()),
            UnknownVariantExpr::Struct(ident,fields) => format!("ast::VariantExpr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
        }
    }
}

impl Render for (Vec<Pat>,Option<Box<Expr>>,Box<Expr>) {
    fn render(&self) -> String {
        format!("({},{},Box::new({}))",self.0.render(),self.1.render(),self.2.render())
    }
}

impl Render for Expr {
    fn render(&self) -> String {
        match self {
            Expr::Boolean(value) => format!("ast::Expr::Boolean({})",if *value { "true" } else { "false" }),
            Expr::Integer(value) => format!("ast::Expr::Integer({} as i64)",*value),
            Expr::Float(value) => format!("ast::Expr::Float({} as f64)",*value),
            Expr::Array(exprs) => format!("ast::Expr::Array({})",exprs.render()),
            Expr::Cloned(expr,expr2) => format!("ast::Expr::Cloned(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Index(expr,expr2) => format!("ast::Expr::Index(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Cast(expr,type_) => format!("ast::Expr::Cast(Box::new({}),Box::new({}))",expr.render(),type_.render()),
            Expr::AnonTuple(exprs) => format!("ast::Expr::AnonTuple({})",exprs.render()),
            Expr::Unary(op,expr) => format!("ast::Expr::Unary({},Box::new({}))",op.render(),expr.render()),
            Expr::Binary(expr,op,expr2) => format!("ast::Expr::Binary(Box::new({}),{},Box::new({}))",expr.render(),op.render(),expr2.render()),
            Expr::Continue => "ast::Expr::Continue".to_string(),
            Expr::Break(expr) => format!("ast::Expr::Break({})",expr.render()),
            Expr::Return(expr) => format!("ast::Expr::Return({})",expr.render()),
            Expr::Block(block) => format!("ast::Expr::Block({})",block.render()),
            Expr::If(expr,block,else_expr) => format!("ast::Expr::If(Box::new({}),{},{})",expr.render(),block.render(),else_expr.render()),
            Expr::Loop(block) => format!("ast::Expr::Loop({})",block.render()),
            Expr::While(expr,block) => format!("ast::Expr::While(Box::new({}),{})",expr.render(),block.render()),
            Expr::IfLet(pats,expr,block,else_expr) => format!("ast::Expr::IfLet({},Box::new({}),{},{})",pats.render(),expr.render(),block.render(),else_expr.render()),
            Expr::For(pats,range,block) => format!("ast::Expr::For({},{},{})",pats.render(),range.render(),block.render()),
            Expr::WhileLet(pats,expr,block) => format!("ast::Expr::WhileLet({},Box::new({}),{})",pats.render(),expr.render(),block.render()),
            Expr::Match(expr,arms) => format!("ast::Expr::Match(Box::new({}),{})",expr.render(),arms.render()),
            Expr::UnknownIdent(ident) => format!("ast::Expr::UnknownIdent(\"{}\".to_string())",ident),
            Expr::UnknownTupleOrCall(ident,exprs) => format!("ast::Expr::TupleOrCall(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::UnknownStruct(ident,fields) => format!("ast::Expr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
            Expr::UnknownVariant(ident,variant) => format!("ast::Expr::Variant(\"{}\".to_string(),{})",ident,variant.render()),
            Expr::UnknownMethod(expr,ident,exprs) => format!("ast::Expr::Method(Box::new({}),\"{}\".to_string(),{})",expr.render(),ident,exprs.render()),
            Expr::UnknownField(expr,ident) => format!("ast::Expr::Field(Box::new({}),\"{}\".to_string())",expr.render(),ident),
            Expr::UnknownTupleIndex(expr,index) => format!("ast::Expr::TupleIndex(Box::new({}),{})",expr.render(),index),
        }
    }
}

impl Render for Stat {
    fn render(&self) -> String {
        match self {
            Stat::Let(pat,type_,expr) => format!("ast::Stat::Let(Box::new({}),Box::new({}),Box::new({}))",pat.render(),type_.render(),expr.render()),
            Stat::Expr(expr) => format!("ast::Stat::Expr(Box::new({}))",expr.render()),
        }
    }
}

impl Render for Struct {
    fn render(&self) -> String {
        let mut r = format!("ast::Struct {{ ident: \"{}\".to_string(),fields: vec![",self.ident);
        for (ident,type_) in self.fields.iter() {
            r += &format!("(\"{}\".to_string(),{}),",ident,type_.render());
        }
        r += "], }";
        r
    }
}

impl Render for Tuple {
    fn render(&self) -> String {
        format!("ast::Tuple {{ ident: \"{}\".to_string(),types: {}, }}",self.ident,self.types.render())
    }
}

impl Render for Variant {
    fn render(&self) -> String {
        match self {
            Variant::Naked(ident) => format!("ast::Variant::Naked(\"{}\".to_string())",ident),
            Variant::Tuple(ident,types) => format!("ast::Variant::Tuple(\"{}\".to_string(),{})",ident,types.render()),
            Variant::Struct(ident,fields) => format!("ast::Variant::Struct(\"{}\".to_string(),{})",ident,fields.render()),
        }
    }
}

impl Render for Enum {
    fn render(&self) -> String {
        format!("ast::Enum {{ ident: \"{}\".to_string(),variants: {}, }}",self.ident,self.variants.render())
    }
}

impl Render for Alias {
    fn render(&self) -> String {
        format!("ast::Alias {{ ident: \"{}\".to_string(),type_: {}, }}",self.ident,self.type_.render())
    }
}

impl Render for Const {
    fn render(&self) -> String {
        format!("ast::Const {{ ident: \"{}\".to_string(),type_: {},expr: {}, }}",self.ident,self.type_.render(),self.expr.render())
    }
}

impl Render for Function {
    fn render(&self) -> String {
        format!("ast::Function {{ ident: \"{}\".to_string(),params: {},type_: {},block: {}, }}",self.ident,self.params.render(),self.type_.render(),self.block.render())
    }
}

impl Render for Module {
    fn render(&self) -> String {

        let mut extern_struct_idents: Vec<String> = Vec::new();

        let mut main_found = false;
        for function in self.functions.iter() {
            if function.ident == "main" {
                for (_,type_) in function.params.iter() {
                    if let Type::UnknownIdent(ident) = type_ {
                        extern_struct_idents.push(ident.clone());
                    }
                }
                main_found = true;
                break;
            }
        }
        if !main_found {
            panic!("missing main function");
        }

        let mut r = "{ use { super::*,std::collections::HashMap, }; ".to_string();

        if self.tuples.len() > 0 {
            r += &format!("let mut tuples: HashMap<String,ast::Tuple> = HashMap::new(); ");
            for tuple in self.tuples.iter() {
                r += &format!("tuples.insert(\"{}\".to_string(),{}); ",tuple.ident,tuple.render());
            }
        }
        else {
            r += "let tuples: HashMap<String,ast::Tuple> = HashMap::new(); ";
        }

        if extern_struct_idents.len() > 0 {
            r += "let mut extern_structs: HashMap<String,ast::Struct> = HashMap::new();";
            for extern_struct_ident in extern_struct_idents.iter() {
                // hack: if struct contains <>, it's not an external struct, but something defined in the standard library
                // there are probably other structs as well, but for now let's assume they're not defined in the standard library
                if !extern_struct_ident.contains('<') {
                    r += &format!("extern_structs.insert(\"{}\".to_string(),super::{}::ast());",extern_struct_ident,extern_struct_ident);
                }
            }
        }
        else {
            r += "let extern_structs: HashMap<String,ast::Struct> = HashMap::new(); ";
        }

        if self.structs.len() > 0 {
            r += &format!("let mut structs: HashMap<String,ast::Struct> = HashMap::new();");
            for struct_ in self.structs.iter() {
                r += &format!("structs.insert(\"{}\".to_string(),{}); ",struct_.ident,struct_.render());
            }
        }
        else {
            r += "let mut structs: HashMap<String,ast::Struct> = HashMap::new(); ";
        }

        if self.enums.len() > 0 {
            r += &format!("let mut enums: HashMap<String,ast::Enum> = HashMap::new(); ");
            for enum_ in self.enums.iter() {
                r += &format!("enums.insert(\"{}\".to_string(),{}); ",enum_.ident,enum_.render());
            }
        }
        else {
            r += "let enums: HashMap<String,ast::Enum> = HashMap::new(); ";
        }
        if self.aliases.len() > 0 {
            r += &format!("let mut aliases: HashMap<String,ast::Alias> = HashMap::new(); ");
            for alias in self.aliases.iter() {
                r += &format!("aliases.insert(\"{}\".to_string(),{}); ",alias.ident,alias.render());
            }
        }
        else {
            r += "let aliases: HashMap<String,ast::Alias> = HashMap::new(); ";
        }
        if self.consts.len() > 0 {
            r += &format!("let mut consts: HashMap<String,ast::Const> = HashMap::new(); ");
            for const_ in self.consts.iter() {
                r += &format!("consts.insert(\"{}\".to_string(),{}); ",const_.ident,const_.render());
            }
        }
        else {
            r += "let consts: HashMap<String,ast::Const> = HashMap::new(); ";
        }
        if self.functions.len() > 0 {
            r += &format!("let mut functions: HashMap<String,ast::Function> = HashMap::new(); ");
            for function in self.functions.iter() {
                r += &format!("functions.insert(\"{}\".to_string(),{}); ",function.ident,function.render());
            }
        }
        else {
            r += "let functions: HashMap<String,ast::Function> = Vec::new(); ";
        }
        r += &format!("ast::RustModule {{ ident: \"{}\".to_string(),tuples,structs,extern_structs,enums,aliases,consts,functions, }} }}",self.ident);
        r
    }
}
