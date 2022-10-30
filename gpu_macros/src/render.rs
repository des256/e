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
            Type::Inferred => "e::ast::Type::Inferred".to_string(),
            Type::Void => "e::ast::Type::Void".to_string(),
            Type::Integer => "e::ast::Type::Integer".to_string(),
            Type::Float => "e::ast::Type::Float".to_string(),
            Type::Bool => "e::ast::Type::Bool".to_string(),
            Type::U8 => "e::ast::Type::U8".to_string(),
            Type::I8 => "e::ast::Type::I8".to_string(),
            Type::U16 => "e::ast::Type::U16".to_string(),
            Type::I16 => "e::ast::Type::I16".to_string(),
            Type::U32 => "e::ast::Type::U32".to_string(),
            Type::I32 => "e::ast::Type::I32".to_string(),
            Type::U64 => "e::ast::Type::U64".to_string(),
            Type::I64 => "e::ast::Type::I64".to_string(),
            Type::USize => "e::ast::Type::USize".to_string(),
            Type::ISize => "e::ast::Type::ISize".to_string(),
            Type::F16 => "e::ast::Type::F16".to_string(),
            Type::F32 => "e::ast::Type::F32".to_string(),
            Type::F64 => "e::ast::Type::F64".to_string(),
            Type::AnonTuple(types) => format!("e::ast::Type::AnonTuple({})",types.render()),
            Type::Array(type_,expr) => format!("e::ast::Type::Array(Box::new({}),Box::new({}))",type_.render(),expr.render()),
            Type::UnknownIdent(ident) => format!("e::ast::Type::UnknownIdent(\"{}\".to_string())",ident),
        }
    }
}

impl Render for UnknownFieldPat {
    fn render(&self) -> String {
        match self {
            UnknownFieldPat::Wildcard => "e::ast::FieldPat::Wildcard".to_string(),
            UnknownFieldPat::Rest => "e::ast::FieldPat::Rest".to_string(),
            UnknownFieldPat::Ident(ident) => format!("e::ast::FieldPat::Ident(\"{}\".to_string())",ident),
            UnknownFieldPat::IdentPat(ident,pat) => format!("e::ast::FieldPat::IdentPat(\"{}\".to_string(),{})",ident,pat.render()),
        }
    }
}

impl Render for UnknownVariantPat {
    fn render(&self) -> String {
        match self {
            UnknownVariantPat::Naked(ident) => format!("e::ast::VariantPat::Naked(\"{}\".to_string())",ident),
            UnknownVariantPat::Tuple(ident,pats) => format!("e::ast::VariantPat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            UnknownVariantPat::Struct(ident,identpats) => format!("e::ast::VariantPat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
        }
    }
}

impl Render for Pat {
    fn render(&self) -> String {
        match self {
            Pat::Wildcard => "e::ast::Pat::Wildcard".to_string(),
            Pat::Rest => "e::ast::Pat::Rest".to_string(),
            Pat::Boolean(value) => format!("e::ast::Pat::Boolean({})",if *value { "true" } else { "false" }),
            Pat::Integer(value) => format!("e::ast::Pat::Integer({})",*value),
            Pat::Float(value) => format!("e::ast::Pat::Float({})",*value),
            Pat::AnonTuple(pats) => format!("e::ast::Pat::AnonTuple({})",pats.render()),
            Pat::Array(pats) => format!("e::ast::Pat::Array({})",pats.render()),
            Pat::Range(pat0,pat1) => format!("e::ast::Pat::Range({},{})",pat0.render(),pat1.render()),
            Pat::UnknownIdent(ident) => format!("e::ast::Pat::UnknownIdent(\"{}\".to_string())",ident),
            Pat::UnknownTuple(ident,pats) => format!("e::ast::Pat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            Pat::UnknownStruct(ident,identpats) => format!("e::ast::Pat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
            Pat::UnknownVariant(ident,variant) => format!("e::ast::Pat::Variant(\"{}\".to_string(),{})",ident,variant.render()),
        }
    }
}

impl Render for Block {
    fn render(&self) -> String {
        let mut r = "e::ast::Block { stats: vec![".to_string();
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
            UnaryOp::Neg => "e::ast::UnaryOp::Neg".to_string(),
            UnaryOp::Not => "e::ast::UnaryOp::Not".to_string(),
        }
    }
}

impl Render for BinaryOp {
    fn render(&self) -> String {
        match self {
            BinaryOp::Mul => "e::ast::BinaryOp::Mul".to_string(),
            BinaryOp::Div => "e::ast::BinaryOp::Div".to_string(),
            BinaryOp::Mod => "e::ast::BinaryOp::Mod".to_string(),
            BinaryOp::Add => "e::ast::BinaryOp::Add".to_string(),
            BinaryOp::Sub => "e::ast::BinaryOp::Sub".to_string(),
            BinaryOp::Shl => "e::ast::BinaryOp::Shl".to_string(),
            BinaryOp::Shr => "e::ast::BinaryOp::Shr".to_string(),
            BinaryOp::And => "e::ast::BinaryOp::And".to_string(),
            BinaryOp::Or => "e::ast::BinaryOp::Or".to_string(),
            BinaryOp::Xor => "e::ast::BinaryOp::Xor".to_string(),
            BinaryOp::Eq => "e::ast::BinaryOp::Eq".to_string(),
            BinaryOp::NotEq => "e::ast::BinaryOp::NotEq".to_string(),
            BinaryOp::Greater => "e::ast::BinaryOp::Greater".to_string(),
            BinaryOp::Less => "e::ast::BinaryOp::Less".to_string(),
            BinaryOp::GreaterEq => "e::ast::BinaryOp::GreaterEq".to_string(),
            BinaryOp::LessEq => "e::ast::BinaryOp::LessEq".to_string(),
            BinaryOp::LogAnd => "e::ast::BinaryOp::LogAnd".to_string(),
            BinaryOp::LogOr => "e::ast::BinaryOp::LogOr".to_string(),
            BinaryOp::Assign => "e::ast::BinaryOp::Assign".to_string(),
            BinaryOp::AddAssign => "e::ast::BinaryOp::AddAssign".to_string(),
            BinaryOp::SubAssign => "e::ast::BinaryOp::SubAssign".to_string(),
            BinaryOp::MulAssign => "e::ast::BinaryOp::MulAssign".to_string(),
            BinaryOp::DivAssign => "e::ast::BinaryOp::DivAssign".to_string(),
            BinaryOp::ModAssign => "e::ast::BinaryOp::ModAssign".to_string(),
            BinaryOp::AndAssign => "e::ast::BinaryOp::AndAssign".to_string(),
            BinaryOp::OrAssign => "e::ast::BinaryOp::OrAssign".to_string(),
            BinaryOp::XorAssign => "e::ast::BinaryOp::XorAssign".to_string(),
            BinaryOp::ShlAssign => "e::ast::BinaryOp::ShlAssign".to_string(),
            BinaryOp::ShrAssign => "e::ast::BinaryOp::ShrAssign".to_string(),
        }
    }
}

impl Render for Range {
    fn render(&self) -> String {
        match self {
            Range::Only(expr) => format!("e::ast::Range::Only({})",expr.render()),
            Range::FromTo(expr,expr2) => format!("e::ast::Range::FromTo({},{})",expr.render(),expr2.render()),
            Range::FromToIncl(expr,expr2) => format!("e::ast::Range::FromToIncl({},{})",expr.render(),expr2.render()),
            Range::From(expr) => format!("e::ast::Range::From({})",expr.render()),
            Range::To(expr) => format!("e::ast::Range::To({})",expr.render()),
            Range::ToIncl(expr) => format!("e::ast::Range::ToIncl({})",expr.render()),
            Range::All => "e::ast::Range::All".to_string(),
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
        format!("e::ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }}",self.0,self.1.render())
    }
}

impl Render for UnknownVariantExpr {
    fn render(&self) -> String {
        match self {
            UnknownVariantExpr::Naked(ident) => format!("e::ast::VariantExpr::Naked(\"{}\".to_string())",ident),
            UnknownVariantExpr::Tuple(ident,exprs) => format!("e::ast::VariantExpr::Tuple(\"{}\".to_string(),{})",ident,exprs.render()),
            UnknownVariantExpr::Struct(ident,fields) => format!("e::ast::VariantExpr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
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
            Expr::Boolean(value) => format!("e::ast::Expr::Boolean({})",if *value { "true" } else { "false" }),
            Expr::Integer(value) => format!("e::ast::Expr::Integer({} as i64)",*value),
            Expr::Float(value) => format!("e::ast::Expr::Float({} as f64)",*value),
            Expr::Array(exprs) => format!("e::ast::Expr::Array({})",exprs.render()),
            Expr::Cloned(expr,expr2) => format!("e::ast::Expr::Cloned(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Index(expr,expr2) => format!("e::ast::Expr::Index(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Cast(expr,type_) => format!("e::ast::Expr::Cast(Box::new({}),Box::new({}))",expr.render(),type_.render()),
            Expr::AnonTuple(exprs) => format!("e::ast::Expr::AnonTuple({})",exprs.render()),
            Expr::Unary(op,expr) => format!("e::ast::Expr::Unary({},Box::new({}))",op.render(),expr.render()),
            Expr::Binary(expr,op,expr2) => format!("e::ast::Expr::Binary(Box::new({}),{},Box::new({}))",expr.render(),op.render(),expr2.render()),
            Expr::Continue => "e::ast::Expr::Continue".to_string(),
            Expr::Break(expr) => format!("e::ast::Expr::Break({})",expr.render()),
            Expr::Return(expr) => format!("e::ast::Expr::Return({})",expr.render()),
            Expr::Block(block) => format!("e::ast::Expr::Block({})",block.render()),
            Expr::If(expr,block,else_expr) => format!("e::ast::Expr::If(Box::new({}),{},{})",expr.render(),block.render(),else_expr.render()),
            Expr::Loop(block) => format!("e::ast::Expr::Loop({})",block.render()),
            Expr::While(expr,block) => format!("e::ast::Expr::While(Box::new({}),{})",expr.render(),block.render()),
            Expr::IfLet(pats,expr,block,else_expr) => format!("e::ast::Expr::IfLet({},Box::new({}),{},{})",pats.render(),expr.render(),block.render(),else_expr.render()),
            Expr::For(pats,range,block) => format!("e::ast::Expr::For({},{},{})",pats.render(),range.render(),block.render()),
            Expr::WhileLet(pats,expr,block) => format!("e::ast::Expr::WhileLet({},Box::new({}),{})",pats.render(),expr.render(),block.render()),
            Expr::Match(expr,arms) => format!("e::ast::Expr::Match(Box::new({}),{})",expr.render(),arms.render()),
            Expr::UnknownIdent(ident) => format!("e::ast::Expr::UnknownIdent(\"{}\".to_string())",ident),
            Expr::UnknownTupleOrCall(ident,exprs) => format!("e::ast::Expr::TupleOrCall(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::UnknownStruct(ident,fields) => format!("e::ast::Expr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
            Expr::UnknownVariant(ident,variant) => format!("e::ast::Expr::Variant(\"{}\".to_string(),{})",ident,variant.render()),
            Expr::UnknownMethod(expr,ident,exprs) => format!("e::ast::Expr::Method(Box::new({}),\"{}\".to_string(),{})",expr.render(),ident,exprs.render()),
            Expr::UnknownField(expr,ident) => format!("e::ast::Expr::Field(Box::new({}),\"{}\".to_string())",expr.render(),ident),
            Expr::UnknownTupleIndex(expr,index) => format!("e::ast::Expr::TupleIndex(Box::new({}),{})",expr.render(),index),
        }
    }
}

impl Render for Stat {
    fn render(&self) -> String {
        match self {
            Stat::Let(pat,type_,expr) => format!("e::ast::Stat::Let(Box::new({}),Box::new({}),Box::new({}))",pat.render(),type_.render(),expr.render()),
            Stat::Expr(expr) => format!("e::ast::Stat::Expr(Box::new({}))",expr.render()),
        }
    }
}

impl Render for Struct {
    fn render(&self) -> String {
        let mut r = format!("e::ast::Struct {{ ident: \"{}\".to_string(),fields: vec![",self.ident);
        for (ident,type_) in self.fields.iter() {
            r += &format!("e::ast::Symbol {{ ident: \"{}\".to_string(),type_: {}, }},",ident,type_.render());
        }
        r += "], }";
        r
    }
}

impl Render for Tuple {
    fn render(&self) -> String {
        format!("e::ast::Tuple {{ ident: \"{}\".to_string(),types: {}, }}",self.ident,self.types.render())
    }
}

impl Render for Variant {
    fn render(&self) -> String {
        match self {
            Variant::Naked(ident) => format!("e::ast::Variant::Naked(\"{}\".to_string())",ident),
            Variant::Tuple(ident,types) => format!("e::ast::Variant::Tuple(\"{}\".to_string(),{})",ident,types.render()),
            Variant::Struct(ident,fields) => format!("e::ast::Variant::Struct(\"{}\".to_string(),{})",ident,fields.render()),
        }
    }
}

impl Render for Enum {
    fn render(&self) -> String {
        format!("e::ast::Enum {{ ident: \"{}\".to_string(),variants: {}, }}",self.ident,self.variants.render())
    }
}

impl Render for Alias {
    fn render(&self) -> String {
        format!("e::ast::Alias {{ ident: \"{}\".to_string(),type_: {}, }}",self.ident,self.type_.render())
    }
}

impl Render for Const {
    fn render(&self) -> String {
        format!("e::ast::Const {{ ident: \"{}\".to_string(),type_: {},expr: {}, }}",self.ident,self.type_.render(),self.expr.render())
    }
}

impl Render for Function {
    fn render(&self) -> String {
        format!("e::ast::Function {{ ident: \"{}\".to_string(),params: {},type_: {},block: {}, }}",self.ident,self.params.render(),self.type_.render(),self.block.render())
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
            r += &format!("let mut tuples: HashMap<String,e::ast::Tuple> = HashMap::new(); ");
            for tuple in self.tuples.iter() {
                r += &format!("tuples.insert(\"{}\".to_string(),{}); ",tuple.ident,tuple.render());
            }
        }
        else {
            r += "let tuples: HashMap<String,e::ast::Tuple> = HashMap::new(); ";
        }

        if extern_struct_idents.len() > 0 {
            r += "let mut extern_structs: HashMap<String,e::ast::Struct> = HashMap::new();";
            for extern_struct_ident in extern_struct_idents.iter() {
                // hack: if struct contains <>, it's not an external struct, but something defined in the standard library
                // there are probably other structs as well, but for now let's assume they're not defined in the standard library
                if !extern_struct_ident.contains('<') {
                    r += &format!("extern_structs.insert(\"{}\".to_string(),super::{}::ast());",extern_struct_ident,extern_struct_ident);
                }
            }
        }
        else {
            r += "let extern_structs: HashMap<String,e::ast::Struct> = HashMap::new(); ";
        }

        if self.structs.len() > 0 {
            r += &format!("let mut structs: HashMap<String,e::ast::Struct> = HashMap::new();");
            for struct_ in self.structs.iter() {
                r += &format!("structs.insert(\"{}\".to_string(),{}); ",struct_.ident,struct_.render());
            }
        }
        else {
            r += "let mut structs: HashMap<String,e::ast::Struct> = HashMap::new(); ";
        }

        if self.enums.len() > 0 {
            r += &format!("let mut enums: HashMap<String,e::ast::Enum> = HashMap::new(); ");
            for enum_ in self.enums.iter() {
                r += &format!("enums.insert(\"{}\".to_string(),{}); ",enum_.ident,enum_.render());
            }
        }
        else {
            r += "let enums: HashMap<String,e::ast::Enum> = HashMap::new(); ";
        }
        if self.aliases.len() > 0 {
            r += &format!("let mut aliases: HashMap<String,e::ast::Alias> = HashMap::new(); ");
            for alias in self.aliases.iter() {
                r += &format!("aliases.insert(\"{}\".to_string(),{}); ",alias.ident,alias.render());
            }
        }
        else {
            r += "let aliases: HashMap<String,e::ast::Alias> = HashMap::new(); ";
        }
        if self.consts.len() > 0 {
            r += &format!("let mut consts: HashMap<String,e::ast::Const> = HashMap::new(); ");
            for const_ in self.consts.iter() {
                r += &format!("consts.insert(\"{}\".to_string(),{}); ",const_.ident,const_.render());
            }
        }
        else {
            r += "let consts: HashMap<String,e::ast::Const> = HashMap::new(); ";
        }
        if self.functions.len() > 0 {
            r += &format!("let mut functions: HashMap<String,e::ast::Function> = HashMap::new(); ");
            for function in self.functions.iter() {
                r += &format!("functions.insert(\"{}\".to_string(),{}); ",function.ident,function.render());
            }
        }
        else {
            r += "let functions: HashMap<String,e::ast::Function> = Vec::new(); ";
        }
        r += &format!("e::ast::RustModule {{ ident: \"{}\".to_string(),tuples,structs,extern_structs,enums,aliases,consts,functions, }} }}",self.ident);
        r
    }
}
