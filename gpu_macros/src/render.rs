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
            Type::Inferred => "Type::Inferred".to_string(),
            Type::Void => "Type::Void".to_string(),
            Type::Integer => "Type::Integer".to_string(),
            Type::Float => "Type::Float".to_string(),
            Type::Bool => "Type::Bool".to_string(),
            Type::U8 => "Type::U8".to_string(),
            Type::I8 => "Type::I8".to_string(),
            Type::U16 => "Type::U16".to_string(),
            Type::I16 => "Type::I16".to_string(),
            Type::U32 => "Type::U32".to_string(),
            Type::I32 => "Type::I32".to_string(),
            Type::U64 => "Type::U64".to_string(),
            Type::I64 => "Type::I64".to_string(),
            Type::USize => "Type::USize".to_string(),
            Type::ISize => "Type::ISize".to_string(),
            Type::F16 => "Type::F16".to_string(),
            Type::F32 => "Type::F32".to_string(),
            Type::F64 => "Type::F64".to_string(),
            Type::AnonTuple(types) => format!("Type::AnonTuple({})",types.render()),
            Type::Array(type_,expr) => format!("Type::Array(Box::new({}),Box::new({}))",type_.render(),expr.render()),
            Type::UnknownIdent(ident) => format!("Type::UnknownIdent(\"{}\".to_string())",ident),
        }
    }
}

impl Render for UnknownFieldPat {
    fn render(&self) -> String {
        match self {
            UnknownFieldPat::Wildcard => "UnknownFieldPat::Wildcard".to_string(),
            UnknownFieldPat::Rest => "UnknownFieldPat::Rest".to_string(),
            UnknownFieldPat::Ident(ident) => format!("UnknownFieldPat::Ident(\"{}\".to_string())",ident),
            UnknownFieldPat::IdentPat(ident,pat) => format!("UnknownFieldPat::IdentPat(\"{}\".to_string(),{})",ident,pat.render()),
        }
    }
}

impl Render for UnknownVariantPat {
    fn render(&self) -> String {
        match self {
            UnknownVariantPat::Naked(ident) => format!("UnknownVariantPat::Naked(\"{}\".to_string())",ident),
            UnknownVariantPat::Tuple(ident,pats) => format!("UnknownVariantPat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            UnknownVariantPat::Struct(ident,identpats) => format!("UnknownVariantPat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
        }
    }
}

impl Render for Pat {
    fn render(&self) -> String {
        match self {
            Pat::Wildcard => "Pat::Wildcard".to_string(),
            Pat::Rest => "Pat::Rest".to_string(),
            Pat::Boolean(value) => format!("Pat::Boolean({})",if *value { "true" } else { "false" }),
            Pat::Integer(value) => format!("Pat::Integer({})",*value),
            Pat::Float(value) => format!("Pat::Float({})",*value),
            Pat::AnonTuple(pats) => format!("Pat::AnonTuple({})",pats.render()),
            Pat::Array(pats) => format!("Pat::Array({})",pats.render()),
            Pat::Range(pat0,pat1) => format!("Pat::Range({},{})",pat0.render(),pat1.render()),
            Pat::UnknownIdent(ident) => format!("Pat::UnknownIdent(\"{}\".to_string())",ident),
            Pat::UnknownTuple(ident,pats) => format!("Pat::UnknownTuple(\"{}\".to_string(),{})",ident,pats.render()),
            Pat::UnknownStruct(ident,identpats) => format!("Pat::UnknownStruct(\"{}\".to_string(),{})",ident,identpats.render()),
            Pat::UnknownVariant(ident,variant) => format!("Pat::UnknownVariant(\"{}\",to_string(),{})",ident,variant.render()),
        }
    }
}

impl Render for Block {
    fn render(&self) -> String {
        let mut r = "Block { stats: vec![".to_string();
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
            UnaryOp::Neg => "UnaryOp::Neg".to_string(),
            UnaryOp::Not => "UnaryOp::Not".to_string(),
        }
    }
}

impl Render for BinaryOp {
    fn render(&self) -> String {
        match self {
            BinaryOp::Mul => "BinaryOp::Mul".to_string(),
            BinaryOp::Div => "BinaryOp::Div".to_string(),
            BinaryOp::Mod => "BinaryOp::Mod".to_string(),
            BinaryOp::Add => "BinaryOp::Add".to_string(),
            BinaryOp::Sub => "BinaryOp::Sub".to_string(),
            BinaryOp::Shl => "BinaryOp::Shl".to_string(),
            BinaryOp::Shr => "BinaryOp::Shr".to_string(),
            BinaryOp::And => "BinaryOp::And".to_string(),
            BinaryOp::Or => "BinaryOp::Or".to_string(),
            BinaryOp::Xor => "BinaryOp::Xor".to_string(),
            BinaryOp::Eq => "BinaryOp::Eq".to_string(),
            BinaryOp::NotEq => "BinaryOp::NotEq".to_string(),
            BinaryOp::Greater => "BinaryOp::Greater".to_string(),
            BinaryOp::Less => "BinaryOp::Less".to_string(),
            BinaryOp::GreaterEq => "BinaryOp::GreaterEq".to_string(),
            BinaryOp::LessEq => "BinaryOp::LessEq".to_string(),
            BinaryOp::LogAnd => "BinaryOp::LogAnd".to_string(),
            BinaryOp::LogOr => "BinaryOp::LogOr".to_string(),
            BinaryOp::Assign => "BinaryOp::Assign".to_string(),
            BinaryOp::AddAssign => "BinaryOp::AddAssign".to_string(),
            BinaryOp::SubAssign => "BinaryOp::SubAssign".to_string(),
            BinaryOp::MulAssign => "BinaryOp::MulAssign".to_string(),
            BinaryOp::DivAssign => "BinaryOp::DivAssign".to_string(),
            BinaryOp::ModAssign => "BinaryOp::ModAssign".to_string(),
            BinaryOp::AndAssign => "BinaryOp::AndAssign".to_string(),
            BinaryOp::OrAssign => "BinaryOp::OrAssign".to_string(),
            BinaryOp::XorAssign => "BinaryOp::XorAssign".to_string(),
            BinaryOp::ShlAssign => "BinaryOp::ShlAssign".to_string(),
            BinaryOp::ShrAssign => "BinaryOp::ShrAssign".to_string(),
        }
    }
}

impl Render for Range {
    fn render(&self) -> String {
        match self {
            Range::Only(expr) => format!("Range::Only({})",expr.render()),
            Range::FromTo(expr,expr2) => format!("Range::FromTo({},{})",expr.render(),expr2.render()),
            Range::FromToIncl(expr,expr2) => format!("Range::FromToIncl({},{})",expr.render(),expr2.render()),
            Range::From(expr) => format!("Range::From({})",expr.render()),
            Range::To(expr) => format!("Range::To({})",expr.render()),
            Range::ToIncl(expr) => format!("Range::ToIncl({})",expr.render()),
            Range::All => "Range::All".to_string(),
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
        format!("(\"{}\".to_string(),{})",self.0,self.1.render())
    }
}

impl Render for UnknownVariantExpr {
    fn render(&self) -> String {
        match self {
            UnknownVariantExpr::Naked(ident) => format!("UnknownVariantExpr::Naked(\"{}\".to_string())",ident),
            UnknownVariantExpr::Tuple(ident,exprs) => format!("UnknownVariantExpr::Tuple(\"{}\".to_string(),{})",ident,exprs.render()),
            UnknownVariantExpr::Struct(ident,fields) => format!("UnknownVariantExpr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
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
            Expr::Boolean(value) => format!("Expr::Boolean({})",if *value { "true" } else { "false" }),
            Expr::Integer(value) => format!("Expr::Integer({} as i64)",*value),
            Expr::Float(value) => format!("Expr::Float({} as f64)",*value),
            Expr::Array(exprs) => format!("Expr::Array({})",exprs.render()),
            Expr::Cloned(expr,expr2) => format!("Expr::Cloned(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Index(expr,expr2) => format!("Expr::Index(Box::new({}),Box::new({}))",expr.render(),expr2.render()),
            Expr::Cast(expr,type_) => format!("Expr::Cast(Box::new({}),Box::new({}))",expr.render(),type_.render()),
            Expr::AnonTuple(exprs) => format!("Expr::AnonTuple({})",exprs.render()),
            Expr::Unary(op,expr) => format!("Expr::Unary({},Box::new({}))",op.render(),expr.render()),
            Expr::Binary(expr,op,expr2) => format!("Expr::Binary(Box::new({}),{},Box::new({}))",expr.render(),op.render(),expr2.render()),
            Expr::Continue => "Expr::Continue".to_string(),
            Expr::Break(expr) => format!("Expr::Break({})",expr.render()),
            Expr::Return(expr) => format!("Expr::Return({})",expr.render()),
            Expr::Block(block) => format!("Expr::Block({})",block.render()),
            Expr::If(expr,block,else_expr) => format!("Expr::If(Box::new({}),{},{})",expr.render(),block.render(),else_expr.render()),
            Expr::Loop(block) => format!("Expr::Loop({})",block.render()),
            Expr::While(expr,block) => format!("Expr::While(Box::new({}),{})",expr.render(),block.render()),
            Expr::IfLet(pats,expr,block,else_expr) => format!("Expr::IfLet({},Box::new({}),{},{})",pats.render(),expr.render(),block.render(),else_expr.render()),
            Expr::For(pats,range,block) => format!("Expr::For({},{},{})",pats.render(),range.render(),block.render()),
            Expr::WhileLet(pats,expr,block) => format!("Expr::WhileLet({},Box::new({}),{})",pats.render(),expr.render(),block.render()),
            Expr::Match(expr,arms) => format!("Expr::Match(Box::new({}),{})",expr.render(),arms.render()),
            Expr::UnknownIdent(ident) => format!("Expr::UnknownIdent(\"{}\".to_string())",ident),
            Expr::UnknownTupleOrCall(ident,exprs) => format!("Expr::UnknownTupleOrCall(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::UnknownStruct(ident,fields) => format!("Expr::UnknownStruct(\"{}\".to_string(),{})",ident,fields.render()),
            Expr::UnknownVariant(ident,variant) => format!("Expr::UnknownVariant(\"{}\".to_string(),{})",ident,variant.render()),
            Expr::UnknownMethod(expr,ident,exprs) => format!("Expr::UnknownMethod(Box::new({}),\"{}\".to_string(),{})",expr.render(),ident,exprs.render()),
            Expr::UnknownField(expr,ident) => format!("Expr::UnknownField(Box::new({}),\"{}\".to_string())",expr.render(),ident),
            Expr::UnknownTupleIndex(expr,index) => format!("Expr::UnknowTupleIndex(Box::new({}),{})",expr.render(),index),
        }
    }
}

impl Render for Stat {
    fn render(&self) -> String {
        match self {
            Stat::Let(pat,type_,expr) => format!("Stat::Let(Box::new({}),Box::new({}),Box::new({}))",pat.render(),type_.render(),expr.render()),
            Stat::Expr(expr) => format!("Stat::Expr({})",expr.render()),
        }
    }
}

impl Render for Struct {
    fn render(&self) -> String {
        let mut r = format!("Struct {{ ident: \"{}\".to_string(),fields: vec![",self.ident);
        for (ident,type_) in self.fields.iter() {
            r += &format!("Symbol {{ ident: \"{}\".to_string(),type_: {}, }},",ident,type_.render());
        }
        r += "], }";
        r
    }
}

impl Render for Tuple {
    fn render(&self) -> String {
        format!("Tuple {{ ident: \"{}\".to_string(),types: {}, }}",self.ident,self.types.render())
    }
}

impl Render for Variant {
    fn render(&self) -> String {
        match self {
            Variant::Naked(ident) => format!("Variant::Naked(\"{}\".to_string()),",ident),
            Variant::Tuple(ident,types) => format!("Variant::Tuple(\"{}\".to_string(),{})",ident,types.render()),
            Variant::Struct(ident,fields) => format!("Variant::Struct(\"{}\".to_string(),{})",ident,fields.render()),
        }
    }
}

impl Render for Enum {
    fn render(&self) -> String {
        format!("Enum {{ ident: \"{}\".to_string(),variants: {}, }}",self.ident,self.variants.render())
    }
}

impl Render for Alias {
    fn render(&self) -> String {
        format!("Alias {{ ident: \"{}\".to_string(),type_: {}, }}",self.ident,self.type_.render())
    }
}

impl Render for Const {
    fn render(&self) -> String {
        format!("Const {{ ident: \"{}\".to_string(),type_: {},expr: {}, }}",self.ident,self.type_.render(),self.expr.render())
    }
}

impl Render for Function {
    fn render(&self) -> String {
        format!("Function {{ ident: \"{}\".to_string(),params: {},type_: {},block: {}, }}",self.ident,self.params.render(),self.type_.render(),self.block.render())
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

        let mut r = "{ use super::*,e::gpu::shader::ast; ".to_string();

        if self.tuples.len() > 0 {
            r += &format!("let tuples: Vec<Tuple> = {}; ",self.tuples.render());
        }
        else {
            r += "let tuples: Vec<Tuple> = Vec::new(); ";
        }

        if extern_struct_idents.len() > 0 {
            r += "let extern_structs: Vec<Struct> = vec![";
            for extern_struct_ident in extern_struct_idents.iter() {
                r += &format!("super::{}::ast(),",extern_struct_ident);
            }
            r += "]; ";
        }
        else {
            r += "let extern_structs: Vec<Struct> = Vec::new(); ";
        }

        if self.structs.len() > 0 {
            r += &format!("let structs: Vec<Struct> = {}; ",self.structs.render());
        }
        else {
            r += "let mut structs: Vec<Struct> = Vec::new(); ";
        }

        if self.enums.len() > 0 {
            r += &format!("let enums: Vec<Enum> = {}; ",self.enums.render());
        }
        else {
            r += "let enums: Vec<Enum> = Vec::new(); ";
        }
        if self.aliases.len() > 0 {
            r += &format!("let aliases: Vec<Alias> = {}; ",self.aliases.render());
        }
        else {
            r += "let aliases: Vec<Alias> = HashMap::new(); ";
        }
        if self.consts.len() > 0 {
            r += &format!("let consts: Vec<Const> = {}; ",self.consts.render());
        }
        else {
            r += "let consts: Vec<Const> = Vec::new(); ";
        }
        if self.functions.len() > 0 {
            r += &format!("let functions: Vec<Function> = {}; ",self.functions.render());
        }
        else {
            r += "let functions: Vec<Function> = Vec::new(); ";
        }
        r += &format!("Source {{ ident: \"{}\".to_string(),tuples,structs,extern_structs,enums,aliases,consts,functions, }} }}",self.ident);
        r
    }
}
