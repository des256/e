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
            Type::F16 => "Type::F16".to_string(),
            Type::F32 => "Type::F32".to_string(),
            Type::F64 => "Type::F64".to_string(),
            Type::AnonTuple(types) => format!("Type::AnonTuple({})",types.render()),
            Type::Array(type_,expr) => format!("Type::Array(Box::new({}),Box::new({}))",type_.render(),expr.render()),
            Type::UnknownStructTupleEnumAlias(ident) => format!("Type::UnknownStructTupleEnumAlias(\"{}\".to_string())",ident),
            Type::Struct(ident) => format!("Type::Struct(\"{}\".to_string())",ident),
            Type::Tuple(ident) => format!("Type::Tuple(\"{}\".to_string())",ident),
            Type::Enum(ident) => format!("Type::Enum(\"{}\".to_string())",ident),
            Type::Alias(ident) => format!("Type::Alias(\"{}\".to_string())",ident),
        }
    }
}

impl Render for FieldPat {
    fn render(&self) -> String {
        match self {
            FieldPat::Wildcard => "FieldPat::Wildcard".to_string(),
            FieldPat::Rest => "FieldPat::Rest".to_string(),
            FieldPat::Ident(ident) => format!("FieldPat::Ident(\"{}\".to_string())",ident),
            FieldPat::IdentPat(ident,pat) => format!("FieldPat::IdentPat(\"{}\".to_string(),{})",ident,pat.render()),
        }
    }
}

impl Render for VariantPat {
    fn render(&self) -> String {
        match self {
            VariantPat::Naked => format!("VariantPat::Naked"),
            VariantPat::Tuple(pats) => format!("VariantPat::Tuple({})",pats.render()),
            VariantPat::Struct(identpats) => format!("VariantPat::Struct({})",identpats.render()),
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
            Pat::Ident(ident) => format!("Pat::Ident(\"{}\".to_string())",ident),
            Pat::Tuple(ident,pats) => format!("Pat::Tuple(\"{}\".to_string(),{})",ident,pats.render()),
            Pat::Struct(ident,identpats) => format!("Pat::Struct(\"{}\".to_string(),{})",ident,identpats.render()),
            Pat::Variant(enum_ident,variant_ident,variant) => format!("Pat::Variant(\"{}\".to_string(),\"{}\".to_string(),{})",enum_ident,variant_ident,variant.render()),
        }
    }
}

impl Render for VariantExpr {
    fn render(&self) -> String {
        match self {
            VariantExpr::Naked => "VariantExpr::Naked".to_string(),
            VariantExpr::Tuple(exprs) => format!("VariantExpr::Tuple({})",exprs.render()),
            VariantExpr::Struct(fields) => format!("VariantExpr::Struct({})",fields.render()),
        }
    }
}

impl Render for Block {
    fn render(&self) -> String {
        let mut r = format!("Block {{ stats: {},expr: ",self.stats.render());
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
            Expr::While(expr,block) => format!("Expr::While(Box::new({}),{})",expr.render(),block.render()),
            Expr::Loop(block) => format!("Expr::Loop({})",block.render()),
            Expr::IfLet(pats,expr,block,else_expr) => format!("Expr::IfLet({},Box::new({}),{},{})",pats.render(),expr.render(),block.render(),else_expr.render()),
            Expr::For(pats,range,block) => format!("Expr::For({},{},{})",pats.render(),range.render(),block.render()),
            Expr::WhileLet(pats,expr,block) => format!("Expr::WhileLet({},Box::new({}),{})",pats.render(),expr.render(),block.render()),
            Expr::Match(expr,arms) => format!("Expr::Match(Box::new({}),{})",expr.render(),arms.render()),
            Expr::UnknownLocalConst(ident) => format!("Expr::UnknownLocalConst(\"{}\".to_string())",ident),
            Expr::Local(ident) => format!("Expr::Local(\"{}\".to_string())",ident),
            Expr::Const(ident) => format!("Expr::Const(\"{}\".to_string())",ident),
            Expr::UnknownTupleFunctionCall(ident,exprs) => format!("Expr::UnknownTupleFunctionCall(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::Tuple(ident,exprs) => format!("Expr::Tuple(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::FunctionCall(ident,exprs) => format!("Expr::FunctionCall(\"{}\".to_string(),{})",ident,exprs.render()),
            Expr::UnknownStruct(ident,fields) => format!("Expr::UnknownStruct(\"{}\".to_string(),{})",ident,fields.render()),
            Expr::Struct(ident,fields) => format!("Expr::Struct(\"{}\".to_string(),{})",ident,fields.render()),
            Expr::UnknownVariant(enum_ident,variant_ident,variant) => format!("Expr::UnknownVariant(\"{}\".to_string(),\"{}\".to_string(),{})",enum_ident,variant_ident,variant.render()),
            Expr::Variant(enum_ident,index,variant) => format!("Expr::Variant(\"{}\".to_string(),{},{})",enum_ident,index,variant.render()),
            Expr::UnknownMethodCall(expr,ident,exprs) => format!("Expr::UnknownMethodCall(Box::new({}),\"{}\".to_string(),{})",expr.render(),ident,exprs.render()),
            Expr::MethodCall(expr,ident,exprs) => format!("Expr::MethodCall(Box::new({}),\"{}\".to_string(),{})",expr.render(),ident,exprs.render()),
            Expr::UnknownField(expr,ident) => format!("Expr::UnknownField(Box::new({}),\"{}\".to_string())",expr.render(),ident),
            Expr::Field(expr,struct_ident,index) => format!("Expr::Field(Box::new({}),\"{}\".to_string(),{})",expr.render(),struct_ident,index),
            Expr::UnknownTupleIndex(expr,index) => format!("Expr::UnknownTupleIndex(Box::new({}),{})",expr.render(),index),
            Expr::TupleIndex(expr,tuple_ident,index) => format!("Expr::TupleIndex(Box::new({}),{},{})",expr.render(),tuple_ident,index),
        }
    }
}

impl Render for Stat {
    fn render(&self) -> String {
        match self {
            Stat::Let(pat,type_,expr) => format!("Stat::Let(Box::new({}),Box::new({}),Box::new({}))",pat.render(),type_.render(),expr.render()),
            Stat::Local(ident,type_,expr) => format!("Stat::Local(\"{}\".to_string(),Box::new({}),Box::new({}))",ident,type_.render(),expr.render()),
            Stat::Expr(expr) => format!("Stat::Expr(Box::new({}))",expr.render()),
        }
    }
}

impl Render for Struct {
    fn render(&self) -> String {
        let mut r = format!("Struct {{ ident: \"{}\".to_string(),fields: vec![",self.ident);
        for (ident,type_) in self.fields.iter() {
            r += &format!("(\"{}\".to_string(),{}),",ident,type_.render());
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
            Variant::Naked => format!("Variant::Naked"),
            Variant::Tuple(types) => format!("Variant::Tuple({})",types.render()),
            Variant::Struct(fields) => format!("Variant::Struct({})",fields.render()),
        }
    }
}

impl Render for Enum {
    fn render(&self) -> String {
        let mut r = format!("Enum {{ ident: \"{}\".to_string(),variants: vec![",self.ident);
        for (ident,variant) in self.variants.iter() {
            r += &format!("(\"{}\".to_string(),{})",ident,variant.render());
        }
        r += "], }";
        r
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
            if function.0 == "main" {
                for (_,type_) in function.1.params.iter() {
                    if let Type::UnknownStructTupleEnumAlias(ident) = type_ {
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

        let mut r = "{ use { super::*,std::collections::HashMap }; ".to_string();

        if self.tuples.len() > 0 {
            r += &format!("let mut tuples: HashMap<String,Tuple> = HashMap::new(); ");
            for tuple in self.tuples.iter() {
                r += &format!("tuples.insert(\"{}\".to_string(),{}); ",tuple.0,tuple.1.render());
            }
        }
        else {
            r += "let tuples: HashMap<String,Tuple> = HashMap::new(); ";
        }

        if extern_struct_idents.len() > 0 {
            r += "let mut extern_structs: HashMap<String,Struct> = HashMap::new();";
            for extern_struct_ident in extern_struct_idents.iter() {
                // hack: if struct contains <>, it's not an external struct, but something defined in the standard library
                // there are probably other structs as well, but for now let's assume they're not defined in the standard library
                if !extern_struct_ident.contains('<') {
                    r += &format!("extern_structs.insert(\"{}\".to_string(),super::{}::ast());",extern_struct_ident,extern_struct_ident);
                }
            }
        }
        else {
            r += "let extern_structs: HashMap<String,Struct> = HashMap::new(); ";
        }

        if self.structs.len() > 0 {
            r += &format!("let mut structs: HashMap<String,Struct> = HashMap::new();");
            for struct_ in self.structs.iter() {
                r += &format!("structs.insert(\"{}\".to_string(),{}); ",struct_.0,struct_.1.render());
            }
        }
        else {
            r += "let mut structs: HashMap<String,Struct> = HashMap::new(); ";
        }

        if self.enums.len() > 0 {
            r += &format!("let mut enums: HashMap<String,Enum> = HashMap::new(); ");
            for enum_ in self.enums.iter() {
                r += &format!("enums.insert(\"{}\".to_string(),{}); ",enum_.0,enum_.1.render());
            }
        }
        else {
            r += "let enums: HashMap<String,Enum> = HashMap::new(); ";
        }
        if self.aliases.len() > 0 {
            r += &format!("let mut aliases: HashMap<String,Alias> = HashMap::new(); ");
            for alias in self.aliases.iter() {
                r += &format!("aliases.insert(\"{}\".to_string(),{}); ",alias.0,alias.1.render());
            }
        }
        else {
            r += "let aliases: HashMap<String,Alias> = HashMap::new(); ";
        }
        if self.consts.len() > 0 {
            r += &format!("let mut consts: HashMap<String,Const> = HashMap::new(); ");
            for const_ in self.consts.iter() {
                r += &format!("consts.insert(\"{}\".to_string(),{}); ",const_.0,const_.1.render());
            }
        }
        else {
            r += "let consts: HashMap<String,Const> = HashMap::new(); ";
        }
        if self.functions.len() > 0 {
            r += &format!("let mut functions: HashMap<String,Function> = HashMap::new(); ");
            for function in self.functions.iter() {
                r += &format!("functions.insert(\"{}\".to_string(),{}); ",function.0,function.1.render());
            }
        }
        else {
            r += "let functions: HashMap<String,Function> = Vec::new(); ";
        }
        r += &format!("Module {{ ident: \"{}\".to_string(),tuples,structs,extern_structs,tuple_structs: HashMap::new(),anon_tuple_structs: HashMap::new(),enums,aliases,consts,functions, }} }}",self.ident);
        r
    }
}
