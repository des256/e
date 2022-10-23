use {
    super::ast::*,
    std::{
        fmt::{
            Display,
            Formatter,
            Result,
        },
    },
};

impl Display for Type {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Type::Inferred => write!(f,"_"),
            Type::Void => write!(f,"()"),
            Type::Integer => write!(f,"{{integer}}"),
            Type::Float => write!(f,"{{float}}"),
            Type::Bool => write!(f,"bool"),
            Type::U8 => write!(f,"u8"),
            Type::I8 => write!(f,"i8"),
            Type::U16 => write!(f,"u16"),
            Type::I16 => write!(f,"i16"),
            Type::U32 => write!(f,"u32"),
            Type::I32 => write!(f,"i32"),
            Type::U64 => write!(f,"u64"),
            Type::I64 => write!(f,"i64"),
            Type::USize => write!(f,"usize"),
            Type::ISize => write!(f,"isize"),
            Type::F16 => write!(f,"f16"),
            Type::F32 => write!(f,"f32"),
            Type::F64 => write!(f,"f64"),
            Type::AnonTuple(types) => {
                write!(f,"(")?;
                let mut first = true;
                for type_ in types.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",type_)?;
                    first = false;
                }
                write!(f,")")
            },
            Type::Array(type_,expr) => write!(f,"[{}; {}]",type_,expr),
            Type::UnknownIdent(ident) => write!(f,"{}",ident),
            Type::Tuple(tuple_ident) => write!(f,"{}",tuple_ident),
            Type::Struct(struct_ident) => write!(f,"{}",struct_ident),
            Type::Enum(enum_ident) => write!(f,"{}",enum_ident),
            Type::Alias(alias_ident) => write!(f,"{}",alias_ident),
        }
    }
}

impl Display for Pat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Pat::Wildcard => write!(f,"_"),
            Pat::Rest => write!(f,".."),
            Pat::Boolean(value) => write!(f,"{}",if *value { "true" } else { "false " }),
            Pat::Integer(value) => write!(f,"{}",value),
            Pat::Float(value) => write!(f,"{}",value),
            Pat::AnonTuple(pats) => {
                write!(f,"(")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",pat)?;
                    first = false;
                }
                write!(f,")")
            },
            Pat::Array(pats) => {
                write!(f,"[")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    write!(f,"{}",pat)?;
                    first = false;
                }
                write!(f,"]")
            },
            Pat::Range(from,to) => write!(f,"{} ..= {}",from,to),
            Pat::UnknownIdent(ident) => write!(f,"{}",ident),
            Pat::Const(ident) => write!(f,"{}",ident),
            Pat::Tuple(tuple_ident,pats) => {
                write!(f,"{}(",tuple_ident)?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",pat)?;
                    first = false;
                }
                write!(f,")")
            },
            Pat::Struct(struct_ident,index_pats) => {
                write!(f,"{} {{ ",struct_ident)?;
                let mut first = true;
                for indexpat in index_pats.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    match indexpat {
                        FieldPat::Wildcard => write!(f,"_")?,
                        FieldPat::Rest => write!(f,"..")?,
                        FieldPat::Ident(ident) => write!(f,"{}",ident)?,
                        FieldPat::IdentPat(ident,pat) => write!(f,"{}: {}",ident,pat)?,
                    }
                    first = false;
                }
                write!(f," }}")
            },
            Pat::Variant(_,_) => {
                // TODO
                write!(f,"TODO")
            },
        }
    }
}

impl Display for Block {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"{{ ")?;
        for stat in self.stats.iter() {
            write!(f,"{} ",stat)?;
        }
        if let Some(expr) = &self.expr {
            write!(f,"{} ",expr)?;
        }
        write!(f,"}}")
    }
}

impl Display for UnaryOp {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            UnaryOp::Neg => write!(f,"-"),
            UnaryOp::Not => write!(f,"!"),
        }
    }
}

impl Display for BinaryOp {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            BinaryOp::Mul => write!(f,"*"),
            BinaryOp::Div => write!(f,"/"),
            BinaryOp::Mod => write!(f,"%"),
            BinaryOp::Add => write!(f,"+"),
            BinaryOp::Sub => write!(f,"-"),
            BinaryOp::Shl => write!(f,"<<"),
            BinaryOp::Shr => write!(f,">>"),
            BinaryOp::And => write!(f,"&"),
            BinaryOp::Or => write!(f,"|"),
            BinaryOp::Xor => write!(f,"^"),
            BinaryOp::Eq => write!(f,"=="),
            BinaryOp::NotEq => write!(f,"!="),
            BinaryOp::Greater => write!(f,">"),
            BinaryOp::Less => write!(f,"<"),
            BinaryOp::GreaterEq => write!(f,">="),
            BinaryOp::LessEq => write!(f,"<="),
            BinaryOp::LogAnd => write!(f,"&&"),
            BinaryOp::LogOr => write!(f,"||"),
            BinaryOp::Assign => write!(f,"="),
            BinaryOp::AddAssign => write!(f,"+="),
            BinaryOp::SubAssign => write!(f,"-="),
            BinaryOp::MulAssign => write!(f,"*="),
            BinaryOp::DivAssign => write!(f,"/="),
            BinaryOp::ModAssign => write!(f,"%="),
            BinaryOp::AndAssign => write!(f,"&="),
            BinaryOp::OrAssign => write!(f,"|="),
            BinaryOp::XorAssign => write!(f,"^="),
            BinaryOp::ShlAssign => write!(f,"<<="),
            BinaryOp::ShrAssign => write!(f,">>="),
        }
    }
}

impl Display for Range {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Range::Only(expr) => write!(f,"{}",expr),
            Range::FromTo(expr,expr2) => write!(f,"{}..{}",expr,expr2),
            Range::FromToIncl(expr,expr2) => write!(f,"{}..={}",expr,expr2),
            Range::From(expr) => write!(f,"{}..",expr),
            Range::To(expr) => write!(f,"..{}",expr),
            Range::ToIncl(expr) => write!(f,"..={}",expr),
            Range::All => write!(f,".."),
        }
    }
}

impl Display for Expr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Expr::Boolean(value) => write!(f,"{}",if *value { "true" } else { "false" }),
            Expr::Integer(value) => write!(f,"{}i64",value),
            Expr::Float(value) => write!(f,"{}f64",value),
            Expr::Array(exprs) => {
                write!(f,"[ ")?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,"]")
            },
            Expr::Cloned(expr,expr2) => write!(f,"[ {}; {} ]",expr,expr2),
            Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            Expr::Cast(expr,type_) => write!(f,"{} as {}",expr,type_),
            Expr::AnonTuple(exprs) => {
                write!(f,"(")?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,")")
            },
            Expr::Unary(op,expr) => write!(f,"{}{}",op,expr),
            Expr::Binary(expr,op,expr2) => write!(f,"{} {} {}",expr,op,expr2),
            Expr::Continue => write!(f,"continue"),
            Expr::Break(expr) => if let Some(expr) = expr {
                write!(f,"break {}",expr)
            }
            else {
                write!(f,"break")
            },
            Expr::Return(expr) => if let Some(expr) = expr {
                write!(f,"return {}",expr)
            }
            else {
                write!(f,"return")
            },
            Expr::Block(block) => write!(f,"{}",block),
            Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr {
                write!(f,"if {} {} else {}",expr,block,else_expr)
            }
            else {
                write!(f,"if {} {}",expr,block)
            },
            Expr::Loop(block) => write!(f,"loop {}",block),
            Expr::While(expr,block) => write!(f,"while {} {}",expr,block),
            Expr::IfLet(pats,expr,block,else_expr) => if let Some(else_expr) = else_expr {
                write!(f,"if let ")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first = false;
                }
                write!(f,"= {} {} else {}",expr,block,else_expr)
            }
            else {
                write!(f,"if let ")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first = false;
                }
                write!(f,"= {} {}",expr,block)
            },
            Expr::For(pats,range,block) => {
                write!(f,"for ")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first = false;
                }
                write!(f,"in {} {}",range,block)
            },
            Expr::WhileLet(pats,expr,block) => {
                write!(f,"while let ")?;
                let mut first = true;
                for pat in pats.iter() {
                    if !first {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first = false;
                }
                write!(f,"= {} {}",expr,block)
            },
            Expr::Match(expr,arms) => {
                write!(f,"match {} {{",expr)?;
                let mut first = true;
                for (pats,if_expr,expr) in arms.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    let mut first_pat = true;
                    for pat in pats.iter() {
                        if !first_pat {
                            write!(f,"| ")?;
                        }
                        write!(f,"{} ",pat)?;
                        first_pat = false;
                    }
                    if let Some(if_expr) = if_expr {
                        write!(f,"if {} ",if_expr)?;
                    }
                    write!(f,"=> {}",expr)?;
                    first = false;
                }
                write!(f," }}")
            },
            Expr::UnknownIdent(ident) => write!(f,"{}",ident),
            Expr::TupleOrCall(ident,exprs) => {
                write!(f,"{}(",ident)?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,")")
            },
            Expr::Param(param) => write!(f,"{}",param),
            Expr::Local(local) => write!(f,"{}",local),
            Expr::Const(const_) => write!(f,"{}",const_),
            Expr::Tuple(tuple_ident,exprs) => {
                write!(f,"{}(",tuple_ident)?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,")")
            },
            Expr::Call(function_ident,exprs) => {
                write!(f,"{}(",function_ident)?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,")")
            },
            Expr::Struct(struct_ident,fields) => {
                write!(f,"{} {{ ",struct_ident)?;
                let mut first = true;
                for (ident,expr) in fields.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    write!(f,"{}: {}",ident,expr)?;
                    first = false;
                }
                write!(f," }}")
            },
            Expr::Variant(enum_ident,variant) => {
                write!(f,"{}::",enum_ident)?;
                match variant {
                    VariantExpr::Naked(ident) => write!(f,"{}",ident),
                    VariantExpr::Tuple(ident,exprs) => {
                        write!(f,"{}(",ident)?;
                        let mut first = true;
                        for expr in exprs.iter() {
                            if !first {
                                write!(f,",")?;
                            }
                            write!(f,"{}",expr)?;
                            first = false;
                        }
                        write!(f,")")
                    },
                    VariantExpr::Struct(ident,fields) => {
                        write!(f,"{} {{ ",ident)?;
                        let mut first = true;
                        for (ident,expr) in fields {
                            if !first {
                                write!(f,",")?;
                            }
                            write!(f,"{}: {}",ident,expr)?;
                            first = false;
                        }
                        write!(f," }}")
                    },
                }
            },
            Expr::Method(expr,method_ident,exprs) => {
                write!(f,"{}.{}(",expr,method_ident)?;
                let mut first = true;
                for expr in exprs.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first = false;
                }
                write!(f,")")
            },
            Expr::Field(expr,ident) => write!(f,"{}.{}",expr,ident),
            Expr::TupleIndex(expr,index) => write!(f,"{}.{}",expr,index),
            Expr::Discriminant(expr,variant_ident) => write!(f,"discriminant({} == {})",expr,variant_ident),
            Expr::DestructTuple(expr,variant_ident,index) => write!(f,"{}::{}.{}",expr,variant_ident,index),
            Expr::DestructStruct(expr,variant_ident,ident) => write!(f,"{}::{}.{}",expr,variant_ident,ident),
        }
    }
}

impl Display for Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Stat::Expr(expr) => write!(f,"{};",expr),
            Stat::Let(pat,type_,expr) => write!(f,"let {}: {} = {};",pat,type_,expr),
            Stat::Local(ident,type_,expr) => write!(f,"let {}: {} = {};",ident,type_,expr),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"{}: {}",self.ident,self.type_)
    }
}

impl Display for Method {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"fn {}(self: {}",self.ident,self.from_type)?;
        for param in &self.params {
            write!(f,",{}: {}",param.ident,param.type_)?;
        }
        write!(f,")")?;
        if let Type::Void = self.type_ { } else {
            write!(f," -> {}",self.type_)?;
        }
        write!(f,"")
    }
}

impl Display for Function {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"fn {}(",self.ident)?;
        let mut first = true;
        for param in self.params.iter() {
            if !first {
                write!(f,",")?;
            }
            write!(f,"{}: {}",param.ident,param.type_)?;
            first = false;
        }
        write!(f,")")?;
        if let Type::Void = self.type_ { } else {
            write!(f," -> {}",self.type_)?;
        }
        write!(f," {}",self.block)
    }
}

impl Display for Tuple {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"struct {}(",self.ident)?;
        let mut first = true;
        for type_ in self.types.iter() {
            if !first {
                write!(f,",")?;
            }
            write!(f,"{}",type_)?;
            first = false;
        }
        write!(f,")")
    }
}

impl Display for Struct {
    fn fmt(&self,f: &mut Formatter) -> Result { 
        write!(f,"struct {} {{ ",self.ident)?;
        let mut first = true;
        for field in self.fields.iter() {
            if !first {
                write!(f,", ")?;
            }
            write!(f,"{}: {}",field.ident,field.type_)?;
            first = false;
        }
        write!(f," }}")
    }
}

impl Display for Variant {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Variant::Naked(ident) => write!(f,"{}",ident),
            Variant::Tuple(ident,types) => {
                write!(f,"{}(",ident)?;
                let mut first = true;
                for type_ in types.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}",type_)?;
                    first = false;
                }
                write!(f,")")
            },
            Variant::Struct(ident,fields) => {
                write!(f,"{} {{ ",ident)?;
                let mut first = true;
                for field in fields.iter() {
                    if !first {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",field.ident,field.type_)?;
                    first = false;
                }
                write!(f," }}")
            },
        }
    }
}

impl Display for Enum {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"enum {} {{ ",self.ident)?;
        let mut first = true;
        for variant in self.variants.iter() {
            if !first {
                write!(f,", ")?;
            }
            write!(f,"{}",variant)?;
            first = false;
        }
        write!(f," }}")
    }
}

impl Display for Const {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"const {}: {} = {}",self.ident,self.type_,self.expr)
    }
}

impl Display for Alias {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"type {} = {}",self.ident,self.type_)
    }
}
