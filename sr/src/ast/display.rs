use {
    crate::ast::*,
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
            Type::Tuple(tuple) => write!(f,"{}",tuple.borrow().ident),
            Type::Struct(struct_) => write!(f,"{}",struct_.borrow().ident),
            Type::Enum(enum_) => write!(f,"{}",enum_.borrow().ident),
            Type::Alias(alias) => write!(f,"{}",alias.borrow().ident),
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
            Pat::UnknownTuple(ident,pats) => {
                write!(f,"{}(",ident)?;
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
            Pat::UnknownStruct(ident,identpats) => {
                write!(f,"{} {{ ",ident)?;
                let mut first = true;
                for identpat in identpats.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    match identpat {
                        UnknownFieldPat::Wildcard => write!(f,"_")?,
                        UnknownFieldPat::Rest => write!(f,"..")?,
                        UnknownFieldPat::Ident(ident) => write!(f,"{}",ident)?,
                        UnknownFieldPat::IdentPat(ident,pat) => write!(f,"{}: {}",ident,pat)?,
                    }
                    first = false;
                }
                write!(f," }}")
            },
            Pat::UnknownVariant(ident,identpatvariant) => {
                write!(f,"{}::",ident)?;
                match identpatvariant {
                    UnknownVariantPat::Naked(ident) => write!(f,"{}",ident),
                    UnknownVariantPat::Tuple(ident,pats) => {
                        write!(f,"{}(",ident)?;
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
                    UnknownVariantPat::Struct(ident,identpats) => {
                        write!(f,"{} {{ ",ident)?;
                        let mut first = true;
                        for identpat in identpats.iter() {
                            if !first {
                                write!(f,", ")?;
                            }
                            match identpat {
                                UnknownFieldPat::Wildcard => write!(f,"_")?,
                                UnknownFieldPat::Rest => write!(f,"..")?,
                                UnknownFieldPat::Ident(ident) => write!(f,"{}",ident)?,
                                UnknownFieldPat::IdentPat(ident,pat) => write!(f,"{}: {}",ident,pat)?,
                            }
                            first = false;
                        }
                        write!(f," }}")
                    },
                }
            },
            Pat::Tuple(tuple,pats) => {
                write!(f,"{}(",tuple.borrow().ident)?;
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
            Pat::Struct(struct_,index_pats) => {
                write!(f,"{} {{ ",struct_.borrow().ident)?;
                let mut first = true;
                for indexpat in index_pats.iter() {
                    if !first {
                        write!(f,", ")?;
                    }
                    match indexpat {
                        FieldPat::Wildcard => write!(f,"_")?,
                        FieldPat::Rest => write!(f,"..")?,
                        FieldPat::Index(index) => write!(f,"{}",struct_.borrow().fields[*index].ident)?,
                        FieldPat::IndexPat(index,pat) => write!(f,"{}: {}",struct_.borrow().fields[*index].ident,pat)?,
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
            Expr::UnknownTupleOrCall(ident,exprs) => {
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
            Expr::UnknownStruct(ident,fields) => {
                write!(f,"{} {{ ",ident)?;
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
            Expr::UnknownVariant(ident,variant) => {
                write!(f,"{}::",ident)?;
                match variant {
                    UnknownVariantExpr::Naked(ident) => write!(f,"{}",ident),
                    UnknownVariantExpr::Tuple(ident,exprs) => {
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
                    UnknownVariantExpr::Struct(ident,fields) => {
                        write!(f,"{} {{",ident)?;
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
                }
            },
            Expr::UnknownMethod(expr,ident,exprs) => {
                write!(f,"{}.{}(",expr,ident)?;
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
            Expr::UnknownField(expr,ident) => write!(f,"{}.{}",expr,ident),
            Expr::UnknownTupleIndex(expr,index) => write!(f,"{}.{}",expr,index),
            Expr::Param(param) => write!(f,"{}",param.borrow().ident),
            Expr::Local(local) => write!(f,"{}",local.borrow().ident),
            Expr::Const(const_) => write!(f,"{}",const_.borrow().ident),
            Expr::Tuple(tuple,exprs) => {
                write!(f,"{}(",tuple.borrow().ident)?;
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
            Expr::Call(function,exprs) => {
                write!(f,"{}(",function.borrow().ident)?;
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
            Expr::Struct(struct_,exprs) => {
                write!(f,"{} {{ ",struct_.borrow().ident)?;
                let mut first = true;
                for i in 0..struct_.borrow().fields.len() {
                    if !first {
                        write!(f,", ")?;
                    }
                    write!(f,"{}: {}",struct_.borrow().fields[i].ident,exprs[i])?;
                    first = false;
                }
                write!(f," }}")
            },
            Expr::Variant(enum_,variant) => {
                write!(f,"{}::",enum_.borrow().ident)?;
                match variant {
                    VariantExpr::Naked(index) => if let Variant::Naked(ident) = &enum_.borrow().variants[*index] {
                        write!(f,"{}",ident)
                    }
                    else {
                        panic!("enum variant mismatch");
                    },
                    VariantExpr::Tuple(index,exprs) => if let Variant::Tuple(ident,_) = &enum_.borrow().variants[*index] {
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
                    }
                    else {
                        panic!("enum variant mismatch");
                    },
                    VariantExpr::Struct(index,elements) => if let Variant::Struct(ident,fields) = &enum_.borrow().variants[*index] {
                        write!(f,"{} {{ ",ident)?;
                        let mut first = true;
                        for i in 0..fields.len() {
                            if !first {
                                write!(f,",")?;
                            }
                            write!(f,"{}: {}",fields[i].ident,elements[i])?;
                            first = false;
                        }
                        write!(f," }}")
                    }
                    else {
                        panic!("enum variant mismatch");
                    },
                }
            },
            Expr::Method(expr,method,exprs) => {
                write!(f,"{}.{}(",expr,method.borrow().ident)?;
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
            Expr::Field(struct_,expr,index) => write!(f,"{}.{}",expr,struct_.borrow().fields[*index].ident),
            Expr::TupleIndex(_,expr,index) => write!(f,"{}.{}",expr,index),
            Expr::Discriminant(expr) => write!(f,"discriminant({})",expr),
            Expr::Destructure(expr,variant_index,index) => write!(f,"{}::{}.{}",expr,variant_index,index),
        }
    }
}

impl Display for Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Stat::Expr(expr) => write!(f,"{};",expr),
            Stat::Let(pat,type_,expr) => write!(f,"let {}: {} = {};",pat,type_,expr),
            Stat::Local(local,expr) => write!(f,"let {} = {};",local.borrow().ident,expr),
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
            write!(f,"{}: {}",param.borrow().ident,param.borrow().type_)?;
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

impl Display for Module {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"mod {} {{\n",self.ident)?;
        for ident in self.aliases.keys() {
            write!(f,"{}\n",self.aliases[ident].borrow())?;
        }
        for ident in self.tuples.keys() {
            write!(f,"{}\n",self.tuples[ident].borrow())?;
        }
        for ident in self.structs.keys() {
            write!(f,"{}\n",self.structs[ident].borrow())?;
        }
        for ident in self.enums.keys() {
            write!(f,"{}\n",self.enums[ident].borrow())?;
        }
        for ident in self.consts.keys() {
            write!(f,"{}\n",self.consts[ident].borrow())?;
        }
        for ident in self.functions.keys() {
            write!(f,"{}\n",self.functions[ident].borrow())?;
        }
        write!(f,"}}")
    }
}
