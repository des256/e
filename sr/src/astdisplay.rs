use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Display for Type {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Type::Inferred => write!(f,"_"),
            Type::Boolean => write!(f,"{{boolean}}"),
            Type::Integer => write!(f,"{{integer}}"),
            Type::Float => write!(f,"{{float}}"),
            Type::Void => write!(f,"()"),
            Type::Base(base_type) => write!(f,"{}",base_type.to_rust()),
            Type::UnknownIdent(ident) => write!(f,"{{unknown:{}}}",ident),
            Type::Struct(struct_) => write!(f,"{}",struct_.ident),
            Type::Enum(enum_) => write!(f,"{}",enum_.ident),
            Type::Array(ty,expr) => write!(f,"[{}; {}]",ty,expr),
        }
    }
}

impl Display for IdentPat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            IdentPat::Wildcard => write!(f,"_"),
            IdentPat::Rest => write!(f,".."),
            IdentPat::Ident(ident) => write!(f,"{}",ident),
            IdentPat::IdentPat(ident,pat) => write!(f,"{}: {}",ident,pat),
        }
    }
}

impl Display for VariantPat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            VariantPat::Naked(ident) => write!(f,"{}",ident),
            VariantPat::Tuple(ident,pats) => {
                write!(f,"{}(",ident)?;
                for pat in pats {
                    write!(f,"{},",pat)?;
                }
                write!(f,")")
            },
            VariantPat::Struct(ident,identpats) => {
                write!(f,"{} {{ ",ident)?;
                for identpat in identpats {
                    write!(f,"{},",identpat)?;
                }
                write!(f," }}")
            },
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
            Pat::Ident(ident) => write!(f,"{}",ident),
            Pat::Const(variable) => write!(f,"{}",variable.ident),
            Pat::UnknownStruct(ident,identpats) => {
                write!(f,"{} {{ ",ident)?;
                for identpat in identpats {
                    write!(f,"{},",identpat)?;
                }
                write!(f," }}")
            },
            Pat::Struct(struct_,identpats) => {
                write!(f,"{} {{ ",struct_.ident)?;
                for identpat in identpats {
                    write!(f,"{},",identpat)?;
                }
                write!(f," }}")
            },
            Pat::Array(pats) => {
                write!(f,"[")?;
                for pat in pats {
                    write!(f,"{},",pat)?;
                }
                write!(f,"]")
            },
            Pat::UnknownVariant(ident,variantpat) => write!(f,"{}::{}",ident,variantpat),
            Pat::Variant(enum_,variantpat) => write!(f,"{}::{}",enum_.ident,variantpat),
            Pat::AnonTuple(pats) => {
                write!(f,"(")?;
                for pat in pats {
                    write!(f,"{},",pat)?;
                }
                write!(f,")")
            },
            Pat::Range(pat,pat2) => write!(f,"{}..={}",pat,pat2),
        }
    }
}

impl Display for VariantExpr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            VariantExpr::Naked(ident) => write!(f,"{}",ident),
            VariantExpr::Tuple(ident,exprs) => {
                write!(f,"{}(",ident)?;
                for expr in exprs {
                    write!(f,"{},",expr)?;
                }
                write!(f,")")
            },
            VariantExpr::Struct(ident,fields) => {
                write!(f,"{} {{ ",ident)?;
                for (ident,expr) in fields {
                    write!(f,"{}: {},",ident,expr)?;
                }
                write!(f," }}")
            },
        }
    }
}

impl Display for Block {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"{{ ")?;
        for stat in self.stats.iter() {
            write!(f,"{}; ",stat)?;
        }
        if let Some(expr) = &self.expr {
            write!(f,"{}",expr)?;
        }
        write!(f," }}")
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
            Expr::Integer(value) => write!(f,"{}",value),
            Expr::Float(value) => write!(f,"{}",value),
            Expr::Base(base_type,fields) => {
                write!(f,"{} {{ ",base_type.to_rust())?;
                for (ident,expr) in fields {
                    write!(f,"{}: {},",ident,expr)?;
                }
                write!(f," }}")
            },
            Expr::UnknownIdent(ident) => write!(f,"{}",ident),
            Expr::Local(var) => write!(f,"{}",var.ident),
            Expr::Param(var) => write!(f,"{}",var.ident),
            Expr::Const(var) => write!(f,"{}",var.ident),
            Expr::Array(exprs) => {
                write!(f,"[")?;
                for expr in exprs {
                    write!(f,"{},",expr)?;
                }
                write!(f,"]")
            },
            Expr::Cloned(expr,expr2) => write!(f,"[{}; {}]",expr,expr2),
            Expr::UnknownStruct(ident,fields) => {
                write!(f,"{} {{ ",ident)?;
                for (ident,expr) in fields {
                    write!(f,"{}: {},",ident,expr)?;
                }
                write!(f," }}")
            },
            Expr::Struct(struct_,fields) => {
                write!(f,"{} {{ ",struct_.ident)?;
                for (ident,expr) in fields {
                    write!(f,"{}: {},",ident,expr)?;
                }
                write!(f," }}")
            },
            Expr::UnknownVariant(ident,variantexpr) => write!(f,"{}::{}",ident,variantexpr),
            Expr::Variant(enum_,variantexpr) => write!(f,"{}::{}",enum_.ident,variantexpr),
            Expr::UnknownCall(ident,exprs) => {
                write!(f,"{}(",ident)?;
                for expr in exprs {
                    write!(f,"{},",expr)?;
                }
                write!(f,")")
            },
            Expr::Call(function,exprs) => {
                write!(f,"{}(",function.ident)?;
                for expr in exprs {
                    write!(f,"{},",expr)?;
                }
                write!(f,")")
            },
            Expr::Field(expr,ident) => write!(f,"{}.{}",expr,ident),
            Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            Expr::Cast(expr,ty) => write!(f,"{} as {}",expr,ty),
            Expr::AnonTuple(exprs) => {
                write!(f,"(")?;
                for expr in exprs {
                    write!(f,"{},",expr)?;
                }
                write!(f,")")
            },
            Expr::Neg(expr) => write!(f,"-{}",expr),
            Expr::Not(expr) => write!(f,"!{}",expr),
            Expr::Mul(expr,expr2) => write!(f,"{} * {}",expr,expr2),
            Expr::Div(expr,expr2) => write!(f,"{} / {}",expr,expr2),
            Expr::Mod(expr,expr2) => write!(f,"{} % {}",expr,expr2),
            Expr::Add(expr,expr2) => write!(f,"{} + {}",expr,expr2),
            Expr::Sub(expr,expr2) => write!(f,"{} - {}",expr,expr2),
            Expr::Shl(expr,expr2) => write!(f,"{} << {}",expr,expr2),
            Expr::Shr(expr,expr2) => write!(f,"{} >> {}",expr,expr2),
            Expr::And(expr,expr2) => write!(f,"{} & {}",expr,expr2),
            Expr::Or(expr,expr2) => write!(f,"{} | {}",expr,expr2),
            Expr::Xor(expr,expr2) => write!(f,"{} ^ {}",expr,expr2),
            Expr::Eq(expr,expr2) => write!(f,"{} == {}",expr,expr2),
            Expr::NotEq(expr,expr2) => write!(f,"{} != {}",expr,expr2),
            Expr::Greater(expr,expr2) => write!(f,"{} > {}",expr,expr2),
            Expr::Less(expr,expr2) => write!(f,"{} < {}",expr,expr2),
            Expr::GreaterEq(expr,expr2) => write!(f,"{} >= {}",expr,expr2),
            Expr::LessEq(expr,expr2) => write!(f,"{} <= {}",expr,expr2),
            Expr::LogAnd(expr,expr2) => write!(f,"{} && {}",expr,expr2),
            Expr::LogOr(expr,expr2) => write!(f,"{} || {}",expr,expr2),
            Expr::Assign(expr,expr2) => write!(f,"{} = {}",expr,expr2),
            Expr::AddAssign(expr,expr2) => write!(f,"{} += {}",expr,expr2),
            Expr::SubAssign(expr,expr2) => write!(f,"{} -= {}",expr,expr2),
            Expr::MulAssign(expr,expr2) => write!(f,"{} *= {}",expr,expr2),
            Expr::DivAssign(expr,expr2) => write!(f,"{} /= {}",expr,expr2),
            Expr::ModAssign(expr,expr2) => write!(f,"{} %= {}",expr,expr2),
            Expr::AndAssign(expr,expr2) => write!(f,"{} &= {}",expr,expr2),
            Expr::OrAssign(expr,expr2) => write!(f,"{} |= {}",expr,expr2),
            Expr::XorAssign(expr,expr2) => write!(f,"{} ^= {}",expr,expr2),
            Expr::ShlAssign(expr,expr2) => write!(f,"{} <<= {}",expr,expr2),
            Expr::ShrAssign(expr,expr2) => write!(f,"{} >>= {}",expr,expr2),
            Expr::Continue => write!(f,"continue"),
            Expr::Break(expr) => if let Some(expr) = expr { write!(f,"break {}",expr) } else { write!(f,"break") },
            Expr::Return(expr) => if let Some(expr) = expr { write!(f,"return {}",expr) } else { write!(f,"return") },
            Expr::Block(block) => write!(f,"{}",block),
            Expr::If(expr,block,else_expr) => {
                write!(f,"if {} {}",expr,block)?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            Expr::IfLet(pats,expr,block,else_expr) => {
                write!(f,"if let ")?;
                for pat in pats {
                    write!(f,"| {} ",pat)?;
                }
                write!(f,"in {} {}",expr,block)?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            Expr::Loop(block) => write!(f,"loop {}",block),
            Expr::For(pats,range,block) => {
                write!(f,"for ")?;
                for pat in pats {
                    write!(f,"| {} ",pat)?;
                }
                write!(f,"in {} {}",range,block)
            },
            Expr::While(expr,block) => write!(f,"while {} {}",expr,block),
            Expr::WhileLet(pats,expr,block) => {
                write!(f,"while let ")?;
                for pat in pats {
                    write!(f,"| {} ",pat)?;
                }
                write!(f,"in {} {}",expr,block)
            },
            Expr::Match(expr,arms) => {
                write!(f,"match {} {{ ",expr)?;
                for (pats,if_expr,expr) in arms {
                    for pat in pats {
                        write!(f,"| {} ",pat)?;
                    }
                    if let Some(if_expr) = if_expr {
                        write!(f,"if {} ",if_expr)?;
                    }
                    write!(f,"=> {},",expr)?;
                }
                write!(f," }}")
            },
        }
    }
}

impl Display for Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Stat::Let(pat,type_,expr) => write!(f,"let {}: {} = {};",pat,type_,expr),
            Stat::Expr(expr) => write!(f,"{};",expr),
        }
    }
}

impl Display for Variant {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Variant::Naked(ident) => write!(f,"{}",ident),
            Variant::Tuple(ident,types) => {
                write!(f,"{}(",ident)?;
                for ty in types {
                    write!(f,"{},",ty)?;
                }
                write!(f,")")
            },
            Variant::Struct(ident,fields) => {
                write!(f,"{} {{ ",ident)?;
                for field in fields.iter() {
                    write!(f,"{}: {},",field.ident,field.type_)?;
                }
                write!(f," }}")
            },
        }
    }
}

impl Display for Module {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"mod {} {{ ",self.ident)?;
        for (_,const_) in self.consts.iter() {
            write!(f,"const {}: {} = {}; ",const_.ident,const_.type_,const_.value.as_ref().unwrap())?;
        }
        for (_,struct_) in self.structs.iter() {
            write!(f,"struct {} {{ ",struct_.ident)?;
            for field in struct_.fields.iter() {
                write!(f,"{}: {},",field.ident,field.type_)?;
            }
            write!(f," }}; ")?;
        }
        for (_,enum_) in self.enums.iter() {
            write!(f,"enum {} {{ ",enum_.ident)?;
            for variant in enum_.variants.iter() {
                write!(f,"{},",variant)?;
            }
            write!(f," }}; ")?;
        }
        for (_,function) in self.functions.iter() {
            write!(f,"fn {}(",function.ident)?;
            for param in function.params.iter() {
                write!(f,"{}: {},",param.ident,param.type_)?;
            }
            write!(f,") ")?;
            if let Type::Void = function.return_type { } else {
                write!(f,"-> {} ",function.return_type)?;
            }
            write!(f,"{} ",function.block)?;
        }
        write!(f,"}}")
    }
}
