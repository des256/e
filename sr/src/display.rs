use {
    crate::*,
    std::fmt::{
        Display,
        Formatter,
        Result,
    },
};

impl Display for Expr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Expr::Literal(literal) => write!(f,"{}",literal),
            Expr::Symbol(symbol) => write!(f,"{}",symbol),
            Expr::AnonArray(exprs) => {
                write!(f,"[")?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,"]")
            },
            Expr::AnonTuple(exprs) => {
                write!(f,"(")?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            Expr::AnonCloned(expr,expr2) => write!(f,"[{}; {}]",expr,expr2),
            Expr::Struct(symbol,fields) => {
                write!(f,"{} {{ ",symbol)?;
                let mut first_field = true;
                for (symbol,expr) in fields {
                    if !first_field {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",symbol,expr)?;
                    first_field = false;
                }
                write!(f," }}")
            },
            Expr::Tuple(symbol,exprs) => {
                write!(f,"{}(",symbol)?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            Expr::Field(expr,field) => write!(f,"{}.{}",expr,field),
            Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            Expr::Call(expr,exprs) => {
                write!(f,"{}(",expr)?;
                let mut first_expr = true;
                for expr in exprs {
                    if !first_expr {
                        write!(f,",")?;
                    }
                    write!(f,"{}",expr)?;
                    first_expr = false;
                }
                write!(f,")")
            },
            Expr::Error(expr) => write!(f,"{}?",expr),
            Expr::Cast(expr,ty) => write!(f,"{} as {}",expr,ty),
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
            Expr::Xor(expr,expr2) => write!(f,"{} ^ {}",expr,expr2),
            Expr::Or(expr,expr2) => write!(f,"{} | {}",expr,expr2),
            Expr::Eq(expr,expr2) => write!(f,"{} == {}",expr,expr2),
            Expr::NotEq(expr,expr2) => write!(f,"{} != {}",expr,expr2),
            Expr::Gt(expr,expr2) => write!(f,"{} > {}",expr,expr2),
            Expr::NotGt(expr,expr2) => write!(f,"{} <= {}",expr,expr2),
            Expr::Lt(expr,expr2) => write!(f,"{} < {}",expr,expr2),
            Expr::NotLt(expr,expr2) => write!(f,"{} >= {}",expr,expr2),
            Expr::LogAnd(expr,expr2) => write!(f,"{} && {}",expr,expr2),
            Expr::LogOr(expr,expr2) => write!(f,"{} || {}",expr,expr2),
            Expr::Assign(expr,expr2) => write!(f,"{} = {}",expr,expr2),
            Expr::AddAssign(expr,expr2) => write!(f,"{} += {}",expr,expr2),
            Expr::SubAssign(expr,expr2) => write!(f,"{} -= {}",expr,expr2),
            Expr::MulAssign(expr,expr2) => write!(f,"{} *= {}",expr,expr2),
            Expr::DivAssign(expr,expr2) => write!(f,"{} /= {}",expr,expr2),
            Expr::ModAssign(expr,expr2) => write!(f,"{} %= {}",expr,expr2),
            Expr::AndAssign(expr,expr2) => write!(f,"{} &= {}",expr,expr2),
            Expr::XorAssign(expr,expr2) => write!(f,"{} ^= {}",expr,expr2),
            Expr::OrAssign(expr,expr2) => write!(f,"{} |= {}",expr,expr2),
            Expr::Block(stats) => {
                write!(f,"{{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            Expr::Continue => write!(f,"continue"),
            Expr::Break(expr) => {
                write!(f,"break")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            Expr::Return(expr) => {
                write!(f,"return")?;
                if let Some(expr) = expr {
                    write!(f," {}",expr)?;
                }
                write!(f,"")
            },
            Expr::Loop(stats) => {
                write!(f,"loop {{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            Expr::For(pat,expr,stats) => {
                write!(f,"for {} in {} {{ ",pat,expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            Expr::If(expr,stats,else_expr) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            Expr::IfLet(pats,expr,stats,else_expr) => {
                write!(f,"if let ")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first_pat = false;
                }
                write!(f," = {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")?;
                if let Some(else_expr) = else_expr {
                    write!(f," else {}",else_expr)?;
                }
                write!(f,"")
            },
            Expr::While(expr,stats) => {
                write!(f,"if {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            Expr::WhileLet(pats,expr,stats) => {
                write!(f,"while let ")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,"| ")?;
                    }
                    write!(f,"{} ",pat)?;
                    first_pat = false;
                }
                write!(f," = {} {{ ",expr)?;
                for stat in stats {
                    write!(f,"{}; ",stat)?;
                }
                write!(f,"}}")
            },
            Expr::Match(expr,arms) => {
                write!(f,"match {} {{ ",expr)?;
                for (pats,if_expr,expr) in arms {
                    let mut first_pat = true;
                    for pat in pats {
                        if !first_pat {
                            write!(f,"| ")?;
                        }
                        write!(f,"{} ",pat)?;
                        first_pat = false;
                    }
                    if let Some(if_expr) = if_expr {
                        write!(f,"if {} ",if_expr)?;
                    }
                    write!(f,"=> {},",expr)?;
                }
                write!(f,"}}")
            },
        }
    }
}

impl Display for Item {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Item::Module(symbol,items) => {
                write!(f,"mod {} {{ ",symbol)?;
                for item in items {
                    write!(f,"{} ",item)?;
                }
                write!(f,"}}")
            },
            Item::Function(symbol,params,return_ty,stats) => {
                write!(f,"fn {}(",symbol)?;
                let mut first_param = true;
                for (pat,ty) in params {
                    if !first_param {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",pat,ty)?;
                    first_param = false;
                }
                write!(f,") ")?;
                if let Some(ty) = return_ty {
                    write!(f,"-> {} ",ty)?;
                }
                write!(f,"{{ ")?;
                for stat in stats {
                    write!(f,"{} ",stat)?;
                }
                write!(f,"}}")
            },
            Item::Struct(symbol,fields) => {
                write!(f,"struct {} {{",symbol)?;
                let mut first_field = true;
                for (symbol,ty) in fields {
                    if !first_field {
                        write!(f,",")?;
                    }
                    write!(f,"{}: {}",symbol,ty)?;
                    first_field = false;
                }
                write!(f,"}}")
            },
        }
    }
}

impl Display for Pat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Pat::Wildcard => write!(f,"_"),
            Pat::Rest => write!(f,".."),
            Pat::Literal(literal) => write!(f,"{}",literal),
            Pat::Slice(pats) => {
                write!(f,"[")?;
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        write!(f,",")?;
                    }
                    write!(f,"{}",pat)?;
                    first_pat = false;
                }
                write!(f,"]")
            },
            Pat::Symbol(symbol) => write!(f,"{}",symbol),
        }
    }
}

impl Display for Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Stat::Let(pat,ty,expr) => {
                write!(f,"let {}",pat)?;
                if let Some(ty) = ty {
                    write!(f,": {}",ty)?;
                }
                if let Some(expr) = expr {
                    write!(f," = {}",expr)?;
                }
                write!(f,";")
            },
            Stat::Expr(expr) => write!(f,"{}",expr),
        }
    }
}

impl Display for Type {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Type::Array(ty,expr) => write!(f,"{}[{}]",ty,expr),
            Type::Tuple(types) => {
                write!(f,"(")?;
                let mut first_type = true;
                for ty in types {
                    if !first_type {
                        write!(f,",")?;
                    }
                    write!(f,"{}",ty)?;
                    first_type = false;
                }
                write!(f,")")
            },
            Type::Symbol(symbol) => write!(f,"{}",symbol),
            Type::Inferred => write!(f,"_"),
        }
    }
}
