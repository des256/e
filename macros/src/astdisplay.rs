use {
    super::*,
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
            Type::Bool => write!(f,"bool"),
            Type::U8 => write!(f,"u8"),
            Type::I8 => write!(f,"i8"),
            Type::U16 => write!(f,"u16"),
            Type::I16 => write!(f,"i16"),
            Type::U32 => write!(f,"u32"),
            Type::I32 => write!(f,"i32"),
            Type::U64 => write!(f,"u64"),
            Type::I64 => write!(f,"i64"),
            Type::F16 => write!(f,"f16"),
            Type::F32 => write!(f,"f32"),
            Type::F64 => write!(f,"f64"),
            Type::Vec2Bool => write!(f,"Vec2<bool>"),
            Type::Vec2U8 => write!(f,"Vec2<u8>"),
            Type::Vec2I8 => write!(f,"Vec2<i8>"),
            Type::Vec2U16 => write!(f,"Vec2<u16>"),
            Type::Vec2I16 => write!(f,"Vec2<i16>"),
            Type::Vec2U32 => write!(f,"Vec2<u32>"),
            Type::Vec2I32 => write!(f,"Vec2<i32>"),
            Type::Vec2U64 => write!(f,"Vec2<u64>"),
            Type::Vec2I64 => write!(f,"Vec2<i64>"),
            Type::Vec2F16 => write!(f,"Vec2<f16>"),
            Type::Vec2F32 => write!(f,"Vec2<f32>"),
            Type::Vec2F64 => write!(f,"Vec2<f64>"),
            Type::Vec3Bool => write!(f,"Vec3<bool>"),
            Type::Vec3U8 => write!(f,"Vec3<u8>"),
            Type::Vec3I8 => write!(f,"Vec3<i8>"),
            Type::Vec3U16 => write!(f,"Vec3<u16>"),
            Type::Vec3I16 => write!(f,"Vec3<i16>"),
            Type::Vec3U32 => write!(f,"Vec3<u32>"),
            Type::Vec3I32 => write!(f,"Vec3<i32>"),
            Type::Vec3U64 => write!(f,"Vec3<u64>"),
            Type::Vec3I64 => write!(f,"Vec3<i64>"),
            Type::Vec3F16 => write!(f,"Vec3<f16>"),
            Type::Vec3F32 => write!(f,"Vec3<f32>"),
            Type::Vec3F64 => write!(f,"Vec3<f64>"),
            Type::Vec4Bool => write!(f,"Vec3<bool>"),
            Type::Vec4U8 => write!(f,"Vec4<u8>"),
            Type::Vec4I8 => write!(f,"Vec4<i8>"),
            Type::Vec4U16 => write!(f,"Vec4<u16>"),
            Type::Vec4I16 => write!(f,"Vec4<i16>"),
            Type::Vec4U32 => write!(f,"Vec4<u32>"),
            Type::Vec4I32 => write!(f,"Vec4<i32>"),
            Type::Vec4U64 => write!(f,"Vec4<u64>"),
            Type::Vec4I64 => write!(f,"Vec4<i64>"),
            Type::Vec4F16 => write!(f,"Vec4<f16>"),
            Type::Vec4F32 => write!(f,"Vec4<f32>"),
            Type::Vec4F64 => write!(f,"Vec4<f64>"),
            Type::Mat2x2F32 => write!(f,"Mat2x2<f32>"),
            Type::Mat2x2F64 => write!(f,"Mat2x2<f64>"),
            Type::Mat2x3F32 => write!(f,"Mat2x3<f32>"),
            Type::Mat2x3F64 => write!(f,"Mat2x3<f64>"),
            Type::Mat2x4F32 => write!(f,"Mat2x4<f32>"),
            Type::Mat2x4F64 => write!(f,"Mat2x4<f64>"),
            Type::Mat3x2F32 => write!(f,"Mat3x2<f32>"),
            Type::Mat3x2F64 => write!(f,"Mat3x2<f64>"),
            Type::Mat3x3F32 => write!(f,"Mat3x3<f32>"),
            Type::Mat3x3F64 => write!(f,"Mat3x3<f64>"),
            Type::Mat3x4F32 => write!(f,"Mat3x4<f32>"),
            Type::Mat3x4F64 => write!(f,"Mat3x4<f64>"),
            Type::Mat4x2F32 => write!(f,"Mat4x2<f32>"),
            Type::Mat4x2F64 => write!(f,"Mat4x2<f64>"),
            Type::Mat4x3F32 => write!(f,"Mat4x3<f32>"),
            Type::Mat4x3F64 => write!(f,"Mat4x3<f64>"),
            Type::Mat4x4F32 => write!(f,"Mat4x4<f32>"),
            Type::Mat4x4F64 => write!(f,"Mat4x4<f64>"),        
            Type::AnonTuple(types) => {
                write!(f,"(")?;
                let mut iter = types.iter();
                if let Some(type_) = iter.next() {
                    write!(f,"{}",type_)?;
                }
                for type_ in iter {
                    write!(f,",{}",type_)?;
                }
                write!(f,")")
            },
            Type::Array(type_,count) => write!(f,"[{}; {}]",type_,count),
            Type::Ident(ident) => write!(f,"{}",ident),
        }
    }
}

impl Display for FieldPat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            FieldPat::Wildcard => write!(f,"_"),
            FieldPat::Rest => write!(f,".."),
            FieldPat::Ident(ident) => write!(f,"{}",ident),
            FieldPat::IdentPat(ident,pat) => write!(f,"{}: {}",ident,pat),
        }
    }
}

impl Display for VariantPat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            VariantPat::Naked => write!(f,""),
            VariantPat::Tuple(pats) => {
                write!(f,"(")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f,",{}",pat)?;
                    }
                }
                write!(f,")")
            },
            VariantPat::Struct(field_pats) => {
                write!(f," {{ ")?;
                let mut iter = field_pats.iter();
                if let Some(field_pat) = iter.next() {
                    write!(f,"{}",field_pat)?;
                    for field_pat in iter {
                        write!(f,",{}",field_pat)?;
                    }
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
            Pat::AnonTuple(pats) => {
                write!(f,"(")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f,",{}",pat)?;
                    }
                }
                write!(f,")")
            },
            Pat::Array(pats) => {
                write!(f,"[")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f,",{}",pat)?;
                    }
                }
                write!(f,"]")
            },
            Pat::Range(from,to) => write!(f,"{} ..= {}",from,to),
            Pat::Ident(ident) => write!(f,"{}",ident),
            Pat::Tuple(tuple_ident,pats) => {
                write!(f,"{}(",tuple_ident)?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f,",{}",pat)?;
                    }
                }
                write!(f,")")
            },
            Pat::Struct(struct_ident,field_pats) => {
                write!(f,"{} {{ ",struct_ident)?;
                let mut iter = field_pats.iter();
                if let Some(field_pat) = iter.next() {
                    write!(f,"{}",field_pat)?;
                    for field_pat in iter {
                        write!(f,",{}",field_pat)?;
                    }
                }
                write!(f," }}")
            },
            Pat::Variant(enum_ident,variant_ident,variant_pat) => {
                write!(f,"{}::{}",enum_ident,variant_ident)?;
                write!(f,"{}",variant_pat)
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

impl Display for VariantExpr {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            VariantExpr::Naked => write!(f,""),
            VariantExpr::Tuple(exprs) => {
                write!(f,"(")?;
                let mut iter = exprs.iter();
                if let Some(expr) = iter.next() {
                    write!(f,"{}",expr)?;
                    for expr in iter {
                        write!(f,",{}",expr)?;
                    }
                }
                write!(f,")")
            },
            VariantExpr::Struct(fields) => {
                write!(f," {{ ")?;
                let mut iter = fields.iter();
                if let Some(field) = iter.next() {
                    write!(f,"{}: {}",field.0,field.1)?;
                    for field in iter {
                        write!(f,",{}: {}",field.0,field.1)?;
                    }
                }
                write!(f," }}")
            },
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
                let mut iter = exprs.iter();
                if let Some(expr) = iter.next() {
                    write!(f,"{}",expr)?;
                    for expr in iter {
                        write!(f,",{}",expr)?;
                    }
                }
                write!(f,"]")
            },
            Expr::Cloned(expr,expr2) => write!(f,"[{}; {}]",expr,expr2),
            Expr::Index(expr,expr2) => write!(f,"{}[{}]",expr,expr2),
            Expr::Cast(expr,type_) => write!(f,"{} as {}",expr,type_),
            Expr::AnonTuple(exprs) => {
                write!(f,"(")?;
                let mut iter = exprs.iter();
                if let Some(expr) = iter.next() {
                    write!(f,"{}",expr)?;
                    for expr in iter {
                        write!(f,",{}",expr)?;
                    }
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
            Expr::While(expr,block) => write!(f,"while {} {}",expr,block),
            Expr::Loop(block) => write!(f,"loop {}",block),
            Expr::IfLet(pats,expr,block,else_expr) => if let Some(else_expr) = else_expr {
                write!(f,"if let ")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f," | {}",pat)?;
                    }
                }
                write!(f,"= {} {} else {}",expr,block,else_expr)
            }
            else {
                write!(f,"if let ")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f," | {}",pat)?;
                    }
                }
                write!(f,"= {} {}",expr,block)
            },
            Expr::For(pats,range,block) => {
                write!(f,"for ")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f," | {}",pat)?;
                    }
                }
                write!(f,"in {} {}",range,block)
            },
            Expr::WhileLet(pats,expr,block) => {
                write!(f,"while let ")?;
                let mut iter = pats.iter();
                if let Some(pat) = iter.next() {
                    write!(f,"{}",pat)?;
                    for pat in iter {
                        write!(f," | {}",pat)?;
                    }
                }
                write!(f,"= {} {}",expr,block)
            },
            Expr::Match(expr,arms) => {
                write!(f,"match {} {{",expr)?;
                for (pats,if_expr,expr) in arms.iter() {
                    let mut iter = pats.iter();
                    if let Some(pat) = iter.next() {
                        write!(f,"{}",pat)?;
                        for pat in iter {
                            write!(f," | {}",pat)?;
                        }
                    }
                    if let Some(if_expr) = if_expr {
                        write!(f,"if {} ",if_expr)?;
                    }
                    write!(f,"=> {},",expr)?;
                }
                write!(f," }}")
            },
            Expr::Ident(ident) => write!(f,"{}",ident),
            Expr::TupleLitOrFunctionCall(ident,exprs) => {
                write!(f,"{}(",ident)?;
                let mut iter = exprs.iter();
                if let Some(expr) = iter.next() {
                    write!(f,"{}",expr)?;
                    for expr in iter {
                        write!(f,",{}",expr)?;
                    }
                }
                write!(f,")")
            },
            Expr::StructLit(struct_ident,fields) => {
                write!(f,"{} {{ ",struct_ident)?;
                let mut iter = fields.iter();
                if let Some(field) = iter.next() {
                    write!(f,"{}: {}",field.0,field.1)?;
                    for field in iter {
                        write!(f,",{}: {}",field.0,field.1)?;
                    }
                }
                write!(f," }}")
            },
            Expr::Variant(enum_ident,variant_ident,variant_expr) => {
                write!(f,"{}::{}",enum_ident,variant_ident)?;
                write!(f,"{}",variant_expr)
            },
            Expr::MethodCall(expr,method_ident,exprs) => {
                write!(f,"{}.{}(",expr,method_ident)?;
                let mut iter = exprs.iter();
                if let Some(expr) = iter.next() {
                    write!(f,"{}",expr)?;
                    for expr in iter {
                        write!(f,",{}",expr)?;
                    }
                }
                write!(f,")")
            },
            Expr::Field(expr,ident) => write!(f,"{}.{}",expr,ident),
        }
    }
}

impl Display for Stat {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Stat::Expr(expr) => write!(f,"{};",expr),
            Stat::Let(pat,type_,expr) => write!(f,"let {}: {} = {};",pat,type_,expr),
        }
    }
}

impl Display for Method {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"fn {}(self: {}",self.ident,self.from_type)?;
        for param in self.params.iter() {
            write!(f,",{}: {}",param.0,param.1)?;
        }
        write!(f,")")?;
        if let Type::Void = self.return_type { } else {
            write!(f," -> {}",self.return_type)?;
        }
        write!(f,";")
    }
}

impl Display for Function {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"fn {}(",self.ident)?;
        let mut iter = self.params.iter();
        if let Some(param) = iter.next() {
            write!(f,"{}: {}",param.0,param.1)?;
            for param in iter {
                write!(f,",{}: {}",param.0,param.1)?;
            }
        }
        write!(f,")")?;
        if let Type::Void = self.return_type { } else {
            write!(f," -> {}",self.return_type)?;
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
                write!(f,", ")?;
            }
            write!(f,"{}",type_)?;
            first = false;
        }
        write!(f," }}")
    }
}

impl Display for Struct {
    fn fmt(&self,f: &mut Formatter) -> Result { 
        write!(f,"struct {} {{ ",self.ident)?;
        let mut first = true;
        for (ident,type_) in self.fields.iter() {
            if !first {
                write!(f,", ")?;
            }
            write!(f,"{}: {}",ident,type_)?;
            first = false;
        }
        write!(f," }}")
    }
}

impl Display for Variant {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Variant::Naked => write!(f,""),
            Variant::Tuple(types) => {
                let mut iter = types.iter();
                if let Some(type_) = iter.next() {
                    write!(f,"{}",type_)?;
                    for type_ in iter {
                        write!(f,",{}",type_)?;
                    }
                }
                write!(f,")")
            },
            Variant::Struct(fields) => {
                write!(f," {{ ")?;
                let mut iter = fields.iter();
                if let Some(field) = iter.next() {
                    write!(f,"{}: {}",field.0,field.1)?;
                    for field in iter {
                        write!(f,",{}: {}",field.0,field.1)?;
                    }
                }
                write!(f," }}")
            },
        }
    }
}

impl Display for Enum {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"enum {} {{ ",self.ident)?;
        let mut iter = self.variants.iter();
        if let Some(variant) = iter.next() {
            write!(f,"{}{}",variant.0,variant.1)?;
            for variant in iter {
                write!(f,",{}{}",variant.0,variant.1)?;
            }
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
        for tuple in self.tuples.iter() {
            write!(f,"{};\n",tuple)?;
        }
        for struct_ in self.structs.iter() {
            write!(f,"{};\n",struct_)?;
        }
        for struct_ in self.extern_structs.iter() {
            write!(f,"{};\n",struct_)?;
        }
        for enum_ in self.enums.iter() {
            write!(f,"{};\n",enum_)?;
        }
        for alias in self.aliases.iter() {
            write!(f,"{};\n",alias)?;
        }
        for const_ in self.consts.iter() {
            write!(f,"{};\n",const_)?;
        }
        for function in self.functions.iter() {
            write!(f,"{};\n",function)?;
        }
        write!(f,"}}")
    }
}
