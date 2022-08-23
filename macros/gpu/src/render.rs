use {
    crate::*,    
};

fn render_type(ty: sr::Type) -> String {
    match ty {
        sr::Type::Array(ty,expr) => format!("sr::Type::Array({},{})",render_type(*ty),render_expr(*expr)),
        sr::Type::Tuple(types) => {
            let mut r = format!("sr::Type::Tuple(vec![");
            let mut first_type = true;
            for ty in types {
                if !first_type {
                    r += ",";
                }
                r += &render_type(ty);
                first_type = false;
            }
            r += "])";
            r
        },
        sr::Type::Symbol(symbol) => format!("sr::Type::Symbol(\"{}\".to_string())",symbol),
        sr::Type::Inferred => "sr::Type::Inferred".to_string(),
    }
}

fn render_pat(pat: sr::Pat) -> String {
    match pat {
        sr::Pat::Wildcard => "sr::Pat::Wildcard".to_string(),
        sr::Pat::Rest => "sr::Pat::Rest".to_string(),
        sr::Pat::Literal(literal) => format!("sr::Pat::Literal(\"{}\".to_string())",literal),
        sr::Pat::Slice(pats) => {
            let mut r = "sr::Pat::Slice(vec![".to_string();
            let mut first_pat = true;
            for pat in pats {
                if !first_pat {
                    r += ",";
                }
                r += &render_pat(pat);
                first_pat = false;
            }
            r += "])";
            r
        },
        sr::Pat::Symbol(symbol) => format!("sr::Pat::Symbol(\"{}\".to_string())",symbol),
    }
}

fn render_expr(expr: sr::Expr) -> String {
    match expr {
        sr::Expr::Literal(literal) => format!("sr::Expr::Literal(\"{}\".to_string())",literal),
        sr::Expr::Symbol(symbol) => format!("sr::Expr::Symbol(\"{}\".to_string())",symbol),
        sr::Expr::AnonArray(exprs) => {
            let mut r = "sr::Expr::AnonArray(vec![".to_string();
            let mut first_expr = true;
            for expr in exprs {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += "])";
            r
        },
        sr::Expr::AnonTuple(exprs) => {
            let mut r = "sr::Expr::AnonTuple(vec![".to_string();
            let mut first_expr = true;
            for expr in exprs {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += "])";
            r
        },
        sr::Expr::AnonCloned(expr,expr2) => format!("sr::Expr::AnonCloned(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Struct(symbol,fields) => {
            let mut r = format!("sr::Expr::Struct(\"{}\".to_string(),vec![",symbol);
            let mut first_field = true;
            for (symbol,expr) in fields {
                if !first_field {
                    r += ",";
                }
                r += &format!("(\"{}\".to_string(),{})",symbol,render_expr(expr));
                first_field = false;
            }
            r += "])";
            r
        },
        sr::Expr::Tuple(symbol,exprs) => {
            let mut r = format!("sr::Expr::Tuple(\"{}\".to_string(),vec![",symbol);
            let mut first_expr = true;
            for expr in exprs {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += "])";
            r
        },
        sr::Expr::Field(expr,symbol) => format!("sr::Expr::Field(Box::new({}),\"{}\".to_string())",render_expr(*expr),symbol),
        sr::Expr::Index(expr,expr2) => format!("sr::Expr::Index(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Call(expr,exprs) => {
            let mut r = format!("sr::Expr::Call(Box::new({}),vec![",render_expr(*expr));
            let mut first_expr = true;
            for expr in exprs {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += "])";
            r
        },
        sr::Expr::Error(expr) => format!("sr::Expr::Error(Box::new({}))",render_expr(*expr)),
        sr::Expr::Cast(expr,ty) => format!("sr::Expr::Cast(Box::new({}),Box::new({}))",render_expr(*expr),render_type(*ty)),
        sr::Expr::Neg(expr) => format!("sr::Expr::Neg(Box::new({}))",render_expr(*expr)),
        sr::Expr::Not(expr) => format!("sr::Expr::Not(Box::new({}))",render_expr(*expr)),
        sr::Expr::Mul(expr,expr2) => format!("sr::Expr::Mul(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Div(expr,expr2) => format!("sr::Expr::Div(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Mod(expr,expr2) => format!("sr::Expr::Mod(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Add(expr,expr2) => format!("sr::Expr::Add(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Sub(expr,expr2) => format!("sr::Expr::Sub(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Shl(expr,expr2) => format!("sr::Expr::Shl(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Shr(expr,expr2) => format!("sr::Expr::Shr(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::And(expr,expr2) => format!("sr::Expr::And(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Xor(expr,expr2) => format!("sr::Expr::Xor(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Or(expr,expr2) => format!("sr::Expr::Or(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Eq(expr,expr2) => format!("sr::Expr::Eq(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::NotEq(expr,expr2) => format!("sr::Expr::NotEq(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Gt(expr,expr2) => format!("sr::Expr::Gt(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::NotGt(expr,expr2) => format!("sr::Expr::NotGt(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Lt(expr,expr2) => format!("sr::Expr::Lt(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::NotLt(expr,expr2) => format!("sr::Expr::NotLt(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::LogAnd(expr,expr2) => format!("sr::Expr::LogAnd(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::LogOr(expr,expr2) => format!("sr::Expr::LogOr(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Assign(expr,expr2) => format!("sr::Expr::Assign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::AddAssign(expr,expr2) => format!("sr::Expr::AddAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::SubAssign(expr,expr2) => format!("sr::Expr::SubAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::MulAssign(expr,expr2) => format!("sr::Expr::MulAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::DivAssign(expr,expr2) => format!("sr::Expr::DivAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::ModAssign(expr,expr2) => format!("sr::Expr::ModAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::AndAssign(expr,expr2) => format!("sr::Expr::AndAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::XorAssign(expr,expr2) => format!("sr::Expr::XorAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::OrAssign(expr,expr2) => format!("sr::Expr::OrAssign(Box::new({}),Box::new({}))",render_expr(*expr),render_expr(*expr2)),
        sr::Expr::Block(stats) => {
            let mut r = "sr::Expr::Block(vec![".to_string();
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "])";
            r
        },
        sr::Expr::Continue => "sr::Expr::Continue".to_string(),
        sr::Expr::Break(expr) => {
            let mut r = "sr::Expr::Break(".to_string();
            if let Some(expr) = expr {
                r += &format!("Some(Box::new({}))",render_expr(*expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        sr::Expr::Return(expr) => {
            let mut r = "sr::Expr::Return(".to_string();
            if let Some(expr) = expr {
                r += &format!("Some(Box::new({}))",render_expr(*expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        sr::Expr::Loop(stats) => {
            let mut r = "sr::Expr::Loop(vec![".to_string();
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "])";
            r
        },
        sr::Expr::For(pat,expr,stats) => {
            let mut r = format!("sr::Expr::For({},Box::new({}),vec![",render_pat(pat),render_expr(*expr));
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "])";
            r
        },
        sr::Expr::If(expr,stats,else_expr) => {
            let mut r = format!("sr::Expr::If(Box::new({}),vec![",render_expr(*expr));
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "],";
            if let Some(expr) = else_expr {
                r += &format!("Some(Box::new({}))",render_expr(*expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        sr::Expr::IfLet(pats,expr,stats,else_expr) => {
            let mut r = "sr::Expr::IfLet(vec![".to_string();
            let mut first_pat = true;
            for pat in pats {
                if !first_pat {
                    r += ",";
                }
                r += &render_pat(pat);
                first_pat = false;
            }
            r += &format!("],Box::new({}),vec![",render_expr(*expr));
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "],";
            if let Some(expr) = else_expr {
                r += &format!("Some(Box::new({}))",render_expr(*expr));
            }
            else {
                r += "None";
            }
            r += ")";
            r
        },
        sr::Expr::While(expr,stats) => {
            let mut r = format!("sr::Expr::While(Box::new({}),vec![",render_expr(*expr));
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "])";
            r
        },
        sr::Expr::WhileLet(pats,expr,stats) => {
            let mut r = "sr::Expr::WhileLet(vec![".to_string();
            let mut first_pat = true;
            for pat in pats {
                if !first_pat {
                    r += ",";
                }
                r += &render_pat(pat);
                first_pat = false;
            }
            r += &format!("],Box::new({}),vec![",render_expr(*expr));
            let mut first_stat = true;
            for stat in stats {
                if !first_stat {
                    r += ",";
                }
                r += &render_stat(stat);
                first_stat = false;
            }
            r += "])";
            r
        },
        sr::Expr::Match(expr,arms) => {
            let mut r = format!("sr::Expr::Match(Box::new({}),vec![",render_expr(*expr));
            let mut first_arm = true;
            for (pats,is_expr,expr) in arms {
                if !first_arm {
                    r += ",";
                }
                r += "(vec![";
                let mut first_pat = true;
                for pat in pats {
                    if !first_pat {
                        r += ",";
                    }
                    r += &render_pat(pat);
                    first_pat = false;
                }
                r += "],";
                if let Some(expr) = is_expr {
                    r += &format!("Some(Box::new({}))",render_expr(*expr));
                }
                else {
                    r += "None";
                }
                r += &format!(",Box::new({}))",render_expr(*expr));
                first_arm = false;
            }
            r += "])";
            r
        },
    }
}

fn render_stat(stat: sr::Stat) -> String {
    match stat {
        sr::Stat::Expr(expr) => format!("sr::Stat::Expr(Box::new({}))",render_expr(*expr)),
        sr::Stat::Let(pat,ty,expr) => {
            let mut r = format!("sr::Stat::Let({}",render_pat(pat));
            if let Some(ty) = ty {
                r += &format!(",Some(Box::new({}))",render_type(*ty));
            }
            else {
                r += ",None";
            }
            if let Some(expr) = expr {
                r += &format!(",Some(Box::new({}))",render_expr(*expr));
            }
            else {
                r += ",None";
            }
            r += ")";
            r
        },
    }
}

fn render_function(symbol: String,params: Vec<(sr::Pat,Box<sr::Type>)>,return_ty: Option<Box<sr::Type>>,stats: Vec<sr::Stat>,) -> String {
    let mut r = String::new();
    r += &format!("sr::Item::Function(\"{}\".to_string(),vec![",symbol);
    let mut first_param = true;
    for (pat,ty) in params {
        if !first_param {
            r += ",";
        }
        r += &format!("({},Box::new({}))",&render_pat(pat),&render_type(*ty));
        first_param = false;
    }
    r += "],";
    if let Some(return_ty) = return_ty {
        r += &format!("Some(Box::new({})),",&render_type(*return_ty));
    }
    else {
        r += "None,";
    }
    r += "vec![";
    let mut first_stat = true;
    for stat in stats {
        if !first_stat {
            r += ",";
        }
        r += &render_stat(stat);
        first_stat = false;
    }
    r += "])";
    r
}

pub(crate) fn render_vertex_trait(item: sr::Item) -> String {
    if let sr::Item::Struct(symbol,fields) = item {
        let mut types: Vec<sr::BaseType> = Vec::new();
        for (_,ty) in fields {
            if let sr::Type::Symbol(symbol) = *ty {
                types.push(match symbol.as_str() {
                    "u8" => sr::BaseType::U8,
                    "u16" => sr::BaseType::U16,
                    "u32" => sr::BaseType::U32,
                    "u64" => sr::BaseType::U64,
                    "i8" => sr::BaseType::I8,
                    "i16" => sr::BaseType::I16,
                    "i32" => sr::BaseType::I32,
                    "i64" => sr::BaseType::I64,
                    "f16" => sr::BaseType::F16,
                    "f32" => sr::BaseType::F32,
                    "f64" => sr::BaseType::F64,
                    "Vec2<u8>" => sr::BaseType::Vec2U8,
                    "Vec2<u16>" => sr::BaseType::Vec2U16,
                    "Vec2<u32>" => sr::BaseType::Vec2U32,
                    "Vec2<u64>" => sr::BaseType::Vec2U64,
                    "Vec2<i8>" => sr::BaseType::Vec2I8,
                    "Vec2<i16>" => sr::BaseType::Vec2I16,
                    "Vec2<i32>" => sr::BaseType::Vec2I32,
                    "Vec2<i64>" => sr::BaseType::Vec2I64,
                    "Vec2<f16>" => sr::BaseType::Vec2F16,
                    "Vec2<f32>" => sr::BaseType::Vec2F32,
                    "Vec2<f64>" => sr::BaseType::Vec2F64,
                    "Vec3<u8>" => sr::BaseType::Vec3U8,
                    "Vec3<u16>" => sr::BaseType::Vec3U16,
                    "Vec3<u32>" => sr::BaseType::Vec3U32,
                    "Vec3<u64>" => sr::BaseType::Vec3U64,
                    "Vec3<i8>" => sr::BaseType::Vec3I8,
                    "Vec3<i16>" => sr::BaseType::Vec3I16,
                    "Vec3<i32>" => sr::BaseType::Vec3I32,
                    "Vec3<i64>" => sr::BaseType::Vec3I64,
                    "Vec3<f16>" => sr::BaseType::Vec3F16,
                    "Vec3<f32>" => sr::BaseType::Vec3F32,
                    "Vec3<f64>" => sr::BaseType::Vec3F64,
                    "Vec4<u8>" => sr::BaseType::Vec4U8,
                    "Vec4<u16>" => sr::BaseType::Vec4U16,
                    "Vec4<u32>" => sr::BaseType::Vec4U32,
                    "Vec4<u64>" => sr::BaseType::Vec4U64,
                    "Vec4<i8>" => sr::BaseType::Vec4I8,
                    "Vec4<i16>" => sr::BaseType::Vec4I16,
                    "Vec4<i32>" => sr::BaseType::Vec4I32,
                    "Vec4<i64>" => sr::BaseType::Vec4I64,
                    "Vec4<f16>" => sr::BaseType::Vec4F16,
                    "Vec4<f32>" => sr::BaseType::Vec4F32,
                    "Vec4<f64>" => sr::BaseType::Vec4F64,
                    "Color<u8>" => sr::BaseType::ColorU8,
                    "Color<u16>" => sr::BaseType::ColorU16,
                    "Color<f16>" => sr::BaseType::ColorF16,
                    "Color<f32>" => sr::BaseType::ColorF32,
                    "Color<f64>" => sr::BaseType::ColorF64,
                    _ => panic!("ony base types allowed (not {})",symbol),
                });
            }
            else {
                panic!("only base types allowed (not {})",ty);
            }
        }
        let mut r = format!("impl Vertex for {} {{ fn get_types() -> Vec<BaseType> {{ vec![",symbol);
        let mut first_type = true;
        for ty in types {
            if !first_type {
                r += ",";
            }
            r += &format!("BaseType::{}",sr::base_type_variant(&ty));
            first_type = false;
        }
        r += "] } }";
        r
    }
    else {
        panic!("Vertex can only be derived from structs");
    }
}
