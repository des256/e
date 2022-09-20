use {
    crate::*,
};

fn gl_base_type_name(type_: &sr::BaseType) -> &'static str {
    match type_ {
        sr::BaseType::Bool => "bool",
        sr::BaseType::U8 => "uint",
        sr::BaseType::U16 => "uint",
        sr::BaseType::U32 => "uint",
        sr::BaseType::U64 => "uint",
        sr::BaseType::I8 => "int",
        sr::BaseType::I16 => "int",
        sr::BaseType::I32 => "int",
        sr::BaseType::I64 => "int",
        sr::BaseType::F16 => "float",
        sr::BaseType::F32 => "float",
        sr::BaseType::F64 => "double",
        sr::BaseType::Vec2Bool => "bvec2",
        sr::BaseType::Vec2U8 => "uvec2",
        sr::BaseType::Vec2U16 => "uvec2",
        sr::BaseType::Vec2U32 => "uvec2",
        sr::BaseType::Vec2U64 => "uvec2",
        sr::BaseType::Vec2I8 => "ivec2",
        sr::BaseType::Vec2I16 => "ivec2",
        sr::BaseType::Vec2I32 => "ivec2",
        sr::BaseType::Vec2I64 => "ivec2",
        sr::BaseType::Vec2F16 => "vec2",
        sr::BaseType::Vec2F32 => "vec2",
        sr::BaseType::Vec2F64 => "vec2",
        sr::BaseType::Vec3Bool => "bvec3",
        sr::BaseType::Vec3U8 => "uvec3",
        sr::BaseType::Vec3U16 => "uvec3",
        sr::BaseType::Vec3U32 => "uvec3",
        sr::BaseType::Vec3U64 => "uvec3",
        sr::BaseType::Vec3I8 => "ivec3",
        sr::BaseType::Vec3I16 => "ivec3",
        sr::BaseType::Vec3I32 => "ivec3",
        sr::BaseType::Vec3I64 => "ivec3",
        sr::BaseType::Vec3F16 => "vec3",
        sr::BaseType::Vec3F32 => "vec3",
        sr::BaseType::Vec3F64 => "vec3",
        sr::BaseType::Vec4Bool => "bvec4",
        sr::BaseType::Vec4U8 => "uvec4",
        sr::BaseType::Vec4U16 => "uvec4",
        sr::BaseType::Vec4U32 => "uvec4",
        sr::BaseType::Vec4U64 => "uvec4",
        sr::BaseType::Vec4I8 => "ivec4",
        sr::BaseType::Vec4I16 => "ivec4",
        sr::BaseType::Vec4I32 => "ivec4",
        sr::BaseType::Vec4I64 => "ivec4",
        sr::BaseType::Vec4F16 => "vec4",
        sr::BaseType::Vec4F32 => "vec4",
        sr::BaseType::Vec4F64 => "vec4",
        sr::BaseType::ColorU8 => "vec4",
        sr::BaseType::ColorU16 => "vec4",
        sr::BaseType::ColorF16 => "vec4",
        sr::BaseType::ColorF32 => "vec4",
        sr::BaseType::ColorF64 => "vec4",
    }
}

fn render_type(type_: &sr::Type) -> String {
    
}

fn render_stat(stat: &sr::Stat) -> String {
    match stat {
        sr::Stat::Let(ident,type_,expr) => if let Some(expr) = expr { format!("{} {} = {};",render_type(type_),ident,render_expr(expr)) } else { format!("{} {};",render_type(type_),ident) },
        sr::Stat::Expr(expr) => format!("{};",render_expr(expr)),
    }
}

fn render_vec_literal(type_ident: &str,fields: &Vec<(String,sr::Expr)>) -> String {
    let mut r = format!("{}(",type_ident);
    let mut first_field = true;
    for field in fields.iter() {
        if !first_field {
            r += ",";
        }
        r += &render_expr(&field.1);
        first_field = false;
    }
    r += ")";
    r
}

fn render_expr(expr: &sr::Expr) -> String {
    match expr {
        sr::Expr::Boolean(value) => if *value { "true".to_string() } else { "false".to_string() },
        sr::Expr::Integer(value) => format!("{}",value),
        sr::Expr::Float(value) => format!("{}",value),
        sr::Expr::Base(base_type,fields) => {
            match base_type {
                sr::BaseType::Vec2Bool => render_vec_literal("bvec2",fields),
                sr::BaseType::Vec2U8 |
                sr::BaseType::Vec2U16 |
                sr::BaseType::Vec2U32 |
                sr::BaseType::Vec2U64 => render_vec_literal("uvec2",fields),
                sr::BaseType::Vec2I8 |
                sr::BaseType::Vec2I16 |
                sr::BaseType::Vec2I32 |
                sr::BaseType::Vec2I64 => render_vec_literal("ivec2",fields),
                sr::BaseType::Vec2F16 |
                sr::BaseType::Vec2F32 => render_vec_literal("vec2",fields),
                sr::BaseType::Vec2F64 => render_vec_literal("dvec2",fields),
                sr::BaseType::Vec3Bool => render_vec_literal("bvec3",fields),
                sr::BaseType::Vec3U8 |
                sr::BaseType::Vec3U16 |
                sr::BaseType::Vec3U32 |
                sr::BaseType::Vec3U64 => render_vec_literal("uvec3",fields),
                sr::BaseType::Vec3I8 |
                sr::BaseType::Vec3I16 |
                sr::BaseType::Vec3I32 |
                sr::BaseType::Vec3I64 => render_vec_literal("ivec3",fields),
                sr::BaseType::Vec3F16 |
                sr::BaseType::Vec3F32 => render_vec_literal("vec3",fields),
                sr::BaseType::Vec3F64 => render_vec_literal("dvec3",fields),
                sr::BaseType::Vec4Bool => render_vec_literal("bvec4",fields),
                sr::BaseType::Vec4U8 |
                sr::BaseType::Vec4U16 |
                sr::BaseType::Vec4U32 |
                sr::BaseType::Vec4U64 => render_vec_literal("uvec4",fields),
                sr::BaseType::Vec4I8 |
                sr::BaseType::Vec4I16 |
                sr::BaseType::Vec4I32 |
                sr::BaseType::Vec4I64 => render_vec_literal("ivec4",fields),
                sr::BaseType::Vec4F16 |
                sr::BaseType::Vec4F32 => render_vec_literal("vec4",fields),
                sr::BaseType::Vec4F64 => render_vec_literal("dvec4",fields),
                sr::BaseType::ColorU8 |
                sr::BaseType::ColorU16 |
                sr::BaseType::ColorF16 |
                sr::BaseType::ColorF32 => render_vec_literal("vec4",fields),
                sr::BaseType::ColorF64 => render_vec_literal("dvec4",fields),
                _ => panic!("ERROR: small basetype literals don't show up directly"),
            }
        },
        sr::Expr::Local(variable) |
        sr::Expr::Param(variable) |
        sr::Expr::Const(variable) => variable.ident.clone(),
        sr::Expr::Array(exprs) => {
            let mut r = format!("TODO[{}](",exprs.len());
            let mut first_expr = false;
            for expr in exprs.iter() {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += ")";
            r
        },
        sr::Expr::Cloned(expr,expr2) => format!("TODO temp = {}; TODO[{}](TODO)",expr,expr2),
        sr::Expr::Struct(struct_,fields) => {
            let mut r = format!("{}(",struct_.ident);
            let mut first_field = false;
            for field in fields.iter() {
                if !first_field {
                    r += ",";
                }
                r += &render_expr(&field.1);
                first_field = false;
            }
            r += ")";
            r
        },
        sr::Expr::Call(function,exprs) => {
            let mut r = format!("{}(",function.ident);
            let mut first_expr = false;
            for expr in exprs.iter() {
                if !first_expr {
                    r += ",";
                }
                r += &render_expr(expr);
                first_expr = false;
            }
            r += ")";
            r
        },
        sr::Expr::Field(expr,ident) => format!("{}.{}",render_expr(expr),ident),
        sr::Expr::Index(expr,expr2) => format!("({}[{}])",render_expr(expr),render_expr(expr2)),
        sr::Expr::Cast(expr,type_) => format!("(({}){})",render_type(type_),render_expr(expr)),
        sr::Expr::Neg(expr) => format!("(-{})",render_expr(expr)),
        sr::Expr::Not(expr) => format!("(!{})",render_expr(expr)),
        sr::Expr::Mul(expr,expr2) => format!("({}*{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Div(expr,expr2) => format!("({}/{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Mod(expr,expr2) => format!("({}%{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Add(expr,expr2) => format!("({}+{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Sub(expr,expr2) => format!("({}-{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Shl(expr,expr2) => format!("({}<<{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Shr(expr,expr2) => format!("({}>>{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::And(expr,expr2) => format!("({}&{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Or(expr,expr2) => format!("({}|{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Xor(expr,expr2) => format!("({}^{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Eq(expr,expr2) => format!("({}=={})",render_expr(expr),render_expr(expr2)),
        sr::Expr::NotEq(expr,expr2) => format!("({}!={})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Greater(expr,expr2) => format!("({}>{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Less(expr,expr2) => format!("({}<{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::GreaterEq(expr,expr2) => format!("({}>={})",render_expr(expr),render_expr(expr2)),
        sr::Expr::LessEq(expr,expr2) => format!("({}<={})",render_expr(expr),render_expr(expr2)),
        sr::Expr::LogAnd(expr,expr2) => format!("({}&&{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::LogOr(expr,expr2) => format!("({}||{})",render_expr(expr),render_expr(expr2)),
        sr::Expr::Assign(expr,expr2) => format!("{}={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::AddAssign(expr,expr2) => format!("{}+={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::SubAssign(expr,expr2) => format!("{}-={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::MulAssign(expr,expr2) => format!("{}*={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::DivAssign(expr,expr2) => format!("{}/={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::ModAssign(expr,expr2) => format!("{}%={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::AndAssign(expr,expr2) => format!("{}&={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::OrAssign(expr,expr2) => format!("{}|={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::XorAssign(expr,expr2) => format!("{}^={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::ShlAssign(expr,expr2) => format!("{}<<={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::ShrAssign(expr,expr2) => format!("{}>>={}",render_expr(expr),render_expr(expr2)),
        sr::Expr::Continue => "continue".to_string(),
        sr::Expr::Break(_) => "break".to_string(),
        sr::Expr::Return(expr) => if let Some(expr) = expr { format!("return {}",render_expr(expr)) } else { "return".to_string() },
        sr::Expr::Block(block) => format!("{{ {} }}",render_block(block)),
        sr::Expr::If(expr,block,else_expr) => if let Some(else_expr) = else_expr {
            format!("if ({}) {{ {} }} else {}",render_expr(expr),render_block(block),render_expr(else_expr))
        }
        else {
            format!("if ({}) {{ {} }}",render_expr(expr),render_block(block))
        },
        sr::Expr::Loop(block) => format!("while (true) {{ {} }}",render_block(block)),
        sr::Expr::For(ident,range,block) => format!("for (int {} = TODO; {} < TODO; {}++) {{ {} }}",ident,ident,ident,render_block(block)),
        sr::Expr::While(expr,block) => format!("while ({}) {{ {} }}",render_expr(expr),render_block(block)),
        _ => panic!("ERROR: no Unknown* nodes should exist here"),
    }
}

fn render_block(block: &sr::Block) -> String {
    let mut r = String::new();
    for stat in block.stats.iter() {
        r += &render_stat(stat);
    }
    if let Some(expr) = block.expr {
        r += &format!("return {}",render_expr(&expr));
    }
    r
}

fn collect_layouts(ident: &str,type_: &sr::Type) -> Vec<(String,String)> {
    let mut layouts: Vec<(String,String)> = Vec::new();
    match type_ {
        sr::Type::Struct(struct_) => {
            for field in struct_.fields.iter() {
                let base_type = if let sr::Type::Base(base_type) = field.type_ { base_type } else { panic!("only base types supported in function parameter struct fields"); };
                layouts.push((gl_base_type_name(&base_type).to_string(),format!("{}_{}",ident,field.ident)));
            }
        },
        _ => panic!("unable to handle non-struct parameters just yet"),
    }
    layouts
}

pub fn compile_vertex_shader(module: sr::Module,vertex_ident: String,vertex_fields: Vec<sr::Field>) -> Option<Vec<u8>> {

    process_vertex_shader(module,vertex_ident,vertex_fields);

    // find main function
    let main = &module.functions.get("main").expect("main function missing");

    // collect input layouts
    let mut input_layouts: Vec<(String,String)> = Vec::new();
    for param in main.params {
        input_layouts.extend_from_slice(&collect_layouts(&param.ident,&param.type_));
    }
    
    // collect output layouts
    let main_return_type = if let sr::Type::Void = main.return_type { panic!("main should return non-void"); } else { main.return_type };
    let output_layouts = collect_layouts("output",&main_return_type);
    if output_layouts[0].1 != "vec4" {
        panic!("first element of the output should be a vec4 for the vertex position, optionally followed by other parameters");
    }
    
    // render start
    let mut r = "#version 450\n".to_string();

    // render input locations
    for i in 0..input_layouts.len() {
        r += &format!("layout(location={}) in {} {};\n",i,input_layouts[i].0,input_layouts[i].1);
    }

    // render output locations (0 is gl_Position)
    for i in 1..output_layouts.len() {
        r += &format!("layout(location={}) out {} {};\n",i,output_layouts[i].0,output_layouts[0].1);
    }

    // render main code
    r += "void main() { ";

    r += &render_block(&main.block);

    r += "}\n";

    println!("GLSL code:\n{}",r);

    r += "\0";
    Some(r.into_bytes())
}

pub fn compile_fragment_shader(module: sr::Module) -> Option<Vec<u8>> {

    process_fragment_shader(module);
    println!("TODO: render GLSL");
    None
}
