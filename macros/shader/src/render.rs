use {
    crate::*,
};

fn render_type(ty: ast::Type) -> String {
    "type".to_string()
}

fn render_pat(pat: ast::Pat) -> String {
    "pat".to_string()
}

fn render_stat(stat: ast::Stat) -> String {
    "stat".to_string()
}

fn render_function(symbol: String,params: Vec<(ast::Pat,Box<ast::Type>)>,return_ty: Option<Box<ast::Type>>,stats: Vec<ast::Stat>,) -> String {
    let mut r = String::new();
    r += &format!("tsr::Function(\"{}\",vec![",symbol);
    for (pat,ty) in params {
        r += &format!("({},{}),",&render_pat(pat),&render_type(*ty));
    }
    r += "],";
    if let Some(return_ty) = return_ty {
        r += &format!("Some({}),",&render_type(*return_ty));
    }
    else {
        r += "None,";
    }
    r += "vec![";
    for stat in stats {
        r += &render_stat(stat);
        r += ",";
    }
    r += "],)";
    r
}

pub(crate) fn render_vertex_shader(item: ast::Item,uniform: String,vertex: String,varying: String) -> String {
    let mut r = String::new();
    match item {
        ast::Item::Module(symbol,items) => {
            r += &format!("const {}: VertexShader {{ ",symbol);
            r += &format!("uniform: {}, ",uniform);
            r += &format!("vertex: {}, ",vertex);
            r += &format!("varying: {}, ",varying);
            r += "functions: vec![";
            for item in items {
                if let ast::Item::Function(symbol,params,result_ty,stats) = item {
                    r += &render_function(symbol,params,result_ty,stats);
                }
                else {
                    panic!("module inside module not allowed");
                }
            }
            r += "], };";
        },
        ast::Item::Function(symbol,params,result_ty,stats) => {
            r += &format!("const {}: VertexShader {{ ",symbol);
            r += &format!("uniform: {}, ",uniform);
            r += &format!("vertex: {}, ",vertex);
            r += &format!("varying: {}, ",varying);
            r += &format!("functions: vec![");
            r += &render_function("main".to_string(),params,result_ty,stats);
            r += "], };";
        },
    }
    r
}
