use crate::*;



pub(crate) fn render_vertex_trait(ident: &str,fields: &Vec<(String,Type)>) -> String {

    // make sure all fields are base types
    let mut out_fields: Vec<(String,BaseType)> = Vec::new();
    for (ident,r#type) in fields {
        if let Type::Base(base_type) = *r#type {
            out_fields.push((*ident,base_type));
        }
    }

    let mut r = format!("impl Vertex for {} {{\n",ident);
    r += "    fn get_fields() -> Vec<(String,BaseType)> {{\n";
    r += "        vec![\n";
    let mut first_type = true;
    for (ident,base_type) in out_fields {
        if !first_type {
            r += ",\n";
        }
        r += &format!("            (\"{}\".to_string(),BaseType::{})",ident,base_type.variant());
        first_type = false;
    }
    r += "        ]\n";
    r += "    }\n";
    r += "}";
    r
}

pub(crate) fn render_vertex_shader(module: &Module,vertex: &str) -> String {
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {{\n";
    r += &format!("        compile_vertex_shader({},\"{}\",{}::get_fields())\n",render_module(module),vertex,vertex);
    r += "    }}\n";
    r += "}}\n";
    r
}

pub(crate) fn render_fragment_shader(module: &Module) -> String {
    let mut r = format!("pub mod {} {{\n",module.ident);
    r += "    use super::*;\n\n";
    r += "    pub fn code() -> Option<Vec<u8>> {{\n";
    r += &format!("        compile_fragment_shader({})\n",render_module(module));
    r += "    }}\n";
    r += "}}\n";
    r
}
