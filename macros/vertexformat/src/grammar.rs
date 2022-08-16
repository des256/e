pub(crate) struct Struct {
    pub ident: String,
    pub fields: Vec<StructField>,
}

pub(crate) struct StructField {
    pub ident: String,
    pub ty: String,
    pub gen_param: Option<String>,
}
