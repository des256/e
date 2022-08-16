use {
    crate::*,
};

impl Lexer {

    pub(crate) fn skip_attrs(&mut self) {
        if self.is_punct('#') {
            self.step();
            self.is_bracket_group();
        }
    }

    pub(crate) fn skip_visibility(&mut self) {
        if self.is_ident("pub") {
            self.step();
            self.is_paren_group();
        }
    }

    // StructField = Attrs Visibility IDENTIFIER `:` IDENTIFIER [ `<` IDENTIFIER `>` ] .
    pub(crate) fn is_struct_field(&mut self) -> Option<StructField> {
        self.skip_attrs();
        self.skip_visibility();
        if let Some(ident) = self.is_some_ident() {
            self.is_punct(':');
            let ty = self.is_some_ident().unwrap();
            let mut gen_param: Option<String> = None;
            if self.is_punct('<') {
                gen_param = Some(self.is_some_ident().unwrap());
                self.is_punct('>');
            }
            self.is_punct(',');
            Some(StructField {
                ident,
                ty,
                gen_param,
            })
        }
        else {
            None
        }
    }

    // Struct = Attrs Visibility `struct` IDENTIFIER `{` { StructField [ `,` ] } `;` .
    pub(crate) fn is_struct(&mut self) -> Struct {
        self.skip_attrs();
        self.skip_visibility();
        self.is_ident("struct");
        let ident = self.is_some_ident().unwrap();
        let group = self.is_brace_group().unwrap();
        let mut lexer = Lexer::new(group.stream());
        let mut fields: Vec<StructField> = Vec::new();
        while let Some(field) = lexer.is_struct_field() {
            fields.push(field);
        }
        Struct {
            ident,
            fields,
        }
    }
}
