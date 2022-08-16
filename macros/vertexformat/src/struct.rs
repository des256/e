use {
    crate::*,
};

impl Lexer {

    // StructField = Attrs Visibility IDENTIFIER `:` IDENTIFIER [ `<` IDENTIFIER `>` ] .
    pub(crate) fn parse_struct_field(&mut self) -> Option<StructField> {
        self.skip_attrs();
        self.skip_visibility();
        if let Some(ident) = self.parse_any_ident() {
            self.parse_punct(':');
            let ty = self.parse_any_ident().unwrap();
            let mut gen_param: Option<String> = None;
            if self.parse_punct('<') {
                gen_param = Some(self.parse_any_ident().unwrap());
                self.parse_punct('>');
            }
            self.parse_punct(',');
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
    pub(crate) fn parse_struct(&mut self) -> Struct {
        self.skip_attrs();
        self.skip_visibility();
        self.parse_ident("struct");
        let ident = self.parse_any_ident().unwrap();
        let group = self.parse_brace().unwrap();
        let mut lexer = Lexer::new(group.stream());
        let mut fields: Vec<StructField> = Vec::new();
        while let Some(field) = lexer.parse_struct_field() {
            fields.push(field);
        }
        Struct {
            ident,
            fields,
        }
    }
}
