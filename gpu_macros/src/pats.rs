use {
    crate::*,
};

impl Parser {

    // Wildcard, Rest, Literal, Ident (Const), Struct, Tuple, Array, AnonTuple, Variant
    pub(crate) fn pat(&mut self) -> Pat {
        
        // Wildcard
        if self.punct('_') {
            Pat::Wildcard
        }

        // Rest
        else if self.punct2('.','.') {
            Pat::Rest
        }

        // Literal
        else if let Some(value) = self.boolean_literal() {
            Pat::Boolean(value)
        }
        else if let Some(value) = self.integer_literal() {
            Pat::Integer(value as i64)
        }
        else if let Some(value) = self.float_literal() {
            Pat::Float(value)
        }

        // Ident (Const), Struct, Tuple, Variant
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_pats) = self.brace_ident_pats() {
                Pat::UnknownStruct(ident,ident_pats)
            }

            // Tuple
            else if let Some(pats) = self.paren_pats() {
                let mut ident_pats: Vec<IdentPat> = Vec::new();
                let mut i = 0usize;
                for pat in pats {
                    ident_pats.push(IdentPat::IdentPat(format!("_{}",i),pat));
                    i += 1;
                }
                Pat::UnknownStruct(ident,ident_pats)
            }

            // Variant
            else if self.punct2(':',':') {
                let variant = self.ident().expect("identifier expected");
                if let Some(ident_pats) = self.brace_ident_pats() {
                    Pat::UnknownVariant(ident,VariantPat::Struct(variant,ident_pats))
                }
                else if let Some(pats) = self.paren_pats() {
                    Pat::UnknownVariant(ident,VariantPat::Tuple(variant,pats))
                }
                else {
                    Pat::UnknownVariant(ident,VariantPat::Naked(variant))
                }
            }

            // Ident (Const)
            else {
                Pat::Ident(ident)
            }
        }
        
        // Array
        else if let Some(pats) = self.bracket_pats() {
            Pat::Array(pats)
        }

        // AnonTuple
        else if let Some(pats) = self.paren_pats() {
            Pat::AnonTuple(pats)
        }

        else {
            panic!("pattern expected");
        }
    }

    // Range
    pub(crate) fn ranged_pat(&mut self) -> Pat {
        let pat = self.pat();
        if self.punct3('.','.','=') {
            Pat::Range(Box::new(pat),Box::new(self.pat()))
        }
        else {
            pat
        }
    }

    pub(crate) fn or_pats(&mut self) -> Vec<Pat> {
        self.punct('|');
        let mut pats: Vec<Pat> = Vec::new();
        pats.push(self.ranged_pat());
        while self.punct('|') {
            pats.push(self.ranged_pat());
        }
        pats
    }
}
