use {
    crate::*,
};

impl Parser {

    // Wildcard, Rest, Literal, Ident (Const), Struct, Tuple, Array, AnonTuple, Variant
    pub(crate) fn parse_pat(&mut self) -> Pat {
        
        // Wildcard
        if self.punct('_') {
            Pat::Wildcard
        }

        // Rest
        else if self.punct2('.','.') {
            Pat::Rest
        }

        // Literal
        else if let Some(literal) = self.literal() {
            Pat::Literal(literal)
        }

        // Ident (Const), Struct, Tuple, Variant
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_pats) = self.parse_brace_ident_pats() {
                Pat::Struct(ident,ident_pats)
            }

            // Tuple
            else if let Some(pats) = self.parse_paren_pats() {
                Pat::Tuple(ident,pats)
            }

            // Variant
            else if self.punct2(':',':') {
                let variant = self.ident().expect("identifier expected");
                if let Some(ident_pats) = self.parse_brace_ident_pats() {
                    Pat::Variant(ident,VariantPat::Struct(variant,ident_pats))
                }
                else if let Some(pats) = self.parse_paren_pats() {
                    Pat::Variant(ident,VariantPat::Tuple(variant,pats))
                }
                else {
                    Pat::Variant(ident,VariantPat::Naked(variant))
                }
            }

            // Ident (Const)
            else {
                Pat::Ident(ident)
            }
        }
        
        // Array
        else if let Some(pats) = self.parse_bracket_pats() {
            Pat::Array(pats)
        }

        // AnonTuple
        else if let Some(pats) = self.parse_paren_pats() {
            Pat::AnonTuple(pats)
        }

        else {
            panic!("pattern expected");
        }
    }

    // Range
    pub(crate) fn parse_ranged_pat(&mut self) -> Pat {
        let pat = self.parse_pat();
        if self.punct3('.','.','=') {
            Pat::Range(Box::new(pat),Box::new(self.parse_pat()))
        }
        else {
            pat
        }
    }

    pub(crate) fn parse_or_pats(&mut self) -> Vec<Pat> {
        self.punct('|');
        let mut pats: Vec<Pat> = Vec::new();
        pats.push(self.parse_pat());
        while self.punct('|') {
            pats.push(self.parse_pat());
        }
        pats
    }
}
