use super::*;

impl Parser {

    // Wildcard, Rest, Boolean, Integer, Float, UnknownIdent, UnknownTuple, UnknownStruct, UnknownVariant, AnonTuple, Array, Range
    pub(crate) fn pat(&mut self) -> Pat {
        
        // Wildcard
        if self.punct('_') {
            Pat::Wildcard
        }

        // Rest
        else if self.punct2('.','.') {
            Pat::Rest
        }

        // Boolean
        else if let Some(value) = self.boolean_literal() {
            Pat::Boolean(value)
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            Pat::Integer(value as i64)
        }

        // Float
        else if let Some(value) = self.float_literal() {
            Pat::Float(value)
        }

        // Ident, Tuple, Struct, Variant
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_pats) = self.brace_ident_pats() {
                Pat::Struct(ident,ident_pats)
            }

            // Tuple
            else if let Some(pats) = self.paren_pats() {
                Pat::Tuple(ident,pats)
            }

            // Variant
            else if self.punct2(':',':') {
                let variant_ident = self.ident().expect("identifier expected");
                if let Some(ident_pats) = self.brace_ident_pats() {
                    Pat::Variant(ident,variant_ident,VariantPat::Struct(ident_pats))
                }
                else if let Some(pats) = self.paren_pats() {
                    Pat::Variant(ident,variant_ident,VariantPat::Tuple(pats))
                }
                else {
                    Pat::Variant(ident,variant_ident,VariantPat::Naked)
                }
            }

            // Ident
            else {
                Pat::Ident(ident)
            }
        }
        
        // AnonTuple
        else if let Some(pats) = self.paren_pats() {
            Pat::AnonTuple(pats)
        }

        // Array
        else if let Some(pats) = self.bracket_pats() {
            Pat::Array(pats)
        }

        else {
            self.fatal("pattern expected");
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
