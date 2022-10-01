use {
    crate::*,
    sr::*,
};

impl Parser {

    // Wildcard, Rest, Boolean, Integer, Float, UnknownIdent, UnknownTuple, UnknownStruct, UnknownVariant, AnonTuple, Array, Range
    pub(crate) fn pat(&mut self) -> ast::Pat {
        
        // Wildcard
        if self.punct('_') {
            ast::Pat::Wildcard
        }

        // Rest
        else if self.punct2('.','.') {
            ast::Pat::Rest
        }

        // Boolean
        else if let Some(value) = self.boolean_literal() {
            ast::Pat::Boolean(value)
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            ast::Pat::Integer(value as i64)
        }

        // Float
        else if let Some(value) = self.float_literal() {
            ast::Pat::Float(value)
        }

        // UnknownIdent, UnknownTuple, UnknownStruct, UnknownVariant
        else if let Some(ident) = self.ident() {

            // UnknownStruct
            if let Some(ident_pats) = self.brace_ident_pats() {
                ast::Pat::UnknownStruct(ident,ident_pats)
            }

            // UnknownTuple
            else if let Some(pats) = self.paren_pats() {
                ast::Pat::UnknownTuple(ident,pats)
            }

            // UnknownVariant
            else if self.punct2(':',':') {
                let variant_ident = self.ident().expect("identifier expected");
                if let Some(ident_pats) = self.brace_ident_pats() {
                    ast::Pat::UnknownVariant(ident,ast::UnknownPatVariant::Struct(variant_ident,ident_pats))
                }
                else if let Some(pats) = self.paren_pats() {
                    ast::Pat::UnknownVariant(ident,ast::UnknownPatVariant::Tuple(variant_ident,pats))
                }
                else {
                    ast::Pat::UnknownVariant(ident,ast::UnknownPatVariant::Naked(variant_ident))
                }
            }

            // UnknownIdent
            else {
                ast::Pat::UnknownIdent(ident)
            }
        }
        
        // AnonTuple
        else if let Some(pats) = self.paren_pats() {
            ast::Pat::AnonTuple(pats)
        }

        // Array
        else if let Some(pats) = self.bracket_pats() {
            ast::Pat::Array(pats)
        }

        else {
            self.fatal("pattern expected");
        }
    }

    // Range
    pub(crate) fn ranged_pat(&mut self) -> ast::Pat {
        let pat = self.pat();
        if self.punct3('.','.','=') {
            ast::Pat::Range(Box::new(pat),Box::new(self.pat()))
        }
        else {
            pat
        }
    }

    pub(crate) fn or_pats(&mut self) -> Vec<ast::Pat> {
        self.punct('|');
        let mut pats: Vec<ast::Pat> = Vec::new();
        pats.push(self.ranged_pat());
        while self.punct('|') {
            pats.push(self.ranged_pat());
        }
        pats
    }
}
