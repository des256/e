use super::*;

impl Parser {

    // Wildcard, Rest, Boolean, Integer, Float, UnknownIdent, UnknownTuple, UnknownStruct, UnknownVariant, AnonTuple, Array, Range
    pub(crate) fn pat(&mut self) -> Result<Pat,String> {
        
        // Wildcard
        if self.punct('_') {
            Ok(Pat::Wildcard)
        }

        // Rest
        else if self.punct2('.','.') {
            Ok(Pat::Rest)
        }

        // Boolean
        else if let Some(value) = self.boolean_literal() {
            Ok(Pat::Boolean(value))
        }

        // Integer
        else if let Some(value) = self.integer_literal() {
            Ok(Pat::Integer(value as i64))
        }

        // Float
        else if let Some(value) = self.float_literal() {
            Ok(Pat::Float(value))
        }

        // Ident, Tuple, Struct, Variant
        else if let Some(ident) = self.ident() {

            // Struct
            if let Some(ident_pats) = self.brace_ident_pats()? {
                Ok(Pat::Struct(ident,ident_pats))
            }

            // Tuple
            else if let Some(pats) = self.paren_pats()? {
                Ok(Pat::Tuple(ident,pats))
            }

            // Variant
            else if self.punct2(':',':') {
                let variant_ident = self.ident().expect("identifier expected");
                if let Some(ident_pats) = self.brace_ident_pats()? {
                    Ok(Pat::Variant(ident,variant_ident,VariantPat::Struct(ident_pats)))
                }
                else if let Some(pats) = self.paren_pats()? {
                    Ok(Pat::Variant(ident,variant_ident,VariantPat::Tuple(pats)))
                }
                else {
                    Ok(Pat::Variant(ident,variant_ident,VariantPat::Naked))
                }
            }

            // Ident
            else {
                Ok(Pat::Ident(ident))
            }
        }
        
        // AnonTuple
        else if let Some(pats) = self.paren_pats()? {
            Ok(Pat::AnonTuple(pats))
        }

        // Array
        else if let Some(pats) = self.bracket_pats()? {
            Ok(Pat::Array(pats))
        }

        else {
            self.err("pattern expected")
        }
    }

    // Range
    pub(crate) fn ranged_pat(&mut self) -> Result<Pat,String> {
        let pat = self.pat()?;
        if self.punct3('.','.','=') {
            Ok(Pat::Range(Box::new(pat),Box::new(self.pat()?)))
        }
        else {
            Ok(pat)
        }
    }

    pub(crate) fn or_pats(&mut self) -> Result<Vec<Pat>,String> {
        self.punct('|');
        let mut pats: Vec<Pat> = Vec::new();
        pats.push(self.ranged_pat()?);
        while self.punct('|') {
            pats.push(self.ranged_pat()?);
        }
        Ok(pats)
    }
}
