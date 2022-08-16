use crate::*;

impl Lexer {

    pub(crate) fn skip_attrs(&mut self) {
        if self.parse_punct('#') {
            self.parse_bracket();
        }
    }

    pub(crate) fn skip_visibility(&mut self) {
        if self.parse_ident("pub") {
            self.parse_paren();
        }
    }
}
