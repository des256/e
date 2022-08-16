use crate::*;

impl Lexer {

    pub(crate) fn parse_item(&mut self) -> Option<Item> {
        self.skip_attrs();
        self.skip_visibility();
        if self.parse_ident("mod") {
            let ident = self.parse_any_ident().unwrap();
            let group = self.parse_brace().unwrap();
            // TODO: { Item } .
            Some(Item::Mod(ident,Vec::new()))
        }
        else if self.parse_ident("const") {
            let ident = if let Some(ident) = self.parse_any_ident() {
                Some(ident)
            }
            else {
                self.parse_punct('_');
                None
            };
            self.parse_punct(':');
            let ty = self.parse_type().unwrap();
            self.parse_punct('=');
            let expr = self.parse_expr().unwrap();
            self.parse_punct(';');
            Some(Item::Const(ident,Box::new(ty),Box::new(expr)))
        }
        else if self.parse_ident("fn") {
            let ident = self.parse_any_ident().unwrap();
            let group = self.parse_paren().unwrap();
            // TODO: [ FuncArg { ',' FuncArg } [ `,` ] ] .
            let args = Vec::new();
            let result = if self.parse_punct2('-','>') {
                Some(Box::new(self.parse_type().unwrap()))
            }
            else {
                None
            };
            let group = self.parse_brace().unwrap();
            // TODO: Block .
            let code = Block { exprs: Vec::new(), result: None, };
            Some(Item::Func(ident,args,result,code))
        }
        else if self.parse_ident("type") {
            let ident = self.parse_any_ident().unwrap();
            self.parse_punct('=');
            let ty = self.parse_type().unwrap();
            self.parse_punct(';');
            Some(Item::Alias(ident,Box::new(ty)))
        }
        else if self.parse_ident("struct") {
            let ident = self.parse_any_ident().unwrap();
            if let Some(group) = self.parse_paren() {
                // TODO: [ Type { `,` Type } [ `,` ] ] .
                self.parse_punct(';');
                Some(Item::Tuple(ident,Vec::new()))
            }
            else {
                if let Some(group) = self.parse_brace() {
                    // TODO: [ StructField { `,` StructField } [ `,` ] ] .
                    self.parse_punct(';');
                    Some(Item::Struct(ident,Vec::new()))
                }
                else {
                    self.parse_punct(';');
                    Some(Item::Struct(ident,Vec::new()))
                }
            }
        }
        else if self.parse_ident("enum") {
            let ident = self.parse_any_ident().unwrap();
            let group = self.parse_brace().unwrap();
            // TODO: [ EnumVar { `,` EnumVar } [ `,` ] ] .
            Some(Item::Enum(ident,Vec::new()))
        }
        else if self.parse_ident("union") {
            let ident = self.parse_any_ident().unwrap();
            let group = self.parse_brace().unwrap();
            // TODO: [ StructField { `,` StructField } [ `,` ] ] .
            Some(Item::Union(ident,Vec::new()))
        }
        else if self.parse_ident("static") {
            let is_mut = self.parse_ident("mut");
            let ident = self.parse_any_ident().unwrap();
            self.parse_punct(':');
            let ty = self.parse_type().unwrap();
            self.parse_punct('=');
            let expr = self.parse_expr().unwrap();
            self.parse_punct(';');
            Some(Item::Static(is_mut,ident,Box::new(ty),Box::new(expr)))
        }
        else if self.parse_ident("impl") {
            let ty = self.parse_type().unwrap();
            let group = self.parse_brace().unwrap();
            // TODO: { ImplItem } .
            Some(Item::Impl(Box::new(ty),Vec::new()))
        }
        else {
            panic!("token not supported: {}");
        }
    }
}
