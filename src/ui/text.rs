use {
    super::*,
};

pub struct Text {
    pub text: String,
    pub primitives: Vec<Primitive>,
}

impl Text {
    pub fn new(text: &str) -> Text {
        Text {
            text: text.to_owned(),
            primitives: Vec::new(),
        }
    }
}

impl Widget for Text {
    fn realize(&mut self,context: &Context) -> Primitive {
        Primitive::new()
    }
}