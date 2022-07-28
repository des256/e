use {
    super::*,
};

pub struct Text {
    text: String,
}

impl Text {
    pub fn new(text: &str) -> Text {
        Text {
            text: text.to_owned(),
        }
    }
}

impl Widget for Text {
    fn realize(&self) -> Primitive {
        Primitive {
            text: String::from(&self.text),
            children: Vec::new(),
        }
    }
}
