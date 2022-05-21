use {
    crate::*,
};

pub struct Container {
    pub alignment: Option<Alignment>,
    pub color: Option<Color>,
    pub constraints: Option<Constraints>,
    pub decoration: Option<BoxDecoration>,
    pub margin: EdgeInsets,
    pub padding: EdgeInsets,
}
