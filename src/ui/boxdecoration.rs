use {
    crate::*,
};

pub struct BoxDecoration {
    pub background_blend_mode: Option<BlendMode>,
    pub border: Option<Border>,
    pub border_radius: Option<BorderRadius>,
    pub color: Option<Color>,
    pub padding: Option<EdgeInsets>,
}
