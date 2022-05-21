use {
    crate::*,
};

pub enum BlendMode {
    Src,
    Dst,
}

pub type Position = Vec2<i32>;
pub type Size = Vec2<u32>;
pub type Offset = Vec2<i32>;

pub enum Axis {
    Horizontal,
    Vertical,
}

pub enum BlurStyle {
    Inner,
    Normal,
    Outer,
    Solid,
}

pub enum MainAxisAlignment {
    Start,
    End,
    Center,
    SpaceAround,
    SpaceBetween,
    SpaceEvenly,
}

pub enum CrossAxisAlignment {
    Start,
    End,
    Center,
    Baseline,
    Stretch,
}

pub type Alignment = Vec2<i32>;
impl Alignment {
    pub const TOP_LEFT: Self = Self { x: -1,y: -1, };
    pub const TOP_CENTER: Self = Self { x: 0,y: -1, };
    pub const TOP_RIGHT: Self = Self { x: 1,y: -1, };
    pub const CENTER_LEFT: Self = Self { x: -1,y: 0, };
    pub const CENTER: Self = Self { x: 0,y: 0, };
    pub const CENTER_RIGHT: Self = Self { x: 1,y: 0, };
    pub const BOTTOM_LEFT: Self = Self { x: -1,y: 1, };
    pub const BOTTOM_CENTER: Self = Self { x: 0,y: 1, };
    pub const BOTTOM_RIGHT: Self = Self { x: 1,y: 1, };
}

mod color;
pub use color::*;

mod constraints;
pub use constraints::*;

mod edgeinsets;
pub use edgeinsets::*;

mod border;
pub use border::*;

mod borderradius;
pub use borderradius::*;

mod boxdecoration;
pub use boxdecoration::*;

mod primitive;
pub use primitive::*;

mod widget;
pub use widget::*;

mod container;
pub use container::*;

mod flex;
pub use flex::*;
