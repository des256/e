use {
    crate::*,
    std::{
        fmt::{
            Display,
            Formatter,
            Result,
        },
    },
};

#[derive(Copy,Clone,Debug)]
pub enum MouseButton {
    Left,
    Middle,
    Right,
}

impl Display for MouseButton {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            MouseButton::Left => write!(f,"Left"),
            MouseButton::Middle => write!(f,"Middle"),
            MouseButton::Right => write!(f,"Right"),
        }
    }
}

#[derive(Copy,Clone,Debug)]
pub enum MouseWheel {
    Up,
    Down,
    Left,
    Right,
}

impl Display for MouseWheel {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            MouseWheel::Up => { write!(f,"Up") },
            MouseWheel::Down => { write!(f,"Down") },
            MouseWheel::Left => { write!(f,"Left") },
            MouseWheel::Right => { write!(f,"Right") },
        }
    }
}

pub enum MouseCursor {
    Arrow,
    VArrow,
    Hourglass,
    Crosshair,
    Finger,
    OpenHand,
    GrabbingHand,
    MagnifyingGlass,
    Caret,
    SlashedCircle,
    SizeNSEW,
    SizeNESW,
    SizeNWSE,
    SizeWE,
    SizeNS,
}

#[derive(Copy,Clone,Debug)]
pub enum Event {
    KeyPress(u8),
    KeyRelease(u8),
    MousePress(Vec2<i32>,MouseButton),
    MouseRelease(Vec2<i32>,MouseButton),
    MouseWheel(MouseWheel),
    MouseMove(Vec2<i32>),
    Configure(Rect<i32>),
    Render,
    Close,
}

impl Display for Event {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Event::KeyPress(c) => { write!(f,"KeyPress({})",c) },
            Event::KeyRelease(c) => { write!(f,"KeyRelease({})",c) },
            Event::MousePress(p,b) => { write!(f,"MousePress({},{})",p,b) },
            Event::MouseRelease(p,b) => { write!(f,"MouseRelease({},{})",p,b) },
            Event::MouseWheel(w) => { write!(f,"MouseWheel({})",w) },
            Event::MouseMove(p) => { write!(f,"MouseMove({})",p) },
            Event::Configure(r) => { write!(f,"Configure({})",r) },
            Event::Render => { write!(f,"Render") },
            Event::Close => { write!(f,"Close") },
        }
    }
}

#[cfg(feature="system_linux")]
mod linux;
#[cfg(feature="system_linux")]
pub use linux::*;
