// all interfaces to the underlying operating system

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
pub enum KeyEvent {
    Down { code: u32, },
    Up { code: u32, },
}

impl Display for KeyEvent {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            KeyEvent::Press { code, } => write!(f,"Press { code: {}, }",code),
            KeyEvent::Release { code, } => write!(f,"Release { code: {}, }",code),
        }
    }
}

#[derive(Copy,Clone,Debug)]
pub enum PointerEvent {
    Down { position: Vec2<f32>, buttons: u32, synthesized: bool, },  // the pointer made contact with the device
    Up { position: Vec2<f32>, buttons: u32, synthesized: bool, },  // the pointer has stopped making contact with the device
    Move { position: Vec2<f32>, buttons: u32,synthesized: bool, hover: bool, },  // the pointer has moved with respect to the device while the pointer is in contact with the device
    Enter { position: Vec2<f32>, buttons: u32, synthesized: bool, hover: bool, target: Option<&dyn Any>, },  // the pointer has moved with respect to the device and the pointer entered an object
    Leave { position: Vec2<f32>, buttons: u32, synthesized: bool, hover: bool, target: Option<&dyn Any>, },   // the pointer has moved with respect to the device and the pointer left an object
    Cancel { position: Vec2<f32>, buttons: u32, synthesized: bool, hover: bool, },  // the input from the pointer is no longer directed towards this receiver
    Start { position: Vec2<f32>, },  // a pan/zoom gesture was started
    Update { position: Vec2<f32>, scale: f32, },  // a pan/zoom gesture was updated
    End { position: Vec2<f32>, }, // a pan/zoom gesture was ended
    Scroll { position: Vec2<f32>, buttons: u32, delta: Vec2<f32>, },  // a scroll indication was generated for this pointer (by, for instance, a mouse wheel)
}

impl Display for PointerEvent {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            PointerEvent::Down { position, buttons, synthesized, } => write!(f,"Down { position: {},buttons: {},synthesized: {}, }",position,buttons,pressure,radius,synthesized),
            PointerEvent::Up { position, buttons, synthesized, } => write!(f,"Up { position: {},buttons: {},synthesized: {}, }",position,buttons,pressure,radius,synthesized),
            PointerEvent::Move { position, buttons, synthesized, hover, } => write!(f,"Move { position: {},buttons: {},synthesized: {},hover: {}, }",position,buttons,pressure,radius,synthesized,hover),
            PointerEvent::Enter { position, buttons, synthesized, hover, target, } => write!(f,"Enter { position: {},buttons: {},synthesized: {},hover: {},target: {}, }",position,buttons,pressure,radius,synthesized,hover,target),
            PointerEvent::Leave { position, buttons, synthesized, hover, target, } => write!(f,"Leave { position: {},buttons: {},synthesized: {},hover: {},target: {}, }",position,buttons,pressure,radius,synthesized,hover,target),
            PointerEvent::Cancel { position, buttons, synthesized, hover, } => write!(f,"Leave { position: {},buttons: {},synthesized: {},hover: {}, }",position,buttons,pressure,radius,synthesized,hover),
            PointerEvent::Start { position, } => write!(f,"Start { position: {}, }",position),
            PointerEvent::Update { position, scale, } => write!(f,"Update { position: {}, scale: {}, }",position,scale),
            PointerEvent::End { position, } => write!(f,"End { position: {}, }",position),
            PointerEvent::Scroll { position, buttons, delta, } => write!(f,"Scroll { position: {}, buttons: {}, delta: {}, }",position,buttons,delta),
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
    Key(KeyEvent),
    Pointer(PointerEvent),
    Configure(Rect<f32>),
    Expose(Rect<f32>),
    Close,
}

impl Display for Event {
    fn fmt(&self,f: &mut Formatter) -> Result {
        match self {
            Event::Key(event) => write!(f,"{}",event),
            Event::Pointer(event) => write!(f,"{}",event),
            Event::Configure(rect) => write!(f,"Configure({})",rect),
            Event::Expose(rect) => write!(f,"Expose({})",rect),
            Event::Close => write!(f,"Close"),
        }
    }
}

#[cfg(system="linux")]
mod linux;
#[cfg(system="linux")]
pub use linux::*;

#[cfg(system="web")]
mod web;
#[cfg(system="web")]
pub use web::*;
