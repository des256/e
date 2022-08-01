use {
    crate::*,
};

pub type Position = Vec2<f32>;
pub type Size = Vec2<f32>;
pub type Offset = Vec2<f32>;

pub enum Axis {
    Horizontal,
    Vertical,
}

pub struct Constraints {
    pub min: Size,
    pub max: Size,
}

/*pub struct Context<'a> {
    pub parent: &'a Primitive,
    pub index: usize,
}

impl<'a> Context<'a> {
    fn new(parent: &'a Primitive,index: usize) -> Context<'a> {
        Context {
            parent: parent,
            index: index,
        }
    }

    fn update(&self) {
        self.parent.update(self.index);
    }
}*/

pub trait Widget {
    // create a completely new primitive according to the current state of the widget
    fn realize(&self) -> Primitive;
}

mod primitive;
pub use primitive::*;

mod text;
pub use text::*;

mod container;
pub use container::*;

mod button;
pub use button::*;

impl System {
    /*pub async fn run_ui_loop(system: Rc<System>,ui: Rc<dyn Widget>) {
        let _prims = ui.realize();
        loop {
            // TODO: get event
            // TODO: execute event
        }
    }*/   
}
