use {
    super::*,
    std::{
        rc::{
            Rc,
        },
    },
};

pub struct ButtonState {
    context: Option<Context>,
    pressed: bool,
}

pub struct Button {
    child: Rc<dyn Widget>,
    state: Rc<ButtonState>,
    // on_pressed: some way to store a closure for this button
}

impl Button {
    pub fn new(child: Rc<dyn Widget>) -> Button {
        Button {
            child: child,
            state: Rc::new(
                ButtonState {
                    context: None,
                    pressed: false,
                }
            ),
        }
    }
}

impl Widget for Button {
    fn realize(&mut self,context: Context) -> Primitive {
        self.state.context = Some(context);
        Primitive::new()

    }
}
