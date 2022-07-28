/*pub struct ButtonState<'a> {
    context: Option<Context<'a>>,
    pressed: bool,
}

pub struct Button<'a> {
    child: Rc<dyn Widget>,
    state: Rc<ButtonState<'a>>,
    // on_pressed: some way to store a closure for this button
}

impl<'a> Button<'a> {
    pub fn new(child: Rc<dyn Widget>) -> Button<'a> {
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

impl<'a> Widget for Button<'a> {
    fn realize(&mut self,context: Context) -> Primitive {
        self.state.context = Some(context);
        Primitive::new()
    }
}*/