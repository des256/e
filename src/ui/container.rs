use {
    super::*,
    std::{
        rc::Rc,
    },
};

struct Container {
    child: Rc<dyn Widget>,
    primitives: Vec<Primitive>,
}

impl Container {
    pub fn new(child: Rc<dyn Widget>) -> Container {
        Container {
            child: child,
            primitives: Vec::new(),
        }
    }
}

impl Widget for Container {
    fn realize(&mut self,context: &Context) -> Primitive {
        Primitive::new()
    }
}