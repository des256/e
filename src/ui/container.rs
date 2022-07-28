use {
    super::*,
    std::{
        rc::Rc,
    },
};

pub struct Container {
    _child: Rc<dyn Widget>,
}

impl Container {
    pub fn new(child: Rc<dyn Widget>) -> Container {
        Container {
            _child: child,
        }
    }
}

impl Widget for Container {
    fn realize(&self) -> Primitive {
        Primitive {
            text: String::new(),
            children: Vec::new(),
        }
    }
}
