use {
    crate::*,
    std::rc::Rc,
};

pub struct Semaphore {
    pub system: Rc<System>,
}
