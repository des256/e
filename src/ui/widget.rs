use {
    crate::*,
    std::sync::Arc,
};

pub trait Widget {
    fn create_primitives(&self) -> Vec<Arc<dyn Primitive>>;
}
