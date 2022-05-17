use {
    std::sync::Arc,
    crate::*,
};

pub trait Config {
    // configuration tree aspects
    fn create() -> Arc<dyn View>;
}
