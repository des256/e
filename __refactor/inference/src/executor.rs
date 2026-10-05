use {crate::*, std::sync::Arc};

pub struct Executor {
    model: Arc<Model>,
}

unsafe impl Send for Executor {}

impl Executor {
    pub fn new(model: &Arc<Model>) -> Self {
        Executor {
            model: Arc::clone(model),
        }
    }

    pub fn run(
        &self,
        inputs: &[(&str, &Tensor)],
        outputs: &[(&str, &mut Tensor)],
    ) -> Result<(), InferenceError> {
        todo!()
    }
}
