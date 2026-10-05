pub struct Context {}

impl Context {
    pub fn new(engine: &Arc<Engine>) -> Result<Self, TensorrtError> {
        Ok(Context {})
    }
}
