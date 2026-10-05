pub struct Engine {
    tensorrt: Arc<Tensorrt>,
}

impl Engine {
    pub fn load(tensorrt: &Arc<Tensorrt>, path: &Path) -> Result<Self, TensorrtError> {
        // TODO: load engine
        Ok(Engine {
            tensorrt: Arc::clone(&tensorrt),
        })
    }
}
