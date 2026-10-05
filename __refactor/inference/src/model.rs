use {
    crate::*,
    std::{collections::HashMap, sync::Arc},
};

pub struct Input {
    pub name: String,
    pub element_type: ffi::TensorElementDataType,
    pub shape: Vec<usize>,
}

pub struct Output {
    pub name: String,
    pub element_type: ffi::TensorElementDataType,
    pub shape: Vec<usize>,
}

pub struct Model {
    pub runtime: Arc<Runtime>,
    pub session: *mut ffi::Session,
    pub(crate) inputs: Vec<Input>,
    pub(crate) outputs: Vec<Output>,
    pub(crate) metadata: HashMap<String, String>,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn inputs(&self) -> &[Input] {
        &self.inputs
    }

    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.runtime.functions.ReleaseSession)(self.session) };
    }
}
