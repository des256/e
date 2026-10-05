use {
    crate::*,
    math::*,
    std::{
        collections::HashMap,
        ffi::CString,
        ptr::{null, null_mut},
        sync::Arc,
    },
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
    pub onnx: Arc<Onnx>,
    pub session: *mut ffi::Session,
    pub inputs: Vec<Input>,
    pub outputs: Vec<Output>,
    pub metadata: HashMap<String, String>,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn create_inference(&self) -> Result<Inference, OnnxError> {
        Err(OnnxError::NotImplemented)
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        unsafe { (self.onnx.functions.ReleaseSession)(self.session) };
    }
}
