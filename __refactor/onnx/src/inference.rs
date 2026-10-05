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

pub trait OnnxTensor {
    fn value_ptr(&self) -> *const ffi::Value;
    fn value_mut(&self) -> *mut ffi::Value;
}

macro_rules! impl_onnx_tensor {
    ($type:ty) => {
        impl OnnxTensor for Tensor<$type, Onnx> {
            fn value_ptr(&self) -> *const ffi::Value {
                self.storage.value
            }
            fn value_mut(&self) -> *mut ffi::Value {
                self.storage.value
            }
        }
    };
}

impl_onnx_tensor!(u8);
impl_onnx_tensor!(i8);
impl_onnx_tensor!(u16);
impl_onnx_tensor!(i16);
impl_onnx_tensor!(u32);
impl_onnx_tensor!(i32);
impl_onnx_tensor!(u64);
impl_onnx_tensor!(i64);
impl_onnx_tensor!(F16);
impl_onnx_tensor!(f32);
impl_onnx_tensor!(f64);

pub struct Inference {
    pub model: Arc<Model>,
}

unsafe impl Send for Inference {}

impl Inference {
    pub fn run(
        &self,
        inputs: &[(&str, &dyn OnnxTensor)],
        outputs: &[(&str, &dyn OnnxTensor)],
    ) -> Result<(), OnnxError> {
        let input_name_cstrings: Vec<CString> = inputs
            .iter()
            .map(|(name, _)| CString::new(*name).unwrap())
            .collect();
        let input_name_ptrs: Vec<_> = input_name_cstrings.iter().map(|s| s.as_ptr()).collect();
        let input_value_ptrs: Vec<_> = inputs
            .iter()
            .map(|(_, tensor)| tensor.value_ptr())
            .collect();
        let output_name_cstrings: Vec<CString> = outputs
            .iter()
            .map(|(name, _)| CString::new(*name).unwrap())
            .collect();
        let output_name_ptrs: Vec<_> = output_name_cstrings.iter().map(|s| s.as_ptr()).collect();
        let mut output_value_ptrs: Vec<*mut ffi::Value> = outputs
            .iter()
            .map(|(_, tensor)| tensor.value_mut())
            .collect();
        let status = unsafe {
            (self.model.onnx.functions.Run)(
                self.model.session,
                null(),
                input_name_ptrs.as_ptr(),
                input_value_ptrs.as_ptr(),
                inputs.len(),
                output_name_ptrs.as_ptr(),
                outputs.len(),
                output_value_ptrs.as_mut_ptr(),
            )
        };
        if !status.is_null() {
            return Err(OnnxError::Fail);
        }
        Ok(())
    }
}
