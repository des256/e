use {
    crate::*,
    std::{ffi::CString, ptr::null_mut},
};

pub struct Onnx {
    pub(crate) env: *mut ffi::Env,
    pub(crate) allocator: *mut ffi::Allocator,
    pub(crate) functions: ffi::Functions,
}

unsafe impl Send for Onnx {}
unsafe impl Sync for Onnx {}

impl Onnx {
    pub fn new(onnx_version: usize) -> Result<Self, OnnxError> {
        // initialize ONNX runtime
        let api_base = unsafe { ffi::OrtGetApiBase() };
        if api_base.is_null() {
            return Err(OnnxError::Failed);
        }
        let get_api = unsafe { (*api_base).GetApi };
        let api = unsafe { get_api(onnx_version as u32) };
        if api.is_null() {
            return Err(OnnxError::UnknownVersion);
        }
        let functions = ffi::Functions::new(api);
        let log_id = CString::new("onnx").unwrap();
        let mut env: *mut ffi::Env = null_mut();
        let status = unsafe {
            (functions.CreateEnv)(
                ffi::LoggingLevel::Warning,
                log_id.as_ptr(),
                &mut env as *mut _,
            )
        };
        if !status.is_null() {
            return Err(OnnxError::Failed);
        }
        let allocator: *mut ffi::Allocator = null_mut();
        let status = unsafe { (functions.GetAllocatorWithDefaultOptions)(allocator as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::Failed);
        }

        Ok(Self {
            env,
            allocator,
            functions,
        })
    }
}

impl Drop for Onnx {
    fn drop(&mut self) {
        unsafe { (self.functions.ReleaseEnv)(self.env) };
    }
}
