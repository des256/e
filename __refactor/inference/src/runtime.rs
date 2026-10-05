use {
    crate::*,
    std::{
        collections::HashMap,
        ffi::{CStr, CString},
        os::raw::c_char,
        path::Path,
        ptr::null_mut,
        sync::Arc,
    },
};

pub struct Runtime {
    api: *const ffi::Api,
    env: *mut ffi::Env,
    pub(crate) allocator: *mut ffi::Allocator,
    pub(crate) functions: ffi::Functions,
    #[cfg(feature = "tensorrt")]
    handle: *mut ffi::tensorrt::Runtime,
}

unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    fn new(onnx_version: usize) -> Result<Self, InferenceError> {
        // initialize ONNX runtime
        let api_base = unsafe { ffi::OrtGetApiBase() };
        if api_base.is_null() {
            return Err(InferenceError::Fail);
        }
        let get_api = unsafe { (*api_base).GetApi };
        let api = unsafe { get_api(onnx_version as u32) };
        if api.is_null() {
            return Err(InferenceError::UnknownVersion);
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
            return Err(InferenceError::Fail);
        }
        let allocator: *mut ffi::Allocator = null_mut();
        let status = unsafe { (functions.GetAllocatorWithDefaultOptions)(allocator as *mut _) };
        if !status.is_null() {
            return Err(InferenceError::Fail);
        }

        // TODO: initialize TensorRT runtime

        Ok(Self {
            api,
            env,
            allocator,
            functions,
            #[cfg(feature = "tensorrt")]
            handle: null_mut(),
        })
    }

    fn get_inputs_outputs_metadata(
        self: &Self,
        session: *mut ffi::Session,
    ) -> Result<(Vec<Input>, Vec<Output>, HashMap<String, String>), InferenceError> {
        // get inputs
        let mut inputs: Vec<Input> = Vec::new();
        let mut count: usize = 0;
        let status =
            unsafe { (self.functions.SessionGetInputCount)(session, &mut count as *mut _) };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSession)(session) };
            return Err(InferenceError::Fail);
        }
        for i in 0..count {
            // get name
            let mut name_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (self.functions.SessionGetInputName)(
                    session,
                    i,
                    self.allocator,
                    &mut name_ptr as *mut _,
                )
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            // get type info
            let mut type_info: *mut ffi::TypeInfo = null_mut();
            let status = unsafe {
                (self.functions.SessionGetInputTypeInfo)(session, i, &mut type_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get tensor info from type info
            let mut tensor_info: *const ffi::TensorTypeAndShapeInfo = null_mut();
            let status = unsafe {
                (self.functions.CastTypeInfoToTensorInfo)(type_info, &mut tensor_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get number of dimensions
            let mut dim_count: usize = 0;
            let status = unsafe {
                (self.functions.GetDimensionsCount)(tensor_info, &mut dim_count as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get dimensions
            let mut dims = vec![0i64; dim_count];
            let status = unsafe {
                (self.functions.GetDimensions)(tensor_info, dims.as_mut_ptr(), dim_count)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // extract shape
            let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();

            // get input element type
            let mut element_type = ffi::TensorElementDataType::Undefined;
            let status = unsafe {
                (self.functions.GetTensorElementType)(tensor_info, &mut element_type as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // release type info
            unsafe { (self.functions.ReleaseTypeInfo)(type_info) };

            inputs.push(Input {
                name,
                element_type,
                shape,
            });
        }

        // get outputs
        let mut outputs: Vec<Output> = Vec::new();
        let mut count: usize = 0;
        let status =
            unsafe { (self.functions.SessionGetOutputCount)(session, &mut count as *mut _) };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSession)(session) };
            return Err(InferenceError::Fail);
        }
        for i in 0..count {
            // get name
            let mut name_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (self.functions.SessionGetOutputName)(
                    session,
                    i,
                    self.allocator,
                    &mut name_ptr as *mut _,
                )
            };
            if !status.is_null() {
                return Err(InferenceError::Fail);
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            // get type info
            let mut type_info: *mut ffi::TypeInfo = null_mut();
            let status = unsafe {
                (self.functions.SessionGetOutputTypeInfo)(session, i, &mut type_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get tensor info from type info
            let mut tensor_info: *const ffi::TensorTypeAndShapeInfo = null_mut();
            let status = unsafe {
                (self.functions.CastTypeInfoToTensorInfo)(type_info, &mut tensor_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get number of dimensions
            let mut dim_count: usize = 0;
            let status = unsafe {
                (self.functions.GetDimensionsCount)(tensor_info, &mut dim_count as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // get dimensions
            let mut dims = vec![0i64; dim_count];
            let status = unsafe {
                (self.functions.GetDimensions)(tensor_info, dims.as_mut_ptr(), dim_count)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // extract shape
            let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();

            // get input element type
            let mut element_type = ffi::TensorElementDataType::Undefined;
            let status = unsafe {
                (self.functions.GetTensorElementType)(tensor_info, &mut element_type as *mut _)
            };
            if !status.is_null() {
                unsafe { (self.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }

            // release type info
            unsafe { (self.functions.ReleaseTypeInfo)(type_info) };

            outputs.push(Output {
                name,
                element_type,
                shape,
            });
        }

        // get metadata
        let mut model_metadata: *mut ffi::ModelMetadata = std::ptr::null_mut();
        let status = unsafe {
            (self.functions.SessionGetModelMetadata)(session, &mut model_metadata as *mut _)
        };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSession)(session) };
            return Err(InferenceError::Fail);
        }
        let mut keys_ptr: *mut *mut c_char = null_mut();
        let mut num_keys: i64 = 0;
        let status = unsafe {
            (self.functions.ModelMetadataGetCustomMetadataMapKeys)(
                model_metadata,
                self.allocator,
                &mut keys_ptr as *mut _,
                &mut num_keys as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseModelMetadata)(model_metadata) };
            unsafe { (self.functions.ReleaseSession)(session) };
            return Err(InferenceError::Fail);
        }
        let mut metadata = HashMap::new();
        for i in 0..num_keys as usize {
            let key_ptr = unsafe { *keys_ptr.add(i) };
            let key = unsafe { CStr::from_ptr(key_ptr).to_string_lossy().into_owned() };
            let key_cstr: CString = match CString::new(key.as_str()) {
                Ok(c_str) => c_str,
                Err(_) => {
                    unsafe { (self.functions.ReleaseModelMetadata)(model_metadata) };
                    unsafe { (self.functions.ReleaseSession)(session) };
                    return Err(InferenceError::Fail);
                }
            };
            let mut value_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (self.functions.ModelMetadataLookupCustomMetadataMap)(
                    model_metadata,
                    self.allocator,
                    key_cstr.as_ptr(),
                    &mut value_ptr as *mut _,
                )
            };
            if !status.is_null() {
                unsafe { (self.functions.AllocatorFree)(self.allocator, key_ptr as *mut _) };
                for j in (i + 1)..num_keys as usize {
                    unsafe {
                        (self.functions.AllocatorFree)(self.allocator, *keys_ptr.add(j) as *mut _)
                    };
                }
                unsafe { (self.functions.AllocatorFree)(self.allocator, keys_ptr as *mut _) };
                unsafe { (self.functions.ReleaseModelMetadata)(model_metadata) };
                unsafe { (self.functions.ReleaseSession)(session) };
                return Err(InferenceError::Fail);
            }
            if !value_ptr.is_null() {
                let value = unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() };
                metadata.insert(key, value);
                unsafe { (self.functions.AllocatorFree)(self.allocator, value_ptr as *mut _) };
            }
            unsafe { (self.functions.AllocatorFree)(self.allocator, key_ptr as *mut _) };
        }
        if !keys_ptr.is_null() {
            unsafe { (self.functions.AllocatorFree)(self.allocator, keys_ptr as *mut _) };
        }
        unsafe { (self.functions.ReleaseModelMetadata)(model_metadata) };
        Ok((inputs, outputs, metadata))
    }

    pub fn load_onnx_cpu(
        self: &Arc<Self>,
        path: &Path,
        threads: usize,
        optimization_level: usize,
    ) -> Result<Model, InferenceError> {
        // create session options
        let mut options: *mut ffi::SessionOptions = null_mut();
        let status = unsafe { (self.functions.CreateSessionOptions)(&mut options as *mut _) };
        if !status.is_null() {
            return Err(InferenceError::Fail);
        }

        // set graph optimization level
        let status = unsafe {
            (self.functions.SetSessionGraphOptimizationLevel)(options, optimization_level)
        };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSessionOptions)(options) };
            return Err(InferenceError::Fail);
        }

        // set intra op num threads
        let status = unsafe { (self.functions.SetIntraOpNumThreads)(options, threads as i32) };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSessionOptions)(options) };
            return Err(InferenceError::Fail);
        }

        // process path
        let path_str = path.to_str().unwrap();
        let c_path: CString = match CString::new(path_str) {
            Ok(c_path) => c_path,
            Err(_) => return Err(InferenceError::Fail),
        };

        // create session
        let mut session: *mut ffi::Session = null_mut();
        let status = unsafe {
            (self.functions.CreateSession)(
                self.env,
                c_path.as_ptr(),
                options,
                &mut session as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSessionOptions)(options) };
            return Err(InferenceError::Fail);
        }

        // we don't need the options anymore
        unsafe { (self.functions.ReleaseSessionOptions)(options) };

        // read inputs, outputs and metadata
        let (inputs, outputs, metadata) = self.get_inputs_outputs_metadata(session)?;

        Ok(Model {
            runtime: Arc::clone(&self),
            session,
            inputs,
            outputs,
            metadata,
        })
    }

    #[cfg(feature = "cuda")]
    fn load_onnx_cuda(
        self: &Arc<Self>,
        path: &Path,
        device_id: usize,
    ) -> Result<Model, InferenceError> {
        // create session options
        let mut options: *mut ffi::SessionOptions = null_mut();
        let status = unsafe { (self.functions.CreateSessionOptions)(&mut options as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::Fail);
        }

        // append CUDA execution provider
        let status = unsafe {
            (self.functions.SessionOptionsAppendExecutionProvider_CUDA)(options, device_id as i32)
        };
        if !status.is_null() {
            unsafe { (self.functions.ReleaseSessionOptions)(options) };
            return Err(OnnxError::Fail);
        }

        // process path
        let path_str = model_path.as_ref().to_str().unwrap();
        let c_path: CString = match CString::new(path_str) {
            Ok(c_path) => c_path,
            Err(error) => return Err(OnnxError::InvalidArgument),
        };

        // create session
        let mut session: *mut ffi::Session = null_mut();
        let status = unsafe {
            self.functions.CreateSession(
                self.environment,
                c_path.as_ptr(),
                options,
                &mut session as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { self.functions.ReleaseSessionOptions(options) };
            return Err(OnnxError::Fail);
        }

        // we don't need the options anymore
        unsafe { (self.functions.ReleaseSessionOptions)(options) };

        // read inputs, outputs and metadata
        let (inputs, outputs, metadata) = self.get_inputs_outputs_metadata(session)?;

        Ok(Model {
            Runtime: Arc::clone(&self),
            session,
            inputs,
            outputs,
            metadata,
        })
    }

    #[cfg(feature = "tensorrt")]
    fn load_tensorrt(path: &Path) -> Result<Model, InferenceError> {
        Ok(Model {
            Runtime: Arc::clone(&self),
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        unsafe { (self.functions.ReleaseEnv)(self.env) };
    }
}
