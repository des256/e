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

pub struct Session {
    pub onnx: Arc<Onnx>,
    pub session: *mut ffi::Session,
    pub(crate) inputs: Vec<Input>,
    pub(crate) outputs: Vec<Output>,
    pub(crate) metadata: HashMap<String, String>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    fn get_inputs_outputs_metadata(
        onnx: &Onnx,
        session: *mut ffi::Session,
    ) -> Result<(Vec<Input>, Vec<Output>, HashMap<String, String>), OnnxError> {
        // get inputs
        let mut inputs: Vec<Input> = Vec::new();
        let mut count: usize = 0;
        let status =
            unsafe { (onnx.functions.SessionGetInputCount)(session, &mut count as *mut _) };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSession)(session) };
            return Err(OnnxError::Failed);
        }
        for i in 0..count {
            // get name
            let mut name_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (onnx.functions.SessionGetInputName)(
                    session,
                    i,
                    onnx.allocator,
                    &mut name_ptr as *mut _,
                )
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            // get type info
            let mut type_info: *mut ffi::TypeInfo = null_mut();
            let status = unsafe {
                (onnx.functions.SessionGetInputTypeInfo)(session, i, &mut type_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get tensor info from type info
            let mut tensor_info: *const ffi::TensorTypeAndShapeInfo = null_mut();
            let status = unsafe {
                (onnx.functions.CastTypeInfoToTensorInfo)(type_info, &mut tensor_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get number of dimensions
            let mut dim_count: usize = 0;
            let status = unsafe {
                (onnx.functions.GetDimensionsCount)(tensor_info, &mut dim_count as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get dimensions
            let mut dims = vec![0i64; dim_count];
            let status = unsafe {
                (onnx.functions.GetDimensions)(tensor_info, dims.as_mut_ptr(), dim_count)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // extract shape
            let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();

            // get input element type
            let mut element_type = ffi::TensorElementDataType::Undefined;
            let status = unsafe {
                (onnx.functions.GetTensorElementType)(tensor_info, &mut element_type as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // release type info
            unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };

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
            unsafe { (onnx.functions.SessionGetOutputCount)(session, &mut count as *mut _) };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSession)(session) };
            return Err(OnnxError::Failed);
        }
        for i in 0..count {
            // get name
            let mut name_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (onnx.functions.SessionGetOutputName)(
                    session,
                    i,
                    onnx.allocator,
                    &mut name_ptr as *mut _,
                )
            };
            if !status.is_null() {
                return Err(OnnxError::Failed);
            }
            let name = unsafe { CStr::from_ptr(name_ptr) }
                .to_string_lossy()
                .into_owned();

            // get type info
            let mut type_info: *mut ffi::TypeInfo = null_mut();
            let status = unsafe {
                (onnx.functions.SessionGetOutputTypeInfo)(session, i, &mut type_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get tensor info from type info
            let mut tensor_info: *const ffi::TensorTypeAndShapeInfo = null_mut();
            let status = unsafe {
                (onnx.functions.CastTypeInfoToTensorInfo)(type_info, &mut tensor_info as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get number of dimensions
            let mut dim_count: usize = 0;
            let status = unsafe {
                (onnx.functions.GetDimensionsCount)(tensor_info, &mut dim_count as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // get dimensions
            let mut dims = vec![0i64; dim_count];
            let status = unsafe {
                (onnx.functions.GetDimensions)(tensor_info, dims.as_mut_ptr(), dim_count)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // extract shape
            let shape: Vec<usize> = dims.iter().map(|d| *d as usize).collect();

            // get input element type
            let mut element_type = ffi::TensorElementDataType::Undefined;
            let status = unsafe {
                (onnx.functions.GetTensorElementType)(tensor_info, &mut element_type as *mut _)
            };
            if !status.is_null() {
                unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }

            // release type info
            unsafe { (onnx.functions.ReleaseTypeInfo)(type_info) };

            outputs.push(Output {
                name,
                element_type,
                shape,
            });
        }

        // get metadata
        let mut model_metadata: *mut ffi::ModelMetadata = std::ptr::null_mut();
        let status = unsafe {
            (onnx.functions.SessionGetModelMetadata)(session, &mut model_metadata as *mut _)
        };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSession)(session) };
            return Err(OnnxError::Failed);
        }
        let mut keys_ptr: *mut *mut c_char = null_mut();
        let mut num_keys: i64 = 0;
        let status = unsafe {
            (onnx.functions.ModelMetadataGetCustomMetadataMapKeys)(
                model_metadata,
                onnx.allocator,
                &mut keys_ptr as *mut _,
                &mut num_keys as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseModelMetadata)(model_metadata) };
            unsafe { (onnx.functions.ReleaseSession)(session) };
            return Err(OnnxError::Failed);
        }
        let mut metadata = HashMap::new();
        for i in 0..num_keys as usize {
            let key_ptr = unsafe { *keys_ptr.add(i) };
            let key = unsafe { CStr::from_ptr(key_ptr).to_string_lossy().into_owned() };
            let key_cstr: CString = match CString::new(key.as_str()) {
                Ok(c_str) => c_str,
                Err(_) => {
                    unsafe { (onnx.functions.ReleaseModelMetadata)(model_metadata) };
                    unsafe { (onnx.functions.ReleaseSession)(session) };
                    return Err(OnnxError::Failed);
                }
            };
            let mut value_ptr: *mut c_char = null_mut();
            let status = unsafe {
                (onnx.functions.ModelMetadataLookupCustomMetadataMap)(
                    model_metadata,
                    onnx.allocator,
                    key_cstr.as_ptr(),
                    &mut value_ptr as *mut _,
                )
            };
            if !status.is_null() {
                unsafe { (onnx.functions.AllocatorFree)(onnx.allocator, key_ptr as *mut _) };
                for j in (i + 1)..num_keys as usize {
                    unsafe {
                        (onnx.functions.AllocatorFree)(onnx.allocator, *keys_ptr.add(j) as *mut _)
                    };
                }
                unsafe { (onnx.functions.AllocatorFree)(onnx.allocator, keys_ptr as *mut _) };
                unsafe { (onnx.functions.ReleaseModelMetadata)(model_metadata) };
                unsafe { (onnx.functions.ReleaseSession)(session) };
                return Err(OnnxError::Failed);
            }
            if !value_ptr.is_null() {
                let value = unsafe { CStr::from_ptr(value_ptr).to_string_lossy().into_owned() };
                metadata.insert(key, value);
                unsafe { (onnx.functions.AllocatorFree)(onnx.allocator, value_ptr as *mut _) };
            }
            unsafe { (onnx.functions.AllocatorFree)(onnx.allocator, key_ptr as *mut _) };
        }
        if !keys_ptr.is_null() {
            unsafe { (onnx.functions.AllocatorFree)(onnx.allocator, keys_ptr as *mut _) };
        }
        unsafe { (onnx.functions.ReleaseModelMetadata)(model_metadata) };
        Ok((inputs, outputs, metadata))
    }

    pub fn new_cpu(
        onnx: &Arc<Onnx>,
        path: &Path,
        threads: usize,
        optimization_level: ffi::GraphOptimizationLevel,
    ) -> Result<Self, OnnxError> {
        // create session options
        let mut options: *mut ffi::SessionOptions = null_mut();
        let status = unsafe { (onnx.functions.CreateSessionOptions)(&mut options as *mut _) };
        if !status.is_null() {
            return Err(OnnxError::Failed);
        }

        // set graph optimization level
        let status = unsafe {
            (onnx.functions.SetSessionGraphOptimizationLevel)(options, optimization_level)
        };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSessionOptions)(options) };
            return Err(OnnxError::Failed);
        }

        // set intra op num threads
        let status = unsafe { (onnx.functions.SetIntraOpNumThreads)(options, threads as i32) };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSessionOptions)(options) };
            return Err(OnnxError::Failed);
        }

        // process path
        let path_str = path.to_str().unwrap();
        let c_path: CString = match CString::new(path_str) {
            Ok(c_path) => c_path,
            Err(_) => return Err(OnnxError::Failed),
        };

        // create session
        let mut session: *mut ffi::Session = null_mut();
        let status = unsafe {
            (onnx.functions.CreateSession)(
                onnx.env,
                c_path.as_ptr(),
                options,
                &mut session as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseSessionOptions)(options) };
            return Err(OnnxError::Failed);
        }

        // we don't need the options anymore
        unsafe { (onnx.functions.ReleaseSessionOptions)(options) };

        // read inputs, outputs and metadata
        let (inputs, outputs, metadata) = Self::get_inputs_outputs_metadata(onnx, session)?;

        Ok(Session {
            onnx: Arc::clone(&onnx),
            session,
            inputs,
            outputs,
            metadata,
        })
    }

    #[cfg(feature = "cuda")]
    pub fn new_cuda(
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
