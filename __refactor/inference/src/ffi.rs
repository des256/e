use std::os::raw::{c_char, c_void};

#[repr(C)]
pub struct Env {
    _private: [u8; 0],
}

unsafe impl Send for Env {}
unsafe impl Sync for Env {}

#[repr(C)]
pub struct Session {
    _private: [u8; 0],
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

#[repr(C)]
pub struct SessionOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Value {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Status {
    _private: [u8; 0],
}

#[repr(C)]
pub struct MemoryInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct Allocator {
    _private: [u8; 0],
}

#[repr(C)]
pub struct RunOptions {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TensorTypeAndShapeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct TypeInfo {
    _private: [u8; 0],
}

#[repr(C)]
pub struct ModelMetadata {
    _private: [u8; 0],
}

#[repr(C)]
pub enum LoggingLevel {
    Verbose = 0,
    Info = 1,
    Warning = 2,
    Error = 3,
    Fatal = 4,
}

#[repr(C)]
pub enum TensorElementDataType {
    Undefined = 0,
    F32 = 1,
    U8 = 2,
    I8 = 3,
    U16 = 4,
    I16 = 5,
    I32 = 6,
    I64 = 7,
    String = 8,
    Bool = 9,
    F16 = 10,
    F64 = 11,
    U32 = 12,
    U64 = 13,
    C32 = 14,
    C64 = 15,
    BF16 = 16,
}

#[repr(C)]
pub enum ErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineError = 5,
    RuntimeException = 6,
    InvalidProtobuf = 7,
    ModelLoaded = 8,
    NotImplemented = 9,
    InvalidGraph = 10,
    EpFail = 11,
}

#[repr(C)]
pub enum MemType {
    CpuInput = -2,
    CpuOutput = -1,
    Default = 0,
}

#[repr(C)]
pub enum AllocatorType {
    Invalid = -1,
    Device = 0,
    Arena = 1,
}

#[repr(C)]
#[allow(non_snake_case)]
pub struct ApiBase {
    pub GetApi: unsafe extern "C" fn(version: u32) -> *const Api,
    pub GetVersionString: unsafe extern "C" fn() -> *const c_char,
}

unsafe impl Send for ApiBase {}

#[repr(C)]
pub struct Api {
    _private: [u8; 0],
}

unsafe impl Send for Api {}
unsafe impl Sync for Api {}

impl Api {
    pub unsafe fn get_fn<F>(&self, index: usize) -> F {
        unsafe {
            let vtable = self as *const _ as *const *const ();
            let fn_ptr = *vtable.add(index);
            std::mem::transmute_copy(&fn_ptr)
        }
    }
}

unsafe extern "C" {
    pub unsafe fn OrtGetApiBase() -> *const ApiBase;
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub unsafe fn OrtSessionOptionsAppendExecutionProvider_CUDA(
        options: *mut SessionOptions,
        device_id: i32,
    ) -> *mut Status;
}

#[allow(non_snake_case)]
pub struct Functions {
    pub CreateStatusFn: unsafe extern "C" fn(code: ErrorCode, msg: *const c_char) -> *mut Status,
    pub GetErrorCodeFn: unsafe extern "C" fn(status: *const Status) -> ErrorCode,
    pub GetErrorMessage: unsafe extern "C" fn(status: *const Status) -> *const c_char,
    pub CreateEnv: unsafe extern "C" fn(
        log_level: LoggingLevel,
        log_id: *const c_char,
        out: *mut *mut Env,
    ) -> *mut Status,
    pub CreateSession: unsafe extern "C" fn(
        env: *const Env,
        model_path: *const c_char,
        options: *const SessionOptions,
        out: *mut *mut Session,
    ) -> *mut Status,
    pub Run: unsafe extern "C" fn(
        session: *mut Session,
        run_options: *const RunOptions,
        input_names: *const *const c_char,
        inputs: *const *const Value,
        input_len: usize,
        output_names: *const *const c_char,
        output_names_len: usize,
        outputs: *mut *mut Value,
    ) -> *mut Status,
    pub CreateSessionOptions: unsafe extern "C" fn(out: *mut *mut SessionOptions) -> *mut Status,
    pub SetSessionGraphOptimizationLevel:
        unsafe extern "C" fn(options: *mut SessionOptions, level: usize) -> *mut Status,
    pub SetIntraOpNumThreads:
        unsafe extern "C" fn(options: *mut SessionOptions, num_threads: i32) -> *mut Status,
    pub SessionGetInputCount:
        unsafe extern "C" fn(session: *const Session, out: *mut usize) -> *mut Status,
    pub SessionGetOutputCount:
        unsafe extern "C" fn(session: *const Session, out: *mut usize) -> *mut Status,
    pub SessionGetInputTypeInfo: unsafe extern "C" fn(
        session: *const Session,
        index: usize,
        type_info: *mut *mut TypeInfo,
    ) -> *mut Status,
    pub SessionGetOutputTypeInfo: unsafe extern "C" fn(
        session: *const Session,
        index: usize,
        type_info: *mut *mut TypeInfo,
    ) -> *mut Status,
    pub SessionGetInputName: unsafe extern "C" fn(
        session: *const Session,
        index: usize,
        allocator: *mut Allocator,
        value: *mut *mut c_char,
    ) -> *mut Status,
    pub SessionGetOutputName: unsafe extern "C" fn(
        session: *const Session,
        index: usize,
        allocator: *mut Allocator,
        value: *mut *mut c_char,
    ) -> *mut Status,
    pub CreateTensorWithDataAsOrtValue: unsafe extern "C" fn(
        memory_info: *const MemoryInfo,
        data: *mut c_void,
        data_len: usize,
        shape: *const i64,
        shape_len: usize,
        element_type: TensorElementDataType,
        out: *mut *mut Value,
    ) -> *mut Status,
    pub CreateTensorAsOrtValue: unsafe extern "C" fn(
        allocator: *mut Allocator,
        shape: *const i64,
        shape_len: usize,
        element_type: TensorElementDataType,
        out: *mut *mut Value,
    ) -> *mut Status,
    pub GetTensorMutableData:
        unsafe extern "C" fn(value: *mut Value, out: *mut *mut c_void) -> *mut Status,
    pub CastTypeInfoToTensorInfo: unsafe extern "C" fn(
        type_info: *const TypeInfo,
        out: *mut *const TensorTypeAndShapeInfo,
    ) -> *mut Status,
    pub GetTensorElementType: unsafe extern "C" fn(
        info: *const TensorTypeAndShapeInfo,
        out: *mut TensorElementDataType,
    ) -> *mut Status,
    pub GetDimensionsCount:
        unsafe extern "C" fn(info: *const TensorTypeAndShapeInfo, out: *mut usize) -> *mut Status,
    pub GetDimensions: unsafe extern "C" fn(
        info: *const TensorTypeAndShapeInfo,
        dim_values: *mut i64,
        dim_count: usize,
    ) -> *mut Status,
    pub GetTensorShapeElementCount:
        unsafe extern "C" fn(info: *const TensorTypeAndShapeInfo, out: *mut usize) -> *mut Status,
    pub GetTensorTypeAndShape: unsafe extern "C" fn(
        value: *const Value,
        out: *mut *mut TensorTypeAndShapeInfo,
    ) -> *mut Status,
    pub CreateCpuMemoryInfo: unsafe extern "C" fn(
        allocator_type: AllocatorType,
        mem_type: MemType,
        out: *mut *mut MemoryInfo,
    ) -> *mut Status,
    pub AllocatorFree: unsafe extern "C" fn(allocator: *mut Allocator, ptr: *mut c_void),
    pub GetAllocatorWithDefaultOptions:
        unsafe extern "C" fn(out: *mut *mut Allocator) -> *mut Status,
    pub ReleaseEnv: unsafe extern "C" fn(env: *mut Env),
    pub ReleaseStatus: unsafe extern "C" fn(status: *mut Status),
    pub ReleaseMemoryInfo: unsafe extern "C" fn(info: *mut MemoryInfo),
    pub ReleaseSession: unsafe extern "C" fn(session: *mut Session),
    pub ReleaseValue: unsafe extern "C" fn(value: *mut Value),
    pub ReleaseTypeInfo: unsafe extern "C" fn(info: *mut TypeInfo),
    pub ReleaseTensorTypeAndShapeInfo: unsafe extern "C" fn(info: *mut TensorTypeAndShapeInfo),
    pub ReleaseSessionOptions: unsafe extern "C" fn(options: *mut SessionOptions),
    pub SessionGetModelMetadata:
        unsafe extern "C" fn(session: *const Session, out: *mut *mut ModelMetadata) -> *mut Status,
    pub ModelMetadataLookupCustomMetadataMap: unsafe extern "C" fn(
        model_metadata: *const ModelMetadata,
        allocator: *mut Allocator,
        key: *const c_char,
        value: *mut *mut c_char,
    ) -> *mut Status,
    pub ReleaseModelMetadata: unsafe extern "C" fn(metadata: *mut ModelMetadata),
    pub ModelMetadataGetCustomMetadataMapKeys: unsafe extern "C" fn(
        model_metadata: *const ModelMetadata,
        allocator: *mut Allocator,
        keys: *mut *mut *mut c_char,
        num_keys: *mut i64,
    ) -> *mut Status,
    pub GetAvailableProviders: unsafe extern "C" fn(
        out_ptr: *mut *mut *mut c_char,
        provider_length: *mut i32,
    ) -> *mut Status,
    pub ReleaseAvailableProviders:
        unsafe extern "C" fn(ptr: *mut *mut c_char, providers_length: i32) -> *mut Status,
}

impl Functions {
    pub const IDX_CREATE_STATUS: usize = 0;
    pub const IDX_GET_ERROR_CODE: usize = 1;
    pub const IDX_GET_ERROR_MESSAGE: usize = 2;
    pub const IDX_CREATE_ENV: usize = 3;
    pub const IDX_CREATE_SESSION: usize = 7;
    pub const IDX_RUN: usize = 9;
    pub const IDX_CREATE_SESSION_OPTIONS: usize = 10;
    pub const IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL: usize = 23;
    pub const IDX_SET_INTRA_OP_NUM_THREADS: usize = 24;
    pub const IDX_SESSION_GET_INPUT_COUNT: usize = 30;
    pub const IDX_SESSION_GET_OUTPUT_COUNT: usize = 31;
    pub const IDX_SESSION_GET_INPUT_TYPE_INFO: usize = 33;
    pub const IDX_SESSION_GET_OUTPUT_TYPE_INFO: usize = 34;
    pub const IDX_SESSION_GET_INPUT_NAME: usize = 36;
    pub const IDX_SESSION_GET_OUTPUT_NAME: usize = 37;
    pub const IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE: usize = 49;
    pub const IDX_CREATE_TENSOR_AS_ORT_VALUE: usize = 50;
    pub const IDX_GET_TENSOR_MUTABLE_DATA: usize = 51;
    pub const IDX_CAST_TYPE_INFO_TO_TENSOR_INFO: usize = 55;
    pub const IDX_GET_TENSOR_ELEMENT_TYPE: usize = 60;
    pub const IDX_GET_DIMENSIONS_COUNT: usize = 61;
    pub const IDX_GET_DIMENSIONS: usize = 62;
    pub const IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT: usize = 64;
    pub const IDX_GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
    pub const IDX_CREATE_CPU_MEMORY_INFO: usize = 69;
    pub const IDX_ALLOCATOR_FREE: usize = 76;
    pub const IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
    pub const IDX_RELEASE_ENV: usize = 92;
    pub const IDX_RELEASE_STATUS: usize = 93;
    pub const IDX_RELEASE_MEMORY_INFO: usize = 94;
    pub const IDX_RELEASE_SESSION: usize = 95;
    pub const IDX_RELEASE_VALUE: usize = 96;
    pub const IDX_RELEASE_TYPE_INFO: usize = 98;
    pub const IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO: usize = 99;
    pub const IDX_RELEASE_SESSION_OPTIONS: usize = 100;
    pub const IDX_SESSION_GET_MODEL_METADATA: usize = 111;
    pub const IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP: usize = 116;
    pub const IDX_RELEASE_MODEL_METADATA: usize = 118;
    pub const IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS: usize = 123;
    pub const IDX_GET_AVAILABLE_PROVIDERS: usize = 125;
    pub const IDX_RELEASE_AVAILABLE_PROVIDERS: usize = 126;

    pub fn new(api: *const Api) -> Self {
        unsafe {
            let api = &*api;
            Self {
                CreateStatusFn: api.get_fn(Self::IDX_CREATE_STATUS),
                GetErrorCodeFn: api.get_fn(Self::IDX_GET_ERROR_CODE),
                GetErrorMessage: api.get_fn(Self::IDX_GET_ERROR_MESSAGE),
                CreateEnv: api.get_fn(Self::IDX_CREATE_ENV),
                CreateSession: api.get_fn(Self::IDX_CREATE_SESSION),
                Run: api.get_fn(Self::IDX_RUN),
                CreateSessionOptions: api.get_fn(Self::IDX_CREATE_SESSION_OPTIONS),
                SetSessionGraphOptimizationLevel: api
                    .get_fn(Self::IDX_SET_SESSION_GRAPH_OPTIMIZATION_LEVEL),
                SetIntraOpNumThreads: api.get_fn(Self::IDX_SET_INTRA_OP_NUM_THREADS),
                SessionGetInputCount: api.get_fn(Self::IDX_SESSION_GET_INPUT_COUNT),
                SessionGetOutputCount: api.get_fn(Self::IDX_SESSION_GET_OUTPUT_COUNT),
                SessionGetInputTypeInfo: api.get_fn(Self::IDX_SESSION_GET_INPUT_TYPE_INFO),
                SessionGetOutputTypeInfo: api.get_fn(Self::IDX_SESSION_GET_OUTPUT_TYPE_INFO),
                SessionGetInputName: api.get_fn(Self::IDX_SESSION_GET_INPUT_NAME),
                SessionGetOutputName: api.get_fn(Self::IDX_SESSION_GET_OUTPUT_NAME),
                CreateTensorWithDataAsOrtValue: api
                    .get_fn(Self::IDX_CREATE_TENSOR_WITH_DATA_AS_ORT_VALUE),
                CreateTensorAsOrtValue: api.get_fn(Self::IDX_CREATE_TENSOR_AS_ORT_VALUE),
                GetTensorMutableData: api.get_fn(Self::IDX_GET_TENSOR_MUTABLE_DATA),
                CastTypeInfoToTensorInfo: api.get_fn(Self::IDX_CAST_TYPE_INFO_TO_TENSOR_INFO),
                GetTensorElementType: api.get_fn(Self::IDX_GET_TENSOR_ELEMENT_TYPE),
                GetDimensionsCount: api.get_fn(Self::IDX_GET_DIMENSIONS_COUNT),
                GetDimensions: api.get_fn(Self::IDX_GET_DIMENSIONS),
                GetTensorShapeElementCount: api.get_fn(Self::IDX_GET_TENSOR_SHAPE_ELEMENT_COUNT),
                GetTensorTypeAndShape: api.get_fn(Self::IDX_GET_TENSOR_TYPE_AND_SHAPE),
                CreateCpuMemoryInfo: api.get_fn(Self::IDX_CREATE_CPU_MEMORY_INFO),
                AllocatorFree: api.get_fn(Self::IDX_ALLOCATOR_FREE),
                GetAllocatorWithDefaultOptions: api
                    .get_fn(Self::IDX_GET_ALLOCATOR_WITH_DEFAULT_OPTIONS),
                ReleaseEnv: api.get_fn(Self::IDX_RELEASE_ENV),
                ReleaseStatus: api.get_fn(Self::IDX_RELEASE_STATUS),
                ReleaseMemoryInfo: api.get_fn(Self::IDX_RELEASE_MEMORY_INFO),
                ReleaseSession: api.get_fn(Self::IDX_RELEASE_SESSION),
                ReleaseValue: api.get_fn(Self::IDX_RELEASE_VALUE),
                ReleaseTypeInfo: api.get_fn(Self::IDX_RELEASE_TYPE_INFO),
                ReleaseTensorTypeAndShapeInfo: api
                    .get_fn(Self::IDX_RELEASE_TENSOR_TYPE_AND_SHAPE_INFO),
                ReleaseSessionOptions: api.get_fn(Self::IDX_RELEASE_SESSION_OPTIONS),
                SessionGetModelMetadata: api.get_fn(Self::IDX_SESSION_GET_MODEL_METADATA),
                ModelMetadataLookupCustomMetadataMap: api
                    .get_fn(Self::IDX_MODEL_METADATA_LOOKUP_CUSTOM_METADATA_MAP),
                ReleaseModelMetadata: api.get_fn(Self::IDX_RELEASE_MODEL_METADATA),
                ModelMetadataGetCustomMetadataMapKeys: api
                    .get_fn(Self::IDX_MODEL_METADATA_GET_CUSTOM_METADATA_MAP_KEYS),
                GetAvailableProviders: api.get_fn(Self::IDX_GET_AVAILABLE_PROVIDERS),
                ReleaseAvailableProviders: api.get_fn(Self::IDX_RELEASE_AVAILABLE_PROVIDERS),
            }
        }
    }
}
