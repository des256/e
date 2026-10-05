use {
    crate::*,
    math::*,
    std::{
        os::raw::c_void,
        ptr::{null, null_mut},
        sync::Arc,
    },
};

pub trait TensorElement: Copy + Zero + 'static {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType;
}

impl TensorElement for i8 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I8;
}

impl TensorElement for u8 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U8;
}

impl TensorElement for i16 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I16;
}

impl TensorElement for u16 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U16;
}

impl TensorElement for u32 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U32;
}

impl TensorElement for i32 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I32;
}

impl TensorElement for u64 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U64;
}

impl TensorElement for i64 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I64;
}

impl TensorElement for F16 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F16;
}

impl TensorElement for f32 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F32;
}

impl TensorElement for f64 {
    const ONNX_DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F64;
}

pub enum Storage {
    Onnx {
        value: *mut ffi::Value,
        element_type: ffi::TensorElementDataType,
    },
    #[cfg(feature = "tensorrt")]
    Tensorrt(cuda::CudaBuffer),
}

pub struct TensorData<'a> {
    runtime: Arc<Runtime>,
    storage: &'a Storage,
}

pub struct TensorDataMut<'a> {
    runtime: Arc<Runtime>,
    storage: &'a mut Storage,
}

pub struct Tensor {
    runtime: Arc<Runtime>,
    shape: Vec<i64>,
    strides: Vec<usize>,
    storage: Storage,
}

fn compute_strides(shape: &[i64]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as usize;
    }
    strides
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            runtime: Arc::clone(&self.runtime),
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            storage: match self.storage {
                Storage::Onnx {
                    value,
                    element_type,
                } => {
                    let mut data_ptr: *mut c_void = null_mut();
                    let status = unsafe {
                        (self.runtime.functions.GetTensorMutableData)(
                            value,
                            &mut data_ptr as *mut _,
                        )
                    };
                    if !status.is_null() {
                        panic!("Failed to get mutable data from ONNX value");
                    }
                    let info: *const ffi::MemoryInfo = null();
                    // TODO: create memory info
                    let mut new_value: *mut ffi::Value = null_mut();
                    let status = unsafe {
                        (self.runtime.functions.CreateTensorWithDataAsOrtValue)(
                            info as *const _,
                            data_ptr,
                            self.shape.iter().product::<i64>() as usize * size_of::<T>(),
                            self.shape.as_ptr(),
                            self.shape.len(),
                            element_type,
                            &mut new_value as *mut _,
                        )
                    };
                    if !status.is_null() {
                        panic!("Failed to create new ONNX value");
                    }
                    // TODO: release memory info
                    Storage::Onnx {
                        value: new_value,
                        element_type,
                    }
                }
                #[cfg(feature = "tensorrt")]
                Storage::Tensorrt(buffer) => {
                    let new_buffer = cuda::CudaBuffer::new(buffer.size).unwrap();
                    new_buffer.copy_from(&buffer, buffer.size).unwrap();
                    Storage::Tensorrt(new_buffer)
                }
            },
        }
    }
}

impl<T: TensorElement + std::fmt::Debug> std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("strides", &self.strides)
            .field(
                "storage",
                &match self.storage {
                    Storage::Onnx { .. } => "Onnx",
                    #[cfg(feature = "tensorrt")]
                    Storage::Tensorrt(buffer) => "Tensorrt",
                },
            )
            .finish()
    }
}

impl Tensor {
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn is_contiguous(&self) -> bool {
        let strides = compute_strides(&self.shape);
        for (i, stride) in strides.iter().enumerate() {
            if stride != &self.strides[i] {
                return false;
            }
        }
        true
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    pub fn data(&self) -> TensorData<'_> {
        TensorData {
            runtime: Arc::clone(&self.runtime),
            storage: &self.storage,
        }
    }

    pub fn data_mut(&mut self) -> TensorDataMut<'_> {
        TensorDataMut {
            runtime: Arc::clone(&self.runtime),
            storage: &mut self.storage,
        }
    }

    pub fn zeros_onnx<T: TensorElement>(runtime: &Arc<Runtime>, shape: &[i64]) -> Self {
        let mut value: *mut ffi::Value = null_mut();
        let status = unsafe {
            (runtime.functions.CreateTensorAsOrtValue)(
                runtime.allocator,
                shape.as_ptr() as *const i64,
                shape.len(),
                T::ONNX_DATA_TYPE,
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            panic!("Failed to create ONNX value");
        }
        Tensor {
            runtime: Arc::clone(&runtime),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            storage: Storage::Onnx {
                value,
                element_type: T::ONNX_DATA_TYPE,
            },
        }
    }

    #[cfg(feature = "tensorrt")]
    pub fn zeros_tensorrt<T: TensorElement>(runtime: &Arc<Runtime>, shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        Tensor {
            runtime: Arc::clone(&runtime),
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            storage: Storage::Tensorrt(cuda::CudaBuffer::new(size).unwrap()),
            marker: PhantomData,
        }
    }
}
