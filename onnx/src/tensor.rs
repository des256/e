use {
    crate::*,
    math::*,
    std::{
        os::raw::c_void,
        ptr::{copy_nonoverlapping, null_mut},
        sync::Arc,
    },
};

pub trait TensorElement: Copy + Zero + 'static {
    const DATA_TYPE: ffi::TensorElementDataType;
}

impl TensorElement for u8 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U8;
}
impl TensorElement for i8 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I8;
}
impl TensorElement for u16 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U16;
}
impl TensorElement for i16 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I16;
}
impl TensorElement for u32 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U32;
}
impl TensorElement for i32 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I32;
}
impl TensorElement for u64 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::U64;
}
impl TensorElement for i64 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::I64;
}
impl TensorElement for F16 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F16;
}
impl TensorElement for f32 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F32;
}
impl TensorElement for f64 {
    const DATA_TYPE: ffi::TensorElementDataType = ffi::TensorElementDataType::F64;
}

pub struct Tensor {
    onnx: Arc<Onnx>,
    shape: Vec<i64>,
    value: *mut ffi::Value,
    element_type: ffi::TensorElementDataType,
    element_size: usize,
}

impl Tensor {
    pub fn new<T: TensorElement>(onnx: &Arc<Onnx>, shape: &[i64]) -> Result<Self, OnnxError> {
        let mut value: *mut ffi::Value = null_mut();
        let status = unsafe {
            (onnx.functions.CreateTensorAsOrtValue)(
                onnx.allocator,
                shape.as_ptr() as *const i64,
                shape.len(),
                T::DATA_TYPE,
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            return Err(OnnxError::Failed);
        }
        Ok(Self {
            onnx: Arc::clone(&onnx),
            shape: shape.to_vec(),
            value,
            element_type: T::DATA_TYPE,
            element_size: size_of::<T>(),
        })
    }

    pub fn from_slice<T: TensorElement>(
        onnx: &Arc<Onnx>,
        shape: &[i64],
        slice: &[T],
    ) -> Result<Self, OnnxError> {
        let mut value: *mut ffi::Value = null_mut();
        let mut memory_info: *mut ffi::MemoryInfo = null_mut();
        let status = unsafe {
            (onnx.functions.CreateCpuMemoryInfo)(
                ffi::AllocatorType::Device,
                ffi::MemType::CpuOutput,
                &mut memory_info as *mut _,
            )
        };
        if !status.is_null() {
            return Err(OnnxError::Failed);
        }
        let status = unsafe {
            (onnx.functions.CreateTensorWithDataAsOrtValue)(
                memory_info,
                slice.as_ptr() as *mut _,
                slice.len() * size_of::<T>(),
                shape.as_ptr(),
                shape.len(),
                T::DATA_TYPE,
                &mut value as *mut _,
            )
        };
        if !status.is_null() {
            unsafe { (onnx.functions.ReleaseMemoryInfo)(memory_info) };
            return Err(OnnxError::Failed);
        }
        Ok(Self {
            onnx: Arc::clone(&onnx),
            shape: shape.to_vec(),
            value,
            element_type: T::DATA_TYPE,
            element_size: size_of::<T>(),
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn data<T: TensorElement>(&self) -> &[T] {
        if self.element_type != T::DATA_TYPE {
            panic!("Tensor element type mismatch");
        }
        let mut data_ptr: *mut c_void = null_mut();
        let status = unsafe {
            (self.onnx.functions.GetTensorMutableData)(self.value, &mut data_ptr as *mut _)
        };
        if !status.is_null() {
            panic!("Failed to get mutable data from ONNX value");
        }
        unsafe {
            std::slice::from_raw_parts(
                data_ptr as *const T,
                self.shape.iter().product::<i64>() as usize,
            )
        }
    }

    pub fn data_mut<T: TensorElement>(&mut self) -> &mut [T] {
        if self.element_type != T::DATA_TYPE {
            panic!("Tensor element type mismatch");
        }
        let mut data_ptr: *mut c_void = null_mut();
        let status = unsafe {
            (self.onnx.functions.GetTensorMutableData)(self.value, &mut data_ptr as *mut _)
        };
        if !status.is_null() {
            panic!("Failed to get mutable data from ONNX value");
        }
        unsafe {
            std::slice::from_raw_parts_mut(
                data_ptr as *mut T,
                self.shape.iter().product::<i64>() as usize,
            )
        }
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        unsafe { (self.onnx.functions.ReleaseValue)(self.value) };
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        // create new ONNX value
        let mut value: *mut ffi::Value = null_mut();
        let status = unsafe {
            (self.onnx.functions.CreateTensorAsOrtValue)(
                self.onnx.allocator,
                self.shape.as_ptr() as *const i64,
                self.shape.len(),
                self.element_type,
                &mut value as *mut _,
            )
        };

        if !status.is_null() {
            panic!("Failed to create ONNX value");
        }

        // get source data pointer
        let mut src_data_ptr: *mut c_void = null_mut();
        let status = unsafe {
            (self.onnx.functions.GetTensorMutableData)(self.value, &mut src_data_ptr as *mut _)
        };
        if !status.is_null() {
            unsafe { (self.onnx.functions.ReleaseValue)(value) };
            panic!("Failed to get mutable data from ONNX value");
        }

        // get destination data pointer
        let mut dst_data_ptr: *mut c_void = null_mut();
        let status = unsafe {
            (self.onnx.functions.GetTensorMutableData)(value, &mut dst_data_ptr as *mut _)
        };
        if !status.is_null() {
            unsafe { (self.onnx.functions.ReleaseValue)(value) };
            panic!("Failed to get mutable data from ONNX value");
        }

        // copy the data
        unsafe {
            copy_nonoverlapping(
                src_data_ptr,
                dst_data_ptr,
                self.shape.iter().product::<i64>() as usize * self.element_size,
            )
        };

        Self {
            onnx: Arc::clone(&self.onnx),
            shape: self.shape.clone(),
            value,
            element_type: self.element_type,
            element_size: self.element_size,
        }
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("value", &self.value)
            .finish()
    }
}
