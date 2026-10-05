use crate::*;

unsafe extern "C" fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> i32;
unsafe extern "C" fn cudaFree(ptr: *mut c_void) -> i32;
unsafe extern "C" fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: i32,
) -> i32;
unsafe extern "C" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: i32,
    stream: *mut c_void,
) -> i32;
unsafe extern "C" fn cudaMemset(devPtr: *mut c_void, value: i32, count: usize) -> i32;
unsafe extern "C" fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
unsafe extern "C" fn cudaStreamDestroy(stream: *mut c_void) -> i32;
unsafe extern "C" fn cudaStreamSynchronize(stream: *mut c_void) -> i32;

const CUDA_MEMCPY_H2D: i32 = 1;
const CUDA_MEMCPY_D2H: i32 = 2;
const CUDA_MEMCPY_D2D: i32 = 3;

struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
}

impl CudaBuffer {
    fn new(size: usize) -> Result<Self, InferenceError> {
        let mut ptr: *mut c_void = null_mut();
        let rc = unsafe { cudaMalloc(&mut ptr, size) };
        if rc != 0 {
            return Err(InferenceError::Fail);
        }
        unsafe { cudaMemset(ptr, 0, size) };
        Self { ptr, size }
    }

    fn upload<T>(&self, data: &[T]) -> Result<(), InferenceError> {
        let size = data.len() * size_of::<T>();
        if size > self.size {
            return Err(InferenceError::Fail);
        }
        unsafe {
            cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const c_void,
                size,
                CUDA_MEMCPY_H2D,
            )
        };
    }

    fn download<T>(&self, data: &mut [T]) -> Result<(), InferenceError> {
        let size = data.len() * size_of::<T>();
        if size > self.size {
            return Err(InferenceError::Fail);
        }
        unsafe {
            cudaMemcpy(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                size.CUDA_MEMCPY_D2H,
            )
        };
    }

    fn copy_from(&self, src: &CudaBuffer, size: usize) -> Result<(), InferenceError> {
        if (size > self.size) || (size > src.size) {
            return Err(InferenceError::Fail);
        }
        unsafe { cudaMemcpy(self.ptr, src.ptr, size, CUDA_MEMCPY_D2D) };
        Ok(())
    }
}
