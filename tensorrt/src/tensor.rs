pub struct Tensor {
    shape: Vec<i64>,
}

pub trait TensorElement: Copy + 'static {
    const DATA_TYPE: i32;
}

impl TensorElement for f32 {
    const DATA_TYPE: i32 = 0; // kFLOAT
}
impl TensorElement for F16 {
    const DATA_TYPE: i32 = 1; // kHALF
}
impl TensorElement for i8 {
    const DATA_TYPE: i32 = 2; // kINT8
}
impl TensorElement for i32 {
    const DATA_TYPE: i32 = 3; // kINT32
}
impl TensorElement for u8 {
    const DATA_TYPE: i32 = 5; // kUINT8
}

impl Tensor {
    pub fn new<T: TensorElement>(
        tensorrt: &Arc<Tensorrt>,
        shape: &[i64],
    ) -> Result<Self, TensorrtError> {
        Ok(Self {
            shape: shape.to_vec(),
        })
    }

    pub fn from_slice<T: TensorElement>(
        tensorrt: &Arc<Tensorrt>,
        shape: &[i64],
        slice: &[T],
    ) -> Result<Self, TensorrtError> {
        Ok(Self {
            shape: shape.to_vec(),
        })
    }

    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn data<T: TensorElement>(&self) -> &[T] {
        todo!()
    }

    pub fn data_mut<T: TensorElement>(&mut self) -> &mut [T] {
        todo!()
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        todo!()
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        todo!()
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .finish()
    }
}
