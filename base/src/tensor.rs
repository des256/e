use {
    crate::{F16, Mat2x2, Mat3x3, Mat4x4, Vec2, Vec3, Vec4, Zero},
    std::fmt::{self, Display, Formatter},
    std::ops::Add,
};

// -- element trait --

/// Marker trait for types that can be stored in a tensor.
///
/// Covers the ONNX/TensorRT type universe: float types (`f32`, `f64`,
/// [`F16`]), signed integers (`i8`, `i32`, `i64`), and unsigned bytes (`u8`).
pub trait TensorElement: Copy + Zero + 'static {}

impl TensorElement for f32 {}
impl TensorElement for f64 {}
impl TensorElement for F16 {}
impl TensorElement for i8 {}
impl TensorElement for i32 {}
impl TensorElement for i64 {}
impl TensorElement for u8 {}

// -- helpers --

/// Compute row-major strides from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// -- tensor --

/// N-dimensional tensor, generic over element type.
///
/// Memory layout is row-major (C-contiguous) by default, matching ONNX and
/// TensorRT conventions. Non-contiguous views (from [`permute`](Tensor::permute)
/// or [`transpose`](Tensor::transpose)) are supported via explicit strides;
/// call [`to_contiguous`](Tensor::to_contiguous) to materialize a copy when
/// needed.
///
/// # Examples
///
/// ```
/// use base::Tensor;
///
/// let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.get(&[1, 0]), 4.0);
/// ```
#[derive(Clone, Debug)]
pub struct Tensor<T> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: Vec<T>,
}

// -- constructors --

impl<T: TensorElement> Tensor<T> {
    /// Create a tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let numel: usize = shape.iter().product();
        Tensor {
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            data: vec![T::ZERO; numel],
        }
    }

    /// Create a tensor from shape and flat row-major data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` does not equal the product of `shape`.
    pub fn from_shape_data(shape: &[usize], data: Vec<T>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "from_shape_data: expected {} elements for shape {:?}, got {}",
            numel,
            shape,
            data.len(),
        );
        Tensor {
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            data,
        }
    }

    /// Create a tensor from shape, explicit strides, and flat data.
    ///
    /// Use this for wrapping non-contiguous buffers (e.g. transposed TensorRT
    /// outputs). The caller must ensure every reachable index falls within
    /// `data`.
    pub fn from_shape_strides_data(shape: &[usize], strides: &[usize], data: Vec<T>) -> Self {
        assert_eq!(shape.len(), strides.len(), "shape and strides must have equal length");
        Tensor {
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            data,
        }
    }

    /// Create a non-owning view from a raw pointer.
    ///
    /// Useful for zero-copy access to TensorRT device-to-host mapped buffers.
    ///
    /// # Safety
    ///
    /// The caller must ensure `ptr` is valid for `len` elements and that the
    /// returned view does not outlive the buffer.
    pub unsafe fn from_raw_parts<'a>(shape: &[usize], ptr: *const T, len: usize) -> TensorView<'a, T> {
        let data = std::slice::from_raw_parts(ptr, len);
        TensorView {
            shape: shape.to_vec(),
            strides: compute_strides(shape),
            data,
        }
    }
}

// -- shape introspection --

impl<T: TensorElement> Tensor<T> {
    /// Dimension sizes.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Per-dimension element strides (not byte strides).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the data is row-major contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Whether this is a 0-dimensional (scalar) tensor.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }
}

// -- data access --

impl<T: TensorElement> Tensor<T> {
    /// Flat data as a slice. Only meaningful when contiguous.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Flat data as a mutable slice. Only meaningful when contiguous.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Raw pointer to the underlying buffer (for FFI / memcpy).
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Read a single element by multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != ndim()` or any index is out of range.
    pub fn get(&self, indices: &[usize]) -> T {
        assert_eq!(indices.len(), self.ndim(), "get: wrong number of indices");
        let offset = self.flat_offset(indices);
        self.data[offset]
    }

    /// Write a single element by multi-dimensional index.
    ///
    /// # Panics
    ///
    /// Panics if `indices.len() != ndim()` or any index is out of range.
    pub fn set(&mut self, indices: &[usize], value: T) {
        assert_eq!(indices.len(), self.ndim(), "set: wrong number of indices");
        let offset = self.flat_offset(indices);
        self.data[offset] = value;
    }

    /// Select along `dim`, returning a view with that dimension removed.
    ///
    /// E.g. for a `[B, C, H, W]` tensor, `index(0, 2)` returns the third
    /// batch item as a `[C, H, W]` view.
    pub fn index(&self, dim: usize, idx: usize) -> TensorView<'_, T> {
        assert!(dim < self.ndim(), "index: dim out of range");
        assert!(idx < self.shape[dim], "index: idx out of range");
        let offset = idx * self.strides[dim];
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(dim);
        new_strides.remove(dim);
        TensorView {
            shape: new_shape,
            strides: new_strides,
            data: &self.data[offset..],
        }
    }

    fn flat_offset(&self, indices: &[usize]) -> usize {
        let mut offset = 0;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            assert!(idx < self.shape[i], "index {} out of bounds for dim {} (size {})", idx, i, self.shape[i]);
            offset += idx * stride;
        }
        offset
    }
}

// -- reshape / view operations --

impl<T: TensorElement> Tensor<T> {
    /// Reshape to a new shape with the same number of elements.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is not contiguous or the element count changes.
    pub fn reshape(&self, new_shape: &[usize]) -> Tensor<T> {
        assert!(self.is_contiguous(), "reshape requires contiguous tensor");
        let new_numel: usize = new_shape.iter().product();
        assert_eq!(self.numel(), new_numel, "reshape: element count mismatch ({} vs {})", self.numel(), new_numel);
        Tensor {
            shape: new_shape.to_vec(),
            strides: compute_strides(new_shape),
            data: self.data.clone(),
        }
    }

    /// Remove a dimension of size 1.
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of range or `shape[dim] != 1`.
    pub fn squeeze(&self, dim: usize) -> Tensor<T> {
        assert!(dim < self.ndim(), "squeeze: dim out of range");
        assert_eq!(self.shape[dim], 1, "squeeze: dimension {} has size {}, expected 1", dim, self.shape[dim]);
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(dim);
        new_strides.remove(dim);
        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    /// Insert a dimension of size 1 at position `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Tensor<T> {
        assert!(dim <= self.ndim(), "unsqueeze: dim out of range");
        let stride_val = if dim < self.ndim() {
            self.strides[dim] * self.shape[dim]
        } else {
            1
        };
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.insert(dim, 1);
        new_strides.insert(dim, stride_val);
        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    /// Reorder dimensions. Only rearranges strides — no data movement.
    ///
    /// Call [`to_contiguous`](Tensor::to_contiguous) afterwards if you need
    /// contiguous data.
    pub fn permute(&self, dims: &[usize]) -> Tensor<T> {
        assert_eq!(dims.len(), self.ndim(), "permute: wrong number of dims");
        let n = self.ndim();
        let mut new_shape = vec![0usize; n];
        let mut new_strides = vec![0usize; n];
        for (i, &d) in dims.iter().enumerate() {
            assert!(d < n, "permute: dim {} out of range", d);
            new_shape[i] = self.shape[d];
            new_strides[i] = self.strides[d];
        }
        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    /// Swap two dimensions. Only rearranges strides — no data movement.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor<T> {
        assert!(dim0 < self.ndim() && dim1 < self.ndim(), "transpose: dim out of range");
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Tensor {
            shape: new_shape,
            strides: new_strides,
            data: self.data.clone(),
        }
    }

    /// Materialize a contiguous copy. Returns a clone if already contiguous.
    pub fn to_contiguous(&self) -> Tensor<T> {
        if self.is_contiguous() {
            return self.clone();
        }
        let numel = self.numel();
        let ndim = self.ndim();
        let mut data = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            data.push(self.data[self.flat_offset(&indices)]);
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Tensor {
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            data,
        }
    }
}

// -- element-wise CPU operations --

impl<T: TensorElement> Tensor<T> {
    /// Apply a function to every element, producing a new tensor.
    ///
    /// The result is always contiguous. Non-contiguous inputs are iterated
    /// in logical (row-major) order.
    pub fn map<U: TensorElement>(&self, f: impl Fn(T) -> U) -> Tensor<U> {
        let cont = self.to_contiguous();
        Tensor {
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            data: cont.data.iter().map(|&x| f(x)).collect(),
        }
    }

    /// Combine two tensors element-wise. Shapes must match exactly.
    pub fn zip_map<U: TensorElement>(&self, other: &Tensor<T>, f: impl Fn(T, T) -> U) -> Tensor<U> {
        assert_eq!(self.shape, other.shape, "zip_map: shape mismatch");
        let a = self.to_contiguous();
        let b = other.to_contiguous();
        Tensor {
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            data: a.data.iter().zip(b.data.iter()).map(|(&x, &y)| f(x, y)).collect(),
        }
    }

    /// Fill the entire tensor with a single value.
    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T: TensorElement + PartialOrd> Tensor<T> {
    /// Clamp every element to `[min, max]`.
    pub fn clamp(&mut self, min: T, max: T) {
        for x in &mut self.data {
            if *x < min {
                *x = min;
            } else if *x > max {
                *x = max;
            }
        }
    }
}

// -- reduction operations --

impl<T: TensorElement + Add<Output = T>> Tensor<T> {
    /// Sum of all elements.
    pub fn sum(&self) -> T {
        let cont = self.to_contiguous();
        cont.data.iter().copied().reduce(|a, b| a + b).unwrap_or(T::ZERO)
    }
}

impl<T: TensorElement + PartialOrd> Tensor<T> {
    /// Maximum element value.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is empty.
    pub fn max(&self) -> T {
        let cont = self.to_contiguous();
        cont.data.iter().copied().reduce(|a, b| if b > a { b } else { a }).expect("max: empty tensor")
    }

    /// Index of the maximum element in the flat (contiguous) buffer.
    ///
    /// # Panics
    ///
    /// Panics if the tensor is empty.
    pub fn argmax(&self) -> usize {
        let cont = self.to_contiguous();
        cont.data
            .iter()
            .enumerate()
            .reduce(|(ai, av), (bi, bv)| if bv > av { (bi, bv) } else { (ai, av) })
            .expect("argmax: empty tensor")
            .0
    }
}

// -- batch / slice operations --

impl<T: TensorElement> Tensor<T> {
    /// Concatenate tensors along `dim`.
    ///
    /// All tensors must have the same number of dimensions and matching sizes
    /// on every dimension except `dim`.
    pub fn concat(tensors: &[&Tensor<T>], dim: usize) -> Tensor<T> {
        assert!(!tensors.is_empty(), "concat: empty input");
        let ndim = tensors[0].ndim();
        assert!(dim < ndim, "concat: dim out of range");
        for t in &tensors[1..] {
            assert_eq!(t.ndim(), ndim, "concat: ndim mismatch");
            for d in 0..ndim {
                if d != dim {
                    assert_eq!(t.shape[d], tensors[0].shape[d], "concat: shape mismatch on dim {}", d);
                }
            }
        }

        let mut out_shape = tensors[0].shape.clone();
        out_shape[dim] = tensors.iter().map(|t| t.shape[dim]).sum();
        let out_strides = compute_strides(&out_shape);
        let out_numel: usize = out_shape.iter().product();
        let mut data = vec![T::ZERO; out_numel];

        let mut indices = vec![0usize; ndim];
        for i in 0..out_numel {
            // determine source tensor and local index along `dim`
            let mut dim_idx = indices[dim];
            let mut src = 0;
            for (ti, t) in tensors.iter().enumerate() {
                if dim_idx < t.shape[dim] {
                    src = ti;
                    break;
                }
                dim_idx -= t.shape[dim];
            }
            let mut src_indices = indices.clone();
            src_indices[dim] = dim_idx;
            data[i] = tensors[src].get(&src_indices);

            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < out_shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        Tensor { shape: out_shape, strides: out_strides, data }
    }

    /// Extract a contiguous sub-range along `dim`.
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Tensor<T> {
        assert!(dim < self.ndim(), "slice: dim out of range");
        assert!(start <= end && end <= self.shape[dim], "slice: range [{}, {}) out of bounds for dim {} (size {})", start, end, dim, self.shape[dim]);

        let mut new_shape = self.shape.clone();
        new_shape[dim] = end - start;
        let new_strides = compute_strides(&new_shape);
        let numel: usize = new_shape.iter().product();
        let mut data = Vec::with_capacity(numel);

        let ndim = self.ndim();
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let mut src_indices = indices.clone();
            src_indices[dim] += start;
            data.push(self.data[self.flat_offset(&src_indices)]);
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < new_shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        Tensor { shape: new_shape, strides: new_strides, data }
    }

    /// Split into `chunks` pieces along `dim`. The last chunk may be smaller.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Vec<Tensor<T>> {
        assert!(chunks > 0, "chunk: need at least 1 chunk");
        assert!(dim < self.ndim(), "chunk: dim out of range");
        let dim_size = self.shape[dim];
        let chunk_size = (dim_size + chunks - 1) / chunks;
        let mut result = Vec::new();
        let mut start = 0;
        while start < dim_size {
            let end = (start + chunk_size).min(dim_size);
            result.push(self.slice(dim, start, end));
            start = end;
        }
        result
    }
}

// -- float-specific operations (f32 / f64) --

macro_rules! tensor_float_impl {
    ($($t:ty)+) => { $(

        impl Tensor<$t> {
            /// Softmax along `dim`.
            ///
            /// Uses the log-sum-exp trick for numerical stability.
            ///
            /// # Panics
            ///
            /// Panics if the tensor is not contiguous.
            pub fn softmax(&self, dim: usize) -> Tensor<$t> {
                assert!(self.is_contiguous(), "softmax requires contiguous tensor");
                assert!(dim < self.ndim(), "softmax: dim out of range");

                let dim_size = self.shape[dim];
                let outer: usize = self.shape[..dim].iter().product();
                let inner: usize = self.shape[dim + 1..].iter().product();

                let mut result = self.clone();
                for o in 0..outer {
                    for i in 0..inner {
                        // find max for stability
                        let mut max_val = <$t>::NEG_INFINITY;
                        for d in 0..dim_size {
                            let idx = o * dim_size * inner + d * inner + i;
                            if self.data[idx] > max_val {
                                max_val = self.data[idx];
                            }
                        }
                        // exp and sum
                        let mut sum: $t = 0.0;
                        for d in 0..dim_size {
                            let idx = o * dim_size * inner + d * inner + i;
                            let v = (self.data[idx] - max_val).exp();
                            result.data[idx] = v;
                            sum += v;
                        }
                        // normalize
                        for d in 0..dim_size {
                            let idx = o * dim_size * inner + d * inner + i;
                            result.data[idx] /= sum;
                        }
                    }
                }
                result
            }

            /// Per-channel normalization: `(x - mean) / std`.
            ///
            /// Expects NCHW layout (4D) or CHW (3D). `mean` and `std` must
            /// have one entry per channel.
            ///
            /// # Panics
            ///
            /// Panics if the tensor is not contiguous, not 3D/4D, or if
            /// `mean`/`std` lengths don't match the channel count.
            pub fn normalize(&self, mean: &[$t], std: &[$t]) -> Tensor<$t> {
                assert!(self.is_contiguous(), "normalize requires contiguous tensor");
                let (channel_dim, channels) = match self.ndim() {
                    3 => (0, self.shape[0]),
                    4 => (1, self.shape[1]),
                    n => panic!("normalize: expected 3D or 4D tensor, got {}D", n),
                };
                assert_eq!(mean.len(), channels, "normalize: mean length mismatch");
                assert_eq!(std.len(), channels, "normalize: std length mismatch");

                let mut result = self.clone();
                let outer: usize = self.shape[..channel_dim].iter().product();
                let inner: usize = self.shape[channel_dim + 1..].iter().product();
                for o in 0..outer {
                    for c in 0..channels {
                        for i in 0..inner {
                            let idx = o * channels * inner + c * inner + i;
                            result.data[idx] = (result.data[idx] - mean[c]) / std[c];
                        }
                    }
                }
                result
            }

            /// Convert HWC / NHWC layout to NCHW.
            ///
            /// - 3D `[H, W, C]` → `[1, C, H, W]`
            /// - 4D `[N, H, W, C]` → `[N, C, H, W]`
            pub fn nhwc_to_nchw(&self) -> Tensor<$t> {
                match self.ndim() {
                    3 => self.permute(&[2, 0, 1]).unsqueeze(0).to_contiguous(),
                    4 => self.permute(&[0, 3, 1, 2]).to_contiguous(),
                    n => panic!("nhwc_to_nchw: expected 3D or 4D tensor, got {}D", n),
                }
            }
        }

    )+ };
}

tensor_float_impl! { f32 f64 }

// -- TensorView (borrowed, non-owning) --

/// Non-owning view into tensor data.
///
/// Returned by [`Tensor::index`] and [`Tensor::from_raw_parts`]. Provides
/// read-only access with the same shape/stride semantics as [`Tensor`].
/// Call [`to_owned`](TensorView::to_owned) to copy into an owned [`Tensor`].
pub struct TensorView<'a, T> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: &'a [T],
}

impl<'a, T: TensorElement> TensorView<'a, T> {
    /// Dimension sizes.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Per-dimension element strides (not byte strides).
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Whether the data is row-major contiguous.
    pub fn is_contiguous(&self) -> bool {
        self.strides == compute_strides(&self.shape)
    }

    /// Whether this is a 0-dimensional (scalar) view.
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Read a single element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> T {
        assert_eq!(indices.len(), self.ndim(), "get: wrong number of indices");
        let offset = self.flat_offset(indices);
        self.data[offset]
    }

    /// Flat data as a slice. Only meaningful when contiguous.
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Select along `dim`, returning a sub-view.
    pub fn index(&self, dim: usize, idx: usize) -> TensorView<'a, T> {
        assert!(dim < self.ndim(), "index: dim out of range");
        assert!(idx < self.shape[dim], "index: idx out of range");
        let offset = idx * self.strides[dim];
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.remove(dim);
        new_strides.remove(dim);
        TensorView {
            shape: new_shape,
            strides: new_strides,
            data: &self.data[offset..],
        }
    }

    /// Copy into an owned [`Tensor`].
    pub fn to_owned(&self) -> Tensor<T> {
        let numel = self.numel();
        let ndim = self.ndim();
        let mut data = Vec::with_capacity(numel);
        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            data.push(self.data[self.flat_offset(&indices)]);
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        Tensor {
            shape: self.shape.clone(),
            strides: compute_strides(&self.shape),
            data,
        }
    }

    fn flat_offset(&self, indices: &[usize]) -> usize {
        let mut offset = 0;
        for (i, (&idx, &stride)) in indices.iter().zip(self.strides.iter()).enumerate() {
            assert!(idx < self.shape[i], "index {} out of bounds for dim {} (size {})", idx, i, self.shape[i]);
            offset += idx * stride;
        }
        offset
    }
}

// -- From impls --

/// Scalar → 0-dimensional tensor.
impl<T: TensorElement> From<T> for Tensor<T> {
    fn from(value: T) -> Self {
        Tensor {
            shape: vec![],
            strides: vec![],
            data: vec![value],
        }
    }
}

/// Slice → 1D tensor.
impl<T: TensorElement> From<&[T]> for Tensor<T> {
    fn from(data: &[T]) -> Self {
        let len = data.len();
        Tensor {
            shape: vec![len],
            strides: vec![1],
            data: data.to_vec(),
        }
    }
}

/// Vec → 1D tensor.
impl<T: TensorElement> From<Vec<T>> for Tensor<T> {
    fn from(data: Vec<T>) -> Self {
        let len = data.len();
        Tensor {
            shape: vec![len],
            strides: vec![1],
            data,
        }
    }
}

/// Nested Vec → 2D tensor.
impl<T: TensorElement> From<Vec<Vec<T>>> for Tensor<T> {
    fn from(data: Vec<Vec<T>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        assert!(data.iter().all(|r| r.len() == cols), "all rows must have equal length");
        let shape = vec![rows, cols];
        let flat: Vec<T> = data.into_iter().flatten().collect();
        Tensor {
            strides: compute_strides(&shape),
            shape,
            data: flat,
        }
    }
}

// -- From math types --

impl<T: TensorElement> From<Vec2<T>> for Tensor<T> {
    fn from(v: Vec2<T>) -> Self {
        Tensor { shape: vec![2], strides: vec![1], data: vec![v.x, v.y] }
    }
}

impl<T: TensorElement> From<Vec3<T>> for Tensor<T> {
    fn from(v: Vec3<T>) -> Self {
        Tensor { shape: vec![3], strides: vec![1], data: vec![v.x, v.y, v.z] }
    }
}

impl<T: TensorElement> From<Vec4<T>> for Tensor<T> {
    fn from(v: Vec4<T>) -> Self {
        Tensor { shape: vec![4], strides: vec![1], data: vec![v.x, v.y, v.z, v.w] }
    }
}

/// Column-major [`Mat2x2`] → row-major `[2, 2]` tensor.
impl<T: TensorElement> From<Mat2x2<T>> for Tensor<T> {
    fn from(m: Mat2x2<T>) -> Self {
        Tensor {
            shape: vec![2, 2],
            strides: vec![2, 1],
            data: vec![
                m.x.x, m.y.x,
                m.x.y, m.y.y,
            ],
        }
    }
}

/// Column-major [`Mat3x3`] → row-major `[3, 3]` tensor.
impl<T: TensorElement> From<Mat3x3<T>> for Tensor<T> {
    fn from(m: Mat3x3<T>) -> Self {
        Tensor {
            shape: vec![3, 3],
            strides: vec![3, 1],
            data: vec![
                m.x.x, m.y.x, m.z.x,
                m.x.y, m.y.y, m.z.y,
                m.x.z, m.y.z, m.z.z,
            ],
        }
    }
}

/// Column-major [`Mat4x4`] → row-major `[4, 4]` tensor.
impl<T: TensorElement> From<Mat4x4<T>> for Tensor<T> {
    fn from(m: Mat4x4<T>) -> Self {
        Tensor {
            shape: vec![4, 4],
            strides: vec![4, 1],
            data: vec![
                m.x.x, m.y.x, m.z.x, m.w.x,
                m.x.y, m.y.y, m.z.y, m.w.y,
                m.x.z, m.y.z, m.z.z, m.w.z,
                m.x.w, m.y.w, m.z.w, m.w.w,
            ],
        }
    }
}

// -- TryFrom tensor → math types --

impl<T: TensorElement> TryFrom<Tensor<T>> for Vec2<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [2] { return Err("shape must be [2]"); }
        Ok(Vec2 { x: t.data[0], y: t.data[1] })
    }
}

impl<T: TensorElement> TryFrom<Tensor<T>> for Vec3<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [3] { return Err("shape must be [3]"); }
        Ok(Vec3 { x: t.data[0], y: t.data[1], z: t.data[2] })
    }
}

impl<T: TensorElement> TryFrom<Tensor<T>> for Vec4<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [4] { return Err("shape must be [4]"); }
        Ok(Vec4 { x: t.data[0], y: t.data[1], z: t.data[2], w: t.data[3] })
    }
}

impl<T: TensorElement> TryFrom<Tensor<T>> for Mat2x2<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [2, 2] { return Err("shape must be [2, 2]"); }
        let t = t.to_contiguous();
        // row-major tensor → column-major Mat2x2
        Ok(Mat2x2 {
            x: Vec2 { x: t.data[0], y: t.data[2] },
            y: Vec2 { x: t.data[1], y: t.data[3] },
        })
    }
}

impl<T: TensorElement> TryFrom<Tensor<T>> for Mat3x3<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [3, 3] { return Err("shape must be [3, 3]"); }
        let t = t.to_contiguous();
        // row-major tensor → column-major Mat3x3
        Ok(Mat3x3 {
            x: Vec3 { x: t.data[0], y: t.data[3], z: t.data[6] },
            y: Vec3 { x: t.data[1], y: t.data[4], z: t.data[7] },
            z: Vec3 { x: t.data[2], y: t.data[5], z: t.data[8] },
        })
    }
}

impl<T: TensorElement> TryFrom<Tensor<T>> for Mat4x4<T> {
    type Error = &'static str;
    fn try_from(t: Tensor<T>) -> Result<Self, Self::Error> {
        if t.shape != [4, 4] { return Err("shape must be [4, 4]"); }
        let t = t.to_contiguous();
        // row-major tensor → column-major Mat4x4
        Ok(Mat4x4 {
            x: Vec4 { x: t.data[0], y: t.data[4], z: t.data[8],  w: t.data[12] },
            y: Vec4 { x: t.data[1], y: t.data[5], z: t.data[9],  w: t.data[13] },
            z: Vec4 { x: t.data[2], y: t.data[6], z: t.data[10], w: t.data[14] },
            w: Vec4 { x: t.data[3], y: t.data[7], z: t.data[11], w: t.data[15] },
        })
    }
}

// -- F16 ↔ f32 / f64 conversion --

impl From<Tensor<F16>> for Tensor<f32> {
    fn from(t: Tensor<F16>) -> Self {
        t.map(|x| f32::from(x))
    }
}

impl From<Tensor<F16>> for Tensor<f64> {
    fn from(t: Tensor<F16>) -> Self {
        t.map(|x| f64::from(x))
    }
}

impl From<Tensor<f32>> for Tensor<F16> {
    fn from(t: Tensor<f32>) -> Self {
        t.map(|x| F16::from_f32(x))
    }
}

impl From<Tensor<f64>> for Tensor<F16> {
    fn from(t: Tensor<f64>) -> Self {
        t.map(|x| F16::from_f32(x as f32))
    }
}

// -- Display --

impl<T: TensorElement + Display> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let type_name = std::any::type_name::<T>();
        // strip module path, keep just the type name
        let short = type_name.rsplit("::").next().unwrap_or(type_name);
        write!(f, "Tensor<{}>{:?}", short, self.shape)?;
        if self.numel() <= 16 {
            write!(f, " [")?;
            let cont = self.to_contiguous();
            for (i, v) in cont.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", v)?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

impl<T: TensorElement + Display> Display for TensorView<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let type_name = std::any::type_name::<T>();
        let short = type_name.rsplit("::").next().unwrap_or(type_name);
        write!(f, "TensorView<{}>{:?}", short, self.shape)?;
        if self.numel() <= 16 {
            write!(f, " [")?;
            let owned = self.to_owned();
            for (i, v) in owned.data.iter().enumerate() {
                if i > 0 { write!(f, ", ")?; }
                write!(f, "{}", v)?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }
}

// -- PartialEq --

impl<T: TensorElement + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape {
            return false;
        }
        let a = self.to_contiguous();
        let b = other.to_contiguous();
        a.data == b.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    // -- constructors --

    #[test]
    fn test_zeros_creates_correct_shape_and_data() {
        let t = Tensor::<f32>::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.numel(), 6);
        assert!(t.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_zeros_scalar() {
        let t = Tensor::<f32>::zeros(&[]);
        assert!(t.is_scalar());
        assert_eq!(t.numel(), 1);
        assert_eq!(t.as_slice(), &[0.0]);
    }

    #[test]
    fn test_from_shape_data() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 2]), 3.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[1, 2]), 6.0);
    }

    #[test]
    #[should_panic(expected = "expected 6 elements")]
    fn test_from_shape_data_panics_on_mismatch() {
        Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0]);
    }

    #[test]
    fn test_from_shape_strides_data() {
        // column-major 2x3
        let t = Tensor::from_shape_strides_data(
            &[2, 3],
            &[1, 2],
            vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0],
        );
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[1, 0]), 4.0);
        assert_eq!(t.get(&[0, 1]), 2.0);
        assert!(!t.is_contiguous());
    }

    // -- shape introspection --

    #[test]
    fn test_strides_row_major() {
        let t = Tensor::<f32>::zeros(&[2, 3, 4]);
        assert_eq!(t.strides(), &[12, 4, 1]);
    }

    #[test]
    fn test_ndim_and_numel() {
        let t = Tensor::<f32>::zeros(&[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn test_is_contiguous() {
        let t = Tensor::<f32>::zeros(&[2, 3]);
        assert!(t.is_contiguous());

        let t2 = t.transpose(0, 1);
        assert!(!t2.is_contiguous());
    }

    // -- data access --

    #[test]
    fn test_get_set() {
        let mut t = Tensor::<f32>::zeros(&[2, 3]);
        t.set(&[1, 2], 42.0);
        assert_eq!(t.get(&[1, 2]), 42.0);
        assert_eq!(t.get(&[0, 0]), 0.0);
    }

    #[test]
    fn test_index_dim0() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row1 = t.index(0, 1);
        assert_eq!(row1.shape(), &[3]);
        assert_eq!(row1.get(&[0]), 4.0);
        assert_eq!(row1.get(&[1]), 5.0);
        assert_eq!(row1.get(&[2]), 6.0);
    }

    #[test]
    fn test_index_dim1() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let col1 = t.index(1, 1);
        assert_eq!(col1.shape(), &[2]);
        assert_eq!(col1.get(&[0]), 2.0);
        assert_eq!(col1.get(&[1]), 5.0);
    }

    // -- reshape / view --

    #[test]
    fn test_reshape() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = t.reshape(&[3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]), 1.0);
        assert_eq!(r.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let t = Tensor::<f32>::zeros(&[1, 3, 4]);
        let s = t.squeeze(0);
        assert_eq!(s.shape(), &[3, 4]);

        let u = s.unsqueeze(0);
        assert_eq!(u.shape(), &[1, 3, 4]);

        let u2 = s.unsqueeze(2);
        assert_eq!(u2.shape(), &[3, 4, 1]);
    }

    #[test]
    fn test_permute() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = t.permute(&[1, 0]);
        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(p.get(&[0, 0]), 1.0);
        assert_eq!(p.get(&[0, 1]), 4.0);
        assert_eq!(p.get(&[1, 0]), 2.0);
        assert!(!p.is_contiguous());
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tr = t.transpose(0, 1);
        assert_eq!(tr.shape(), &[3, 2]);
        assert_eq!(tr.get(&[2, 0]), 3.0);
        assert_eq!(tr.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_to_contiguous() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let p = t.permute(&[1, 0]); // [3, 2], non-contiguous
        let c = p.to_contiguous();
        assert!(c.is_contiguous());
        assert_eq!(c.shape(), &[3, 2]);
        // row 0 of transposed = column 0 of original
        assert_eq!(c.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -- element-wise --

    #[test]
    fn test_map() {
        let t = Tensor::from_shape_data(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let doubled = t.map(|x| x * 2.0);
        assert_eq!(doubled.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_zip_map() {
        let a = Tensor::from_shape_data(&[3], vec![1.0f32, 2.0, 3.0]);
        let b = Tensor::from_shape_data(&[3], vec![10.0f32, 20.0, 30.0]);
        let c = a.zip_map(&b, |x, y| x + y);
        assert_eq!(c.as_slice(), &[11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_fill() {
        let mut t = Tensor::<f32>::zeros(&[2, 2]);
        t.fill(7.0);
        assert!(t.as_slice().iter().all(|&x| x == 7.0));
    }

    #[test]
    fn test_clamp() {
        let mut t = Tensor::from_shape_data(&[4], vec![-1.0f32, 0.5, 1.5, 3.0]);
        t.clamp(0.0, 2.0);
        assert_eq!(t.as_slice(), &[0.0, 0.5, 1.5, 2.0]);
    }

    // -- reduction --

    #[test]
    fn test_sum() {
        let t = Tensor::from_shape_data(&[4], vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(t.sum(), 10.0);
    }

    #[test]
    fn test_max() {
        let t = Tensor::from_shape_data(&[4], vec![1.0f32, 4.0, 2.0, 3.0]);
        assert_eq!(t.max(), 4.0);
    }

    #[test]
    fn test_argmax() {
        let t = Tensor::from_shape_data(&[4], vec![1.0f32, 4.0, 2.0, 3.0]);
        assert_eq!(t.argmax(), 1);
    }

    // -- batch / slice --

    #[test]
    fn test_concat_dim0() {
        let a = Tensor::from_shape_data(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_shape_data(&[1, 2], vec![5.0f32, 6.0]);
        let c = Tensor::concat(&[&a, &b], 0);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_concat_dim1() {
        let a = Tensor::from_shape_data(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_shape_data(&[2, 1], vec![5.0f32, 6.0]);
        let c = Tensor::concat(&[&a, &b], 1);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.as_slice(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_slice() {
        let t = Tensor::from_shape_data(&[4, 2], vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        let s = t.slice(0, 1, 3);
        assert_eq!(s.shape(), &[2, 2]);
        assert_eq!(s.as_slice(), &[3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_chunk() {
        let t = Tensor::from_shape_data(&[6], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let chunks = t.chunk(3, 0);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_slice(), &[1.0, 2.0]);
        assert_eq!(chunks[1].as_slice(), &[3.0, 4.0]);
        assert_eq!(chunks[2].as_slice(), &[5.0, 6.0]);
    }

    #[test]
    fn test_chunk_uneven() {
        let t = Tensor::from_shape_data(&[5], vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let chunks = t.chunk(2, 0);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(chunks[1].as_slice(), &[4.0, 5.0]);
    }

    // -- softmax --

    #[test]
    fn test_softmax_sums_to_one() {
        let t = Tensor::from_shape_data(&[1, 4], vec![1.0f32, 2.0, 3.0, 4.0]);
        let s = t.softmax(1);
        let sum: f32 = s.as_slice().iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_max_gets_highest_prob() {
        let t = Tensor::from_shape_data(&[1, 3], vec![1.0f32, 5.0, 2.0]);
        let s = t.softmax(1);
        assert_eq!(s.argmax(), 1);
    }

    // -- normalize --

    #[test]
    fn test_normalize_chw() {
        let t = Tensor::from_shape_data(&[2, 1, 1], vec![10.0f32, 20.0]);
        let n = t.normalize(&[5.0, 10.0], &[5.0, 5.0]);
        assert_eq!(n.as_slice(), &[1.0, 2.0]);
    }

    // -- nhwc_to_nchw --

    #[test]
    fn test_nhwc_to_nchw_3d() {
        // [H=1, W=1, C=3] → [1, 3, 1, 1]
        let t = Tensor::from_shape_data(&[1, 1, 3], vec![0.1f32, 0.2, 0.3]);
        let n = t.nhwc_to_nchw();
        assert_eq!(n.shape(), &[1, 3, 1, 1]);
        assert!(n.is_contiguous());
        assert_eq!(n.as_slice(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_nhwc_to_nchw_4d() {
        // [N=1, H=2, W=1, C=2] → [1, 2, 2, 1]
        let t = Tensor::from_shape_data(&[1, 2, 1, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let n = t.nhwc_to_nchw();
        assert_eq!(n.shape(), &[1, 2, 2, 1]);
        // channel 0: [1.0, 3.0], channel 1: [2.0, 4.0]
        assert_eq!(n.get(&[0, 0, 0, 0]), 1.0);
        assert_eq!(n.get(&[0, 0, 1, 0]), 3.0);
        assert_eq!(n.get(&[0, 1, 0, 0]), 2.0);
        assert_eq!(n.get(&[0, 1, 1, 0]), 4.0);
    }

    // -- From impls --

    #[test]
    fn test_from_scalar() {
        let t: Tensor<f32> = 42.0f32.into();
        assert!(t.is_scalar());
        assert_eq!(t.as_slice(), &[42.0]);
    }

    #[test]
    fn test_from_slice() {
        let t: Tensor<f32> = [1.0f32, 2.0, 3.0].as_slice().into();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_from_vec() {
        let t: Tensor<f32> = vec![1.0f32, 2.0].into();
        assert_eq!(t.shape(), &[2]);
    }

    #[test]
    fn test_from_nested_vec() {
        let t: Tensor<f32> = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]].into();
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.get(&[1, 0]), 3.0);
    }

    #[test]
    fn test_from_vec3() {
        let v = vec3(1.0f32, 2.0, 3.0);
        let t: Tensor<f32> = v.into();
        assert_eq!(t.shape(), &[3]);
        assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roundtrip_vec3() {
        let v = vec3(1.0f32, 2.0, 3.0);
        let t: Tensor<f32> = v.into();
        let v2: Vec3<f32> = t.try_into().unwrap();
        assert_eq!(v, v2);
    }

    #[test]
    fn test_from_mat3x3_and_back() {
        let m = mat3x3(
            vec3(1.0f32, 4.0, 7.0),
            vec3(2.0, 5.0, 8.0),
            vec3(3.0, 6.0, 9.0),
        );
        let t: Tensor<f32> = m.into();
        assert_eq!(t.shape(), &[3, 3]);
        // row-major: row 0 = [1, 2, 3] (column x.x, y.x, z.x)
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 2.0);
        assert_eq!(t.get(&[0, 2]), 3.0);

        let m2: Mat3x3<f32> = t.try_into().unwrap();
        assert_eq!(m, m2);
    }

    #[test]
    fn test_from_mat4x4_and_back() {
        let m = mat4x4(
            vec4(1.0f32, 5.0, 9.0, 13.0),
            vec4(2.0, 6.0, 10.0, 14.0),
            vec4(3.0, 7.0, 11.0, 15.0),
            vec4(4.0, 8.0, 12.0, 16.0),
        );
        let t: Tensor<f32> = m.into();
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 3]), 4.0);
        assert_eq!(t.get(&[3, 0]), 13.0);
        assert_eq!(t.get(&[3, 3]), 16.0);

        let m2: Mat4x4<f32> = t.try_into().unwrap();
        assert_eq!(m, m2);
    }

    // -- F16 conversion --

    #[test]
    fn test_f16_to_f32_roundtrip() {
        let t = Tensor::from_shape_data(&[3], vec![1.0f32, 2.0, 3.0]);
        let h: Tensor<F16> = t.clone().into();
        let back: Tensor<f32> = h.into();
        assert_eq!(t, back);
    }

    // -- TensorView --

    #[test]
    fn test_view_to_owned() {
        let t = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let v = t.index(0, 1);
        let owned = v.to_owned();
        assert_eq!(owned.shape(), &[3]);
        assert_eq!(owned.as_slice(), &[4.0, 5.0, 6.0]);
    }

    // -- PartialEq --

    #[test]
    fn test_eq_contiguous() {
        let a = Tensor::from_shape_data(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = Tensor::from_shape_data(&[2, 2], vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(a, b);
    }

    #[test]
    fn test_eq_different_strides_same_data() {
        let a = Tensor::from_shape_data(&[2, 3], vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = a.transpose(0, 1).transpose(0, 1); // back to original shape, different strides path
        assert_eq!(a, b);
    }

    // -- Display --

    #[test]
    fn test_display_small() {
        let t = Tensor::from_shape_data(&[3], vec![1.0f32, 2.0, 3.0]);
        let s = format!("{}", t);
        assert!(s.contains("Tensor<f32>"));
        assert!(s.contains("[3]"));
        assert!(s.contains("1, 2, 3"));
    }
}
