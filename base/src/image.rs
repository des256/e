use crate::{
    Argb8, Bgra8, F16, LinearFormat, Nv12, PackedFormat, PixelFormat, Rgba8, Rgba8Pixel, RgbaF16,
    RgbaF16Pixel, RgbaF32, RgbaF32Pixel, Srgba8, Srgba8Pixel, Tensor, UncompressedFormat, Uv8,
    Uv8Pixel, Y8, Y8Pixel, Yuyv8,
};
use std::marker::PhantomData;

// ============================================================
// Image<P> — owned image buffer
// ============================================================

/// Owned image with pixel format `P`.
///
/// Stores raw bytes with an explicit stride (bytes per row) to support
/// GPU texture padding and hardware buffer layouts. The pixel format is
/// encoded at the type level via the `P` parameter.
pub struct Image<P: PixelFormat> {
    width: u32,
    height: u32,
    stride: u32,
    data: Vec<u8>,
    _format: PhantomData<P>,
}

// -- constructors --

impl<P: UncompressedFormat> Image<P> {
    /// Create a zero-filled image with the minimum stride.
    pub fn new(width: u32, height: u32) -> Self {
        let stride = P::min_stride(width);
        let size = P::data_size(width, height, stride);
        Image {
            width,
            height,
            stride,
            data: vec![0u8; size],
            _format: PhantomData,
        }
    }

    /// Create a zero-filled image with a custom stride.
    ///
    /// # Panics
    ///
    /// Panics if `stride < P::min_stride(width)`.
    pub fn new_with_stride(width: u32, height: u32, stride: u32) -> Self {
        assert!(
            stride >= P::min_stride(width),
            "stride {} < min_stride {} for {}",
            stride, P::min_stride(width), P::NAME,
        );
        let size = P::data_size(width, height, stride);
        Image {
            width,
            height,
            stride,
            data: vec![0u8; size],
            _format: PhantomData,
        }
    }

    /// Create an image from existing data with the minimum stride.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is less than the required buffer size.
    pub fn from_data(width: u32, height: u32, data: Vec<u8>) -> Self {
        let stride = P::min_stride(width);
        let size = P::data_size(width, height, stride);
        assert!(
            data.len() >= size,
            "from_data: data.len() {} < required {} for {}x{} {}",
            data.len(), size, width, height, P::NAME,
        );
        Image {
            width,
            height,
            stride,
            data,
            _format: PhantomData,
        }
    }

    /// Create an image from existing data with a custom stride.
    ///
    /// # Panics
    ///
    /// Panics if `stride < min_stride` or `data.len() < data_size`.
    pub fn from_raw(width: u32, height: u32, stride: u32, data: Vec<u8>) -> Self {
        assert!(
            stride >= P::min_stride(width),
            "from_raw: stride {} < min_stride {} for {}",
            stride, P::min_stride(width), P::NAME,
        );
        let size = P::data_size(width, height, stride);
        assert!(
            data.len() >= size,
            "from_raw: data.len() {} < required {} for {}x{} {}",
            data.len(), size, width, height, P::NAME,
        );
        Image {
            width,
            height,
            stride,
            data,
            _format: PhantomData,
        }
    }
}

// -- dimensions & byte access --

impl<P: PixelFormat> Image<P> {
    /// Image width in pixels.
    pub fn width(&self) -> u32 { self.width }
    /// Image height in pixels.
    pub fn height(&self) -> u32 { self.height }
    /// Bytes per row (may exceed `width × bytes_per_pixel` due to padding).
    pub fn stride(&self) -> u32 { self.stride }

    /// Raw byte buffer (read-only).
    pub fn as_bytes(&self) -> &[u8] { &self.data }
    /// Raw byte buffer (mutable).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] { &mut self.data }
    /// Consume the image and return the underlying byte buffer.
    pub fn into_bytes(self) -> Vec<u8> { self.data }
}

// -- row access --

impl<P: UncompressedFormat> Image<P> {
    /// Byte slice for row `y` (stride-wide).
    ///
    /// # Panics
    ///
    /// Panics if `y >= height`.
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row: y={} >= height={}", y, self.height);
        let start = y as usize * self.stride as usize;
        &self.data[start..start + self.stride as usize]
    }

    /// Mutable byte slice for row `y` (stride-wide).
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row_mut: y={} >= height={}", y, self.height);
        let start = y as usize * self.stride as usize;
        &mut self.data[start..start + self.stride as usize]
    }
}

// -- view creation --

impl<P: PixelFormat> Image<P> {
    /// Borrow the image as a read-only [`ImageView`].
    pub fn view(&self) -> ImageView<'_, P> {
        ImageView {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &self.data,
            _format: PhantomData,
        }
    }

    /// Borrow the image as a mutable [`ImageViewMut`].
    pub fn view_mut(&mut self) -> ImageViewMut<'_, P> {
        ImageViewMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.data,
            _format: PhantomData,
        }
    }
}

// ============================================================
// ImageView<'a, P> — borrowed image slice
// ============================================================

/// Borrowed image slice, analogous to `&[T]`.
///
/// Zero-copy: references a region of an existing buffer.
pub struct ImageView<'a, P: PixelFormat> {
    width: u32,
    height: u32,
    stride: u32,
    data: &'a [u8],
    _format: PhantomData<P>,
}

impl<'a, P: PixelFormat> ImageView<'a, P> {
    /// Image width in pixels.
    pub fn width(&self) -> u32 { self.width }
    /// Image height in pixels.
    pub fn height(&self) -> u32 { self.height }
    /// Bytes per row.
    pub fn stride(&self) -> u32 { self.stride }
    /// Raw byte slice.
    pub fn as_bytes(&self) -> &[u8] { self.data }
}

impl<'a, P: UncompressedFormat> ImageView<'a, P> {
    /// Create a view from raw parts.
    ///
    /// # Panics
    ///
    /// Panics if stride or data size constraints are violated.
    pub fn from_raw(width: u32, height: u32, stride: u32, data: &'a [u8]) -> Self {

        assert!(
            stride >= P::min_stride(width),
            "ImageView::from_raw: stride {} < min_stride {} for {}",
            stride, P::min_stride(width), P::NAME,
        );
        let size = P::data_size(width, height, stride);
        assert!(
            data.len() >= size,
            "ImageView::from_raw: data.len() {} < required {} for {}x{} {}",
            data.len(), size, width, height, P::NAME,
        );
        ImageView { width, height, stride, data, _format: PhantomData }
    }

    /// Byte slice for row `y` (stride-wide).
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row: y={} >= height={}", y, self.height);
        let start = y as usize * self.stride as usize;
        &self.data[start..start + self.stride as usize]
    }
}

// ============================================================
// ImageViewMut<'a, P> — mutable borrowed image slice
// ============================================================

/// Mutable borrowed image slice.
pub struct ImageViewMut<'a, P: PixelFormat> {
    width: u32,
    height: u32,
    stride: u32,
    data: &'a mut [u8],
    _format: PhantomData<P>,
}

impl<'a, P: PixelFormat> ImageViewMut<'a, P> {
    /// Image width in pixels.
    pub fn width(&self) -> u32 { self.width }
    /// Image height in pixels.
    pub fn height(&self) -> u32 { self.height }
    /// Bytes per row.
    pub fn stride(&self) -> u32 { self.stride }
    /// Raw byte slice.
    pub fn as_bytes(&self) -> &[u8] { self.data }
    /// Raw byte slice (mutable).
    pub fn as_bytes_mut(&mut self) -> &mut [u8] { self.data }
}

impl<'a, P: UncompressedFormat> ImageViewMut<'a, P> {
    /// Byte slice for row `y` (stride-wide).
    pub fn row(&self, y: u32) -> &[u8] {
        assert!(y < self.height, "row: y={} >= height={}", y, self.height);
        let start = y as usize * self.stride as usize;
        &self.data[start..start + self.stride as usize]
    }

    /// Mutable byte slice for row `y` (stride-wide).
    pub fn row_mut(&mut self, y: u32) -> &mut [u8] {
        assert!(y < self.height, "row_mut: y={} >= height={}", y, self.height);
        let start = y as usize * self.stride as usize;
        &mut self.data[start..start + self.stride as usize]
    }
}

// ============================================================
// Pixel access — LinearFormat
// ============================================================

impl<P: LinearFormat> Image<P> {
    /// Read a pixel by (x, y) coordinate.
    ///
    /// # Panics
    ///
    /// Panics if `x >= width` or `y >= height`.
    pub fn pixel(&self, x: u32, y: u32) -> &P::Pixel {
        assert!(x < self.width && y < self.height,
            "pixel: ({},{}) out of bounds for {}x{}", x, y, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        let ptr = self.data[offset..].as_ptr();
        debug_assert!(ptr as usize % std::mem::align_of::<P::Pixel>() == 0,
            "pixel: unaligned pointer for {}", P::NAME);
        unsafe { &*(ptr as *const P::Pixel) }
    }

    /// Write a pixel by (x, y) coordinate.
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut P::Pixel {
        assert!(x < self.width && y < self.height,
            "pixel_mut: ({},{}) out of bounds for {}x{}", x, y, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        let ptr = self.data[offset..].as_mut_ptr();
        debug_assert!(ptr as usize % std::mem::align_of::<P::Pixel>() == 0,
            "pixel_mut: unaligned pointer for {}", P::NAME);
        unsafe { &mut *(ptr as *mut P::Pixel) }
    }
}

impl<'a, P: LinearFormat> ImageView<'a, P> {
    /// Read a pixel by (x, y) coordinate.
    pub fn pixel(&self, x: u32, y: u32) -> &P::Pixel {
        assert!(x < self.width && y < self.height,
            "pixel: ({},{}) out of bounds for {}x{}", x, y, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        let ptr = self.data[offset..].as_ptr();
        debug_assert!(ptr as usize % std::mem::align_of::<P::Pixel>() == 0,
            "pixel: unaligned pointer for {}", P::NAME);
        unsafe { &*(ptr as *const P::Pixel) }
    }
}

impl<'a, P: LinearFormat> ImageViewMut<'a, P> {
    /// Read a pixel by (x, y) coordinate.
    pub fn pixel(&self, x: u32, y: u32) -> &P::Pixel {
        assert!(x < self.width && y < self.height,
            "pixel: ({},{}) out of bounds for {}x{}", x, y, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        let ptr = self.data[offset..].as_ptr();
        debug_assert!(ptr as usize % std::mem::align_of::<P::Pixel>() == 0,
            "pixel: unaligned pointer for {}", P::NAME);
        unsafe { &*(ptr as *const P::Pixel) }
    }

    /// Write a pixel by (x, y) coordinate.
    pub fn pixel_mut(&mut self, x: u32, y: u32) -> &mut P::Pixel {
        assert!(x < self.width && y < self.height,
            "pixel_mut: ({},{}) out of bounds for {}x{}", x, y, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        let ptr = self.data[offset..].as_mut_ptr();
        debug_assert!(ptr as usize % std::mem::align_of::<P::Pixel>() == 0,
            "pixel_mut: unaligned pointer for {}", P::NAME);
        unsafe { &mut *(ptr as *mut P::Pixel) }
    }
}

// ============================================================
// Macropixel access — PackedFormat
// ============================================================

impl<P: PackedFormat> Image<P> {
    /// Read a macropixel by (mx, y) where mx is the macropixel index.
    pub fn macropixel(&self, mx: u32, y: u32) -> &P::Macropixel {
        let max_mx = self.width / P::PIXELS_PER_MACROPIXEL as u32;
        assert!(mx < max_mx && y < self.height,
            "macropixel: ({},{}) out of bounds for {}x{}", mx, y, max_mx, self.height);
        let offset = y as usize * self.stride as usize + mx as usize * P::BYTES_PER_MACROPIXEL;
        unsafe { &*(self.data[offset..].as_ptr() as *const P::Macropixel) }
    }

    /// Write a macropixel by (mx, y) where mx is the macropixel index.
    pub fn macropixel_mut(&mut self, mx: u32, y: u32) -> &mut P::Macropixel {
        let max_mx = self.width / P::PIXELS_PER_MACROPIXEL as u32;
        assert!(mx < max_mx && y < self.height,
            "macropixel_mut: ({},{}) out of bounds for {}x{}", mx, y, max_mx, self.height);
        let offset = y as usize * self.stride as usize + mx as usize * P::BYTES_PER_MACROPIXEL;
        unsafe { &mut *(self.data[offset..].as_mut_ptr() as *mut P::Macropixel) }
    }
}

impl<'a, P: PackedFormat> ImageView<'a, P> {
    /// Read a macropixel by (mx, y) where mx is the macropixel index.
    pub fn macropixel(&self, mx: u32, y: u32) -> &P::Macropixel {
        let max_mx = self.width / P::PIXELS_PER_MACROPIXEL as u32;
        assert!(mx < max_mx && y < self.height,
            "macropixel: ({},{}) out of bounds for {}x{}", mx, y, max_mx, self.height);
        let offset = y as usize * self.stride as usize + mx as usize * P::BYTES_PER_MACROPIXEL;
        unsafe { &*(self.data[offset..].as_ptr() as *const P::Macropixel) }
    }
}

// ============================================================
// Nv12 plane access
// ============================================================

impl Image<Nv12> {
    /// Y luminance plane as an `ImageView<Y8>`.
    pub fn y_plane(&self) -> ImageView<'_, Y8> {
        let y_size = self.stride as usize * self.height as usize;
        ImageView {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &self.data[..y_size],
            _format: PhantomData,
        }
    }

    /// Interleaved UV chroma plane as an `ImageView<Uv8>`.
    ///
    /// The UV plane has half the width and height of the Y plane but
    /// the same byte stride (width/2 samples × 2 bytes = width bytes).
    pub fn uv_plane(&self) -> ImageView<'_, Uv8> {
        let y_size = self.stride as usize * self.height as usize;
        ImageView {
            width: self.width / 2,
            height: self.height / 2,
            stride: self.stride,
            data: &self.data[y_size..],
            _format: PhantomData,
        }
    }

    /// Mutable Y luminance plane as an `ImageViewMut<Y8>`.
    pub fn y_plane_mut(&mut self) -> ImageViewMut<'_, Y8> {
        let y_size = self.stride as usize * self.height as usize;
        ImageViewMut {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &mut self.data[..y_size],
            _format: PhantomData,
        }
    }

    /// Mutable interleaved UV chroma plane as an `ImageViewMut<Uv8>`.
    pub fn uv_plane_mut(&mut self) -> ImageViewMut<'_, Uv8> {
        let y_size = self.stride as usize * self.height as usize;
        let uv_height = self.height / 2;
        let width = self.width;
        let stride = self.stride;
        ImageViewMut {
            width: width / 2,
            height: uv_height,
            stride,
            data: &mut self.data[y_size..],
            _format: PhantomData,
        }
    }
}

impl<'a> ImageView<'a, Nv12> {
    /// Y luminance plane as an `ImageView<Y8>`.
    pub fn y_plane(&self) -> ImageView<'a, Y8> {
        let y_size = self.stride as usize * self.height as usize;
        ImageView {
            width: self.width,
            height: self.height,
            stride: self.stride,
            data: &self.data[..y_size],
            _format: PhantomData,
        }
    }

    /// Interleaved UV chroma plane as an `ImageView<Uv8>`.
    pub fn uv_plane(&self) -> ImageView<'a, Uv8> {
        let y_size = self.stride as usize * self.height as usize;
        ImageView {
            width: self.width / 2,
            height: self.height / 2,
            stride: self.stride,
            data: &self.data[y_size..],
            _format: PhantomData,
        }
    }
}

// ============================================================
// sub_image — LinearFormat only
// ============================================================

impl<'a, P: LinearFormat> ImageView<'a, P> {
    /// Zero-copy crop: returns a view into a sub-region.
    ///
    /// # Panics
    ///
    /// Panics if the sub-region exceeds the image bounds.
    pub fn sub_image(&self, x: u32, y: u32, width: u32, height: u32) -> ImageView<'a, P> {
        assert!(x + width <= self.width && y + height <= self.height,
            "sub_image: region ({},{})..({},{}) exceeds {}x{}",
            x, y, x + width, y + height, self.width, self.height);
        let offset = y as usize * self.stride as usize + x as usize * P::BYTES_PER_PIXEL;
        ImageView {
            width,
            height,
            stride: self.stride,
            data: &self.data[offset..],
            _format: PhantomData,
        }
    }
}

// ============================================================
// Format conversion
// ============================================================

/// Trait for converting between pixel formats.
///
/// Defined on the source format type, parameterized by the target.
/// Implementations live in `image.rs` alongside `Image<P>`.
pub trait ConvertTo<Q: PixelFormat>: PixelFormat + Sized {
    /// Convert `src` to the target pixel format `Q`.
    fn convert(src: &Image<Self>) -> Image<Q>;
}

impl<P: PixelFormat> Image<P> {
    /// Convert this image to a different pixel format.
    pub fn convert<Q: PixelFormat>(&self) -> Image<Q>
    where
        P: ConvertTo<Q>,
    {
        P::convert(self)
    }
}

// -- sRGB gamma helpers --

fn srgb_to_linear(s: f32) -> f32 {
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(l: f32) -> f32 {
    if l <= 0.0031308 {
        l * 12.92
    } else {
        1.055 * l.powf(1.0 / 2.4) - 0.055
    }
}

fn f32_to_u8(v: f32) -> u8 {
    (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

// -- YUV→RGB (BT.601) helper --

fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let yf = y as f32;
    let uf = u as f32 - 128.0;
    let vf = v as f32 - 128.0;
    let r = (yf + 1.402 * vf).clamp(0.0, 255.0) as u8;
    let g = (yf - 0.344136 * uf - 0.714136 * vf).clamp(0.0, 255.0) as u8;
    let b = (yf + 1.772 * uf).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

// -- Channel reorder: Rgba8 ↔ Argb8 ↔ Bgra8 (6 impls) --

fn convert_linear_4ch<Src: LinearFormat, Dst: LinearFormat>(
    src: &Image<Src>,
    map: fn(&[u8]) -> [u8; 4],
) -> Image<Dst> {
    let mut dst = Image::<Dst>::new(src.width(), src.height());
    let dst_stride = dst.stride();
    for y in 0..src.height() {
        let src_row = &src.as_bytes()[y as usize * src.stride() as usize..];
        let dst_row = &mut dst.as_bytes_mut()[y as usize * dst_stride as usize..];
        for x in 0..src.width() as usize {
            let si = x * 4;
            let out = map(&src_row[si..si + 4]);
            dst_row[si..si + 4].copy_from_slice(&out);
        }
    }
    dst
}

impl ConvertTo<Argb8> for Rgba8 {
    fn convert(src: &Image<Self>) -> Image<Argb8> {
        convert_linear_4ch(src, |p| [p[3], p[0], p[1], p[2]])
    }
}

impl ConvertTo<Bgra8> for Rgba8 {
    fn convert(src: &Image<Self>) -> Image<Bgra8> {
        convert_linear_4ch(src, |p| [p[2], p[1], p[0], p[3]])
    }
}

impl ConvertTo<Rgba8> for Argb8 {
    fn convert(src: &Image<Self>) -> Image<Rgba8> {
        convert_linear_4ch(src, |p| [p[1], p[2], p[3], p[0]])
    }
}

impl ConvertTo<Bgra8> for Argb8 {
    fn convert(src: &Image<Self>) -> Image<Bgra8> {
        convert_linear_4ch(src, |p| [p[3], p[2], p[1], p[0]])
    }
}

impl ConvertTo<Rgba8> for Bgra8 {
    fn convert(src: &Image<Self>) -> Image<Rgba8> {
        convert_linear_4ch(src, |p| [p[2], p[1], p[0], p[3]])
    }
}

impl ConvertTo<Argb8> for Bgra8 {
    fn convert(src: &Image<Self>) -> Image<Argb8> {
        convert_linear_4ch(src, |p| [p[3], p[2], p[1], p[0]])
    }
}

// -- sRGB ↔ linear float --

impl ConvertTo<RgbaF32> for Srgba8 {
    fn convert(src: &Image<Self>) -> Image<RgbaF32> {
        let mut dst = Image::<RgbaF32>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                let r = srgb_to_linear(p.r as f32 / 255.0);
                let g = srgb_to_linear(p.g as f32 / 255.0);
                let b = srgb_to_linear(p.b as f32 / 255.0);
                let a = p.a as f32 / 255.0; // alpha is always linear
                *dst.pixel_mut(x, y) = RgbaF32Pixel { r, g, b, a };
            }
        }
        dst
    }
}

impl ConvertTo<Srgba8> for RgbaF32 {
    fn convert(src: &Image<Self>) -> Image<Srgba8> {
        let mut dst = Image::<Srgba8>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                let r = f32_to_u8(linear_to_srgb(p.r));
                let g = f32_to_u8(linear_to_srgb(p.g));
                let b = f32_to_u8(linear_to_srgb(p.b));
                let a = f32_to_u8(p.a); // alpha stays linear
                *dst.pixel_mut(x, y) = Srgba8Pixel { r, g, b, a };
            }
        }
        dst
    }
}

// -- Precision: Rgba8 ↔ RgbaF32 --

impl ConvertTo<RgbaF32> for Rgba8 {
    fn convert(src: &Image<Self>) -> Image<RgbaF32> {
        let mut dst = Image::<RgbaF32>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                *dst.pixel_mut(x, y) = RgbaF32Pixel {
                    r: p.r as f32 / 255.0,
                    g: p.g as f32 / 255.0,
                    b: p.b as f32 / 255.0,
                    a: p.a as f32 / 255.0,
                };
            }
        }
        dst
    }
}

impl ConvertTo<Rgba8> for RgbaF32 {
    fn convert(src: &Image<Self>) -> Image<Rgba8> {
        let mut dst = Image::<Rgba8>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                *dst.pixel_mut(x, y) = Rgba8Pixel {
                    r: f32_to_u8(p.r),
                    g: f32_to_u8(p.g),
                    b: f32_to_u8(p.b),
                    a: f32_to_u8(p.a),
                };
            }
        }
        dst
    }
}

// -- Precision: RgbaF16 ↔ RgbaF32 --

impl ConvertTo<RgbaF32> for RgbaF16 {
    fn convert(src: &Image<Self>) -> Image<RgbaF32> {
        let mut dst = Image::<RgbaF32>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                *dst.pixel_mut(x, y) = RgbaF32Pixel {
                    r: p.r.to_f32(),
                    g: p.g.to_f32(),
                    b: p.b.to_f32(),
                    a: p.a.to_f32(),
                };
            }
        }
        dst
    }
}

impl ConvertTo<RgbaF16> for RgbaF32 {
    fn convert(src: &Image<Self>) -> Image<RgbaF16> {
        let mut dst = Image::<RgbaF16>::new(src.width(), src.height());
        for y in 0..src.height() {
            for x in 0..src.width() {
                let p = src.pixel(x, y);
                *dst.pixel_mut(x, y) = RgbaF16Pixel {
                    r: F16::from_f32(p.r),
                    g: F16::from_f32(p.g),
                    b: F16::from_f32(p.b),
                    a: F16::from_f32(p.a),
                };
            }
        }
        dst
    }
}

// -- Video: Yuyv8 → Rgba8 --

impl ConvertTo<Rgba8> for Yuyv8 {
    fn convert(src: &Image<Self>) -> Image<Rgba8> {
        let mut dst = Image::<Rgba8>::new(src.width(), src.height());
        for y in 0..src.height() {
            let macropixels = src.width() / 2;
            for mx in 0..macropixels {
                let m = src.macropixel(mx, y);
                let (r0, g0, b0) = yuv_to_rgb(m.y0, m.u, m.v);
                let (r1, g1, b1) = yuv_to_rgb(m.y1, m.u, m.v);
                *dst.pixel_mut(mx * 2, y) = Rgba8Pixel { r: r0, g: g0, b: b0, a: 255 };
                *dst.pixel_mut(mx * 2 + 1, y) = Rgba8Pixel { r: r1, g: g1, b: b1, a: 255 };
            }
        }
        dst
    }
}

// -- Video: Nv12 → Rgba8 --

impl ConvertTo<Rgba8> for Nv12 {
    fn convert(src: &Image<Self>) -> Image<Rgba8> {
        let mut dst = Image::<Rgba8>::new(src.width(), src.height());
        let y_plane = src.y_plane();
        let uv_plane = src.uv_plane();
        for py in 0..src.height() {
            for px in 0..src.width() {
                let y_val = y_plane.pixel(px, py).y;
                let uv = uv_plane.pixel(px / 2, py / 2);
                let (r, g, b) = yuv_to_rgb(y_val, uv.u, uv.v);
                *dst.pixel_mut(px, py) = Rgba8Pixel { r, g, b, a: 255 };
            }
        }
        dst
    }
}

// ============================================================
// Tensor interop
// ============================================================

/// Trait for converting between pixel data and f32 channel arrays.
///
/// Implemented for all [`LinearFormat`] types to enable tensor interop.
pub trait TensorConvertible: LinearFormat {
    /// Number of channels (e.g. 4 for RGBA, 1 for Y8).
    const CHANNELS: usize;
    /// Extract f32 channel values from a pixel.
    fn pixel_to_channels(pixel: &Self::Pixel, out: &mut [f32]);
    /// Reconstruct a pixel from f32 channel values.
    fn channels_to_pixel(channels: &[f32]) -> Self::Pixel;
}

impl TensorConvertible for Rgba8 {
    const CHANNELS: usize = 4;
    fn pixel_to_channels(p: &Rgba8Pixel, out: &mut [f32]) {
        out[0] = p.r as f32 / 255.0;
        out[1] = p.g as f32 / 255.0;
        out[2] = p.b as f32 / 255.0;
        out[3] = p.a as f32 / 255.0;
    }
    fn channels_to_pixel(c: &[f32]) -> Rgba8Pixel {
        Rgba8Pixel {
            r: f32_to_u8(c[0]),
            g: f32_to_u8(c[1]),
            b: f32_to_u8(c[2]),
            a: f32_to_u8(c[3]),
        }
    }
}

impl TensorConvertible for Srgba8 {
    const CHANNELS: usize = 4;
    fn pixel_to_channels(p: &Srgba8Pixel, out: &mut [f32]) {
        out[0] = p.r as f32 / 255.0;
        out[1] = p.g as f32 / 255.0;
        out[2] = p.b as f32 / 255.0;
        out[3] = p.a as f32 / 255.0;
    }
    fn channels_to_pixel(c: &[f32]) -> Srgba8Pixel {
        Srgba8Pixel {
            r: f32_to_u8(c[0]),
            g: f32_to_u8(c[1]),
            b: f32_to_u8(c[2]),
            a: f32_to_u8(c[3]),
        }
    }
}

impl TensorConvertible for RgbaF32 {
    const CHANNELS: usize = 4;
    fn pixel_to_channels(p: &RgbaF32Pixel, out: &mut [f32]) {
        out[0] = p.r;
        out[1] = p.g;
        out[2] = p.b;
        out[3] = p.a;
    }
    fn channels_to_pixel(c: &[f32]) -> RgbaF32Pixel {
        RgbaF32Pixel { r: c[0], g: c[1], b: c[2], a: c[3] }
    }
}

impl TensorConvertible for RgbaF16 {
    const CHANNELS: usize = 4;
    fn pixel_to_channels(p: &RgbaF16Pixel, out: &mut [f32]) {
        out[0] = p.r.to_f32();
        out[1] = p.g.to_f32();
        out[2] = p.b.to_f32();
        out[3] = p.a.to_f32();
    }
    fn channels_to_pixel(c: &[f32]) -> RgbaF16Pixel {
        RgbaF16Pixel {
            r: F16::from_f32(c[0]),
            g: F16::from_f32(c[1]),
            b: F16::from_f32(c[2]),
            a: F16::from_f32(c[3]),
        }
    }
}

impl TensorConvertible for Y8 {
    const CHANNELS: usize = 1;
    fn pixel_to_channels(p: &Y8Pixel, out: &mut [f32]) {
        out[0] = p.y as f32 / 255.0;
    }
    fn channels_to_pixel(c: &[f32]) -> Y8Pixel {
        Y8Pixel { y: f32_to_u8(c[0]) }
    }
}

impl TensorConvertible for Uv8 {
    const CHANNELS: usize = 2;
    fn pixel_to_channels(p: &Uv8Pixel, out: &mut [f32]) {
        out[0] = p.u as f32 / 255.0;
        out[1] = p.v as f32 / 255.0;
    }
    fn channels_to_pixel(c: &[f32]) -> Uv8Pixel {
        Uv8Pixel {
            u: f32_to_u8(c[0]),
            v: f32_to_u8(c[1]),
        }
    }
}

impl<P: TensorConvertible> Image<P> {
    /// Convert this image to a `Tensor<f32>` in NCHW layout.
    ///
    /// Output shape: `[1, C, H, W]` where C = number of channels.
    /// Values are normalized to `[0, 1]` for integer formats.
    pub fn to_tensor_nchw(&self) -> Tensor<f32> {
        let c = P::CHANNELS;
        let h = self.height() as usize;
        let w = self.width() as usize;
        let mut data = vec![0.0f32; c * h * w];
        let mut channels = vec![0.0f32; c];
        for y in 0..self.height() {
            for x in 0..self.width() {
                P::pixel_to_channels(self.pixel(x, y), &mut channels);
                for ch in 0..c {
                    data[ch * h * w + y as usize * w + x as usize] = channels[ch];
                }
            }
        }
        Tensor::from_shape_data(&[1, c, h, w], data)
    }

    /// Reconstruct an image from a `Tensor<f32>` in NCHW layout.
    ///
    /// Accepts shape `[1, C, H, W]` or `[C, H, W]`.
    ///
    /// # Panics
    ///
    /// Panics if the channel count doesn't match or the shape is wrong.
    pub fn from_tensor_nchw(tensor: &Tensor<f32>) -> Self {
        let (c, h, w) = match tensor.shape() {
            [1, c, h, w] => (*c, *h, *w),
            [c, h, w] => (*c, *h, *w),
            other => panic!(
                "from_tensor_nchw: expected [1,C,H,W] or [C,H,W], got {:?}",
                other,
            ),
        };
        assert_eq!(c, P::CHANNELS,
            "from_tensor_nchw: channel mismatch, tensor has {} but {} expects {}",
            c, P::NAME, P::CHANNELS);
        let mut img = Image::<P>::new(w as u32, h as u32);
        let mut channels = vec![0.0f32; c];
        let data = tensor.as_slice();
        for y in 0..h {
            for x in 0..w {
                for ch in 0..c {
                    channels[ch] = data[ch * h * w + y * w + x];
                }
                *img.pixel_mut(x as u32, y as u32) = P::channels_to_pixel(&channels);
            }
        }
        img
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    // -- constructors --

    #[test]
    fn test_image_new_rgba8() {
        let img = Image::<Rgba8>::new(4, 3);
        assert_eq!(img.width(), 4);
        assert_eq!(img.height(), 3);
        assert_eq!(img.stride(), 16); // 4 * 4
        assert_eq!(img.as_bytes().len(), 48); // 16 * 3
        assert!(img.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn test_image_new_with_stride() {
        let img = Image::<Rgba8>::new_with_stride(4, 3, 32);
        assert_eq!(img.stride(), 32);
        assert_eq!(img.as_bytes().len(), 96); // 32 * 3
    }

    #[test]
    #[should_panic(expected = "stride 8 < min_stride 16")]
    fn test_image_new_with_stride_too_small() {
        Image::<Rgba8>::new_with_stride(4, 3, 8);
    }

    #[test]
    fn test_image_from_data() {
        let data = vec![1u8; 48]; // 4*4*3
        let img = Image::<Rgba8>::from_data(4, 3, data);
        assert_eq!(img.width(), 4);
        assert_eq!(img.as_bytes()[0], 1);
    }

    #[test]
    #[should_panic(expected = "from_data: data.len() 10 < required 48")]
    fn test_image_from_data_too_small() {
        Image::<Rgba8>::from_data(4, 3, vec![0u8; 10]);
    }

    #[test]
    fn test_image_from_raw() {
        let data = vec![0u8; 96]; // stride=32, height=3
        let img = Image::<Rgba8>::from_raw(4, 3, 32, data);
        assert_eq!(img.stride(), 32);
        assert_eq!(img.as_bytes().len(), 96);
    }

    #[test]
    #[should_panic(expected = "from_raw: stride 8 < min_stride 16")]
    fn test_image_from_raw_bad_stride() {
        Image::<Rgba8>::from_raw(4, 3, 8, vec![0u8; 100]);
    }

    // -- byte access --

    #[test]
    fn test_image_as_bytes_mut() {
        let mut img = Image::<Rgba8>::new(2, 2);
        img.as_bytes_mut()[0] = 42;
        assert_eq!(img.as_bytes()[0], 42);
    }

    #[test]
    fn test_image_into_bytes() {
        let img = Image::<Rgba8>::new(2, 2);
        let bytes = img.into_bytes();
        assert_eq!(bytes.len(), 16); // 2*4*2
    }

    // -- row access --

    #[test]
    fn test_image_row() {
        let mut img = Image::<Rgba8>::new(2, 3);
        // Write to row 1 start
        img.as_bytes_mut()[8] = 99; // row1 starts at stride*1 = 8
        assert_eq!(img.row(1)[0], 99);
        assert_eq!(img.row(0)[0], 0);
    }

    #[test]
    fn test_image_row_mut() {
        let mut img = Image::<Rgba8>::new(2, 2);
        img.row_mut(1)[0] = 55;
        assert_eq!(img.row(1)[0], 55);
    }

    #[test]
    #[should_panic(expected = "row: y=3 >= height=3")]
    fn test_image_row_oob() {
        let img = Image::<Rgba8>::new(2, 3);
        img.row(3);
    }

    // -- view creation --

    #[test]
    fn test_image_view() {
        let img = Image::<Rgba8>::new(4, 3);
        let view = img.view();
        assert_eq!(view.width(), 4);
        assert_eq!(view.height(), 3);
        assert_eq!(view.stride(), 16);
        assert_eq!(view.as_bytes().len(), 48);
    }

    #[test]
    fn test_image_view_mut() {
        let mut img = Image::<Rgba8>::new(4, 3);
        {
            let mut view = img.view_mut();
            view.as_bytes_mut()[0] = 77;
        }
        assert_eq!(img.as_bytes()[0], 77);
    }

    #[test]
    fn test_image_view_row() {
        let img = Image::<Rgba8>::new(2, 2);
        let view = img.view();
        assert_eq!(view.row(0).len(), 8); // stride = 2*4 = 8
        assert_eq!(view.row(1).len(), 8);
    }

    // -- ImageView from_raw --

    #[test]
    fn test_image_view_from_raw() {
        let data = vec![0u8; 48];
        let view = ImageView::<Rgba8>::from_raw(4, 3, 16, &data);
        assert_eq!(view.width(), 4);
        assert_eq!(view.height(), 3);
    }

    // -- different format types --

    #[test]
    fn test_image_rgbaf32() {
        let img = Image::<RgbaF32>::new(2, 2);
        assert_eq!(img.stride(), 32); // 2 * 16
        assert_eq!(img.as_bytes().len(), 64); // 32 * 2
    }

    #[test]
    fn test_image_y8() {
        let img = Image::<Y8>::new(640, 480);
        assert_eq!(img.stride(), 640);
        assert_eq!(img.as_bytes().len(), 307200);
    }

    #[test]
    fn test_image_nv12() {
        let img = Image::<Nv12>::new(640, 480);
        assert_eq!(img.stride(), 640);
        // Y=640*480 + UV=640*240 = 460800
        assert_eq!(img.as_bytes().len(), 460800);
    }

    #[test]
    fn test_image_yuyv8() {
        let img = Image::<Yuyv8>::new(640, 480);
        assert_eq!(img.stride(), 1280); // 640 * 2
        assert_eq!(img.as_bytes().len(), 614400); // 1280 * 480
    }

    // -- pixel access (LinearFormat) --

    #[test]
    fn test_pixel_write_read_rgba8() {
        let mut img = Image::<Rgba8>::new(4, 4);
        *img.pixel_mut(2, 1) = Rgba8Pixel { r: 10, g: 20, b: 30, a: 40 };
        let p = img.pixel(2, 1);
        assert_eq!(p.r, 10);
        assert_eq!(p.g, 20);
        assert_eq!(p.b, 30);
        assert_eq!(p.a, 40);
        // Other pixels stay zero.
        assert_eq!(img.pixel(0, 0).r, 0);
    }

    #[test]
    fn test_pixel_write_read_rgbaf32() {
        let mut img = Image::<RgbaF32>::new(2, 2);
        *img.pixel_mut(1, 0) = RgbaF32Pixel { r: 0.5, g: 0.25, b: 0.125, a: 1.0 };
        let p = img.pixel(1, 0);
        assert_eq!(p.r, 0.5);
        assert_eq!(p.a, 1.0);
    }

    #[test]
    fn test_pixel_view_access() {
        let mut img = Image::<Rgba8>::new(4, 4);
        *img.pixel_mut(1, 2) = Rgba8Pixel { r: 99, g: 0, b: 0, a: 255 };
        let view = img.view();
        assert_eq!(view.pixel(1, 2).r, 99);
    }

    #[test]
    fn test_pixel_view_mut_access() {
        let mut img = Image::<Rgba8>::new(4, 4);
        {
            let mut view = img.view_mut();
            *view.pixel_mut(3, 3) = Rgba8Pixel { r: 1, g: 2, b: 3, a: 4 };
            assert_eq!(view.pixel(3, 3).r, 1);
        }
        assert_eq!(img.pixel(3, 3).b, 3);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_pixel_oob() {
        let img = Image::<Rgba8>::new(4, 4);
        img.pixel(4, 0);
    }

    // -- macropixel access (PackedFormat) --

    #[test]
    fn test_macropixel_write_read_yuyv8() {
        let mut img = Image::<Yuyv8>::new(4, 2);
        *img.macropixel_mut(0, 0) = Yuyv8Macropixel { y0: 100, u: 128, y1: 110, v: 130 };
        let m = img.macropixel(0, 0);
        assert_eq!(m.y0, 100);
        assert_eq!(m.u, 128);
        assert_eq!(m.y1, 110);
        assert_eq!(m.v, 130);
    }

    #[test]
    fn test_macropixel_view() {
        let mut img = Image::<Yuyv8>::new(4, 2);
        *img.macropixel_mut(1, 1) = Yuyv8Macropixel { y0: 50, u: 60, y1: 70, v: 80 };
        let view = img.view();
        assert_eq!(view.macropixel(1, 1).y0, 50);
    }

    // -- Nv12 plane access --

    #[test]
    fn test_nv12_plane_dimensions() {
        let img = Image::<Nv12>::new(640, 480);
        let y = img.y_plane();
        let uv = img.uv_plane();
        assert_eq!(y.width(), 640);
        assert_eq!(y.height(), 480);
        assert_eq!(y.stride(), 640);
        assert_eq!(uv.width(), 320);
        assert_eq!(uv.height(), 240);
        assert_eq!(uv.stride(), 640);
    }

    #[test]
    fn test_nv12_plane_stride_equality() {
        let img = Image::<Nv12>::new_with_stride(640, 480, 768);
        assert_eq!(img.y_plane().stride(), img.uv_plane().stride());
        assert_eq!(img.y_plane().stride(), 768);
    }

    #[test]
    fn test_nv12_plane_data_sizes() {
        let img = Image::<Nv12>::new(640, 480);
        let y = img.y_plane();
        let uv = img.uv_plane();
        assert_eq!(y.as_bytes().len(), 640 * 480);
        assert_eq!(uv.as_bytes().len(), 640 * 240);
    }

    #[test]
    fn test_nv12_plane_mut_write() {
        let mut img = Image::<Nv12>::new(4, 4);
        {
            let mut y = img.y_plane_mut();
            *y.pixel_mut(0, 0) = Y8Pixel { y: 200 };
        }
        {
            let mut uv = img.uv_plane_mut();
            *uv.pixel_mut(0, 0) = Uv8Pixel { u: 128, v: 130 };
        }
        assert_eq!(img.y_plane().pixel(0, 0).y, 200);
        assert_eq!(img.uv_plane().pixel(0, 0).u, 128);
    }

    #[test]
    fn test_nv12_view_planes() {
        let img = Image::<Nv12>::new(8, 4);
        let view = img.view();
        let y = view.y_plane();
        let uv = view.uv_plane();
        assert_eq!(y.width(), 8);
        assert_eq!(y.height(), 4);
        assert_eq!(uv.width(), 4);
        assert_eq!(uv.height(), 2);
    }

    // -- sub_image --

    #[test]
    fn test_sub_image_dimensions() {
        let img = Image::<Rgba8>::new(8, 8);
        let view = img.view();
        let sub = view.sub_image(2, 3, 4, 3);
        assert_eq!(sub.width(), 4);
        assert_eq!(sub.height(), 3);
        assert_eq!(sub.stride(), 32); // same as parent (8*4)
    }

    #[test]
    fn test_sub_image_reads_correct_pixels() {
        let mut img = Image::<Rgba8>::new(8, 8);
        *img.pixel_mut(3, 4) = Rgba8Pixel { r: 77, g: 88, b: 99, a: 255 };
        let view = img.view();
        let sub = view.sub_image(2, 3, 4, 4);
        // Pixel (3,4) in parent = (1,1) in sub (offset by 2,3)
        assert_eq!(sub.pixel(1, 1).r, 77);
        assert_eq!(sub.pixel(1, 1).g, 88);
    }

    #[test]
    #[should_panic(expected = "sub_image: region")]
    fn test_sub_image_oob() {
        let img = Image::<Rgba8>::new(8, 8);
        let view = img.view();
        view.sub_image(6, 6, 4, 4);
    }

    // -- format conversions: channel reorder --

    #[test]
    fn test_rgba8_to_argb8_round_trip() {
        let mut img = Image::<Rgba8>::new(2, 2);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 10, g: 20, b: 30, a: 40 };
        *img.pixel_mut(1, 0) = Rgba8Pixel { r: 100, g: 150, b: 200, a: 255 };
        let argb: Image<Argb8> = img.convert();
        assert_eq!(argb.pixel(0, 0), &Argb8Pixel { a: 40, r: 10, g: 20, b: 30 });
        let back: Image<Rgba8> = argb.convert();
        assert_eq!(back.pixel(0, 0), img.pixel(0, 0));
        assert_eq!(back.pixel(1, 0), img.pixel(1, 0));
    }

    #[test]
    fn test_rgba8_argb8_bgra8_round_trip() {
        let mut img = Image::<Rgba8>::new(1, 1);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 11, g: 22, b: 33, a: 44 };
        let argb: Image<Argb8> = img.convert();
        let bgra: Image<Bgra8> = argb.convert();
        let back: Image<Rgba8> = bgra.convert();
        assert_eq!(back.pixel(0, 0), &Rgba8Pixel { r: 11, g: 22, b: 33, a: 44 });
    }

    #[test]
    fn test_rgba8_to_bgra8() {
        let mut img = Image::<Rgba8>::new(1, 1);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 10, g: 20, b: 30, a: 40 };
        let bgra: Image<Bgra8> = img.convert();
        assert_eq!(bgra.pixel(0, 0), &Bgra8Pixel { b: 30, g: 20, r: 10, a: 40 });
    }

    // -- sRGB ↔ linear --

    #[test]
    fn test_srgb_to_linear_known_values() {
        // sRGB normalized 0.5 → linear ≈ 0.2140
        let linear = srgb_to_linear(0.5);
        assert!((linear - 0.2140).abs() < 0.001, "got {}", linear);

        // sRGB u8=188 (normalized ≈ 0.737) → linear ≈ 0.502
        let linear2 = srgb_to_linear(188.0 / 255.0);
        assert!((linear2 - 0.502).abs() < 0.01, "got {}", linear2);
    }

    #[test]
    fn test_linear_to_srgb_known_values() {
        let srgb = linear_to_srgb(0.2140);
        assert!((srgb - 0.5).abs() < 0.001, "got {}", srgb);
    }

    #[test]
    fn test_srgba8_to_rgbaf32_and_back() {
        let mut img = Image::<Srgba8>::new(1, 1);
        *img.pixel_mut(0, 0) = Srgba8Pixel { r: 128, g: 64, b: 192, a: 255 };
        let linear: Image<RgbaF32> = img.convert();
        let back: Image<Srgba8> = linear.convert();
        // Round-trip should be within ±1 u8 step
        let orig = img.pixel(0, 0);
        let result = back.pixel(0, 0);
        assert!((orig.r as i16 - result.r as i16).abs() <= 1);
        assert!((orig.g as i16 - result.g as i16).abs() <= 1);
        assert!((orig.b as i16 - result.b as i16).abs() <= 1);
        assert_eq!(orig.a, result.a);
    }

    // -- precision: Rgba8 ↔ RgbaF32 --

    #[test]
    fn test_rgba8_to_rgbaf32_round_trip() {
        let mut img = Image::<Rgba8>::new(1, 1);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 128, g: 64, b: 32, a: 255 };
        let f32img: Image<RgbaF32> = img.convert();
        let p = f32img.pixel(0, 0);
        assert!((p.r - 128.0 / 255.0).abs() < 0.001);
        assert!((p.g - 64.0 / 255.0).abs() < 0.001);
        let back: Image<Rgba8> = f32img.convert();
        assert_eq!(back.pixel(0, 0), &Rgba8Pixel { r: 128, g: 64, b: 32, a: 255 });
    }

    // -- precision: RgbaF16 ↔ RgbaF32 --

    #[test]
    fn test_rgbaf16_to_rgbaf32_round_trip() {
        let mut img = Image::<RgbaF16>::new(1, 1);
        *img.pixel_mut(0, 0) = RgbaF16Pixel {
            r: F16::from_f32(0.5),
            g: F16::from_f32(0.25),
            b: F16::from_f32(0.125),
            a: F16::from_f32(1.0),
        };
        let f32img: Image<RgbaF32> = img.convert();
        let p = f32img.pixel(0, 0);
        assert!((p.r - 0.5).abs() < 0.001);
        assert!((p.g - 0.25).abs() < 0.001);
        let back: Image<RgbaF16> = f32img.convert();
        let bp = back.pixel(0, 0);
        assert!((bp.r.to_f32() - 0.5).abs() < 0.001);
    }

    // -- video: YUYV → Rgba8 --

    #[test]
    fn test_yuyv8_to_rgba8() {
        let mut img = Image::<Yuyv8>::new(2, 1);
        // Pure white in YUV: Y=235, U=128, V=128 → R≈235, G≈235, B≈235
        *img.macropixel_mut(0, 0) = Yuyv8Macropixel { y0: 235, u: 128, y1: 235, v: 128 };
        let rgba: Image<Rgba8> = img.convert();
        let p0 = rgba.pixel(0, 0);
        let p1 = rgba.pixel(1, 0);
        // With U=128, V=128 (neutral chroma), R≈G≈B≈Y
        assert!((p0.r as i16 - 235).abs() <= 1);
        assert!((p0.g as i16 - 235).abs() <= 1);
        assert!((p0.b as i16 - 235).abs() <= 1);
        assert_eq!(p0.a, 255);
        assert_eq!(p1.r, p0.r); // same Y and same chroma
    }

    #[test]
    fn test_yuyv8_to_rgba8_colored() {
        let mut img = Image::<Yuyv8>::new(2, 1);
        // Known BT.601 test: Y=180, U=100, V=200
        *img.macropixel_mut(0, 0) = Yuyv8Macropixel { y0: 180, u: 100, y1: 180, v: 200 };
        let rgba: Image<Rgba8> = img.convert();
        let p = rgba.pixel(0, 0);
        // R = 180 + 1.402*(200-128) = 180 + 100.944 = 280.944 → clamped 255
        // G = 180 - 0.344136*(100-128) - 0.714136*(200-128) = 180 + 9.636 - 51.418 = 138.218
        // B = 180 + 1.772*(100-128) = 180 - 49.616 = 130.384
        assert_eq!(p.r, 255); // clamped
        assert!((p.g as i16 - 138).abs() <= 1);
        assert!((p.b as i16 - 130).abs() <= 1);
    }

    // -- video: NV12 → Rgba8 --

    #[test]
    fn test_nv12_to_rgba8() {
        let mut img = Image::<Nv12>::new(4, 2);
        // Fill Y plane with 200
        {
            let mut y = img.y_plane_mut();
            for py in 0..2u32 {
                for px in 0..4u32 {
                    *y.pixel_mut(px, py) = Y8Pixel { y: 200 };
                }
            }
        }
        // Fill UV plane with neutral chroma (128, 128)
        {
            let mut uv = img.uv_plane_mut();
            for py in 0..1u32 {
                for px in 0..2u32 {
                    *uv.pixel_mut(px, py) = Uv8Pixel { u: 128, v: 128 };
                }
            }
        }
        let rgba: Image<Rgba8> = img.convert();
        // Neutral chroma: R≈G≈B≈Y=200
        for py in 0..2u32 {
            for px in 0..4u32 {
                let p = rgba.pixel(px, py);
                assert!((p.r as i16 - 200).abs() <= 1, "pixel ({},{}) r={}", px, py, p.r);
                assert!((p.g as i16 - 200).abs() <= 1);
                assert!((p.b as i16 - 200).abs() <= 1);
                assert_eq!(p.a, 255);
            }
        }
    }

    // -- tensor interop --

    #[test]
    fn test_rgba8_to_tensor_shape() {
        let img = Image::<Rgba8>::new(4, 3);
        let t = img.to_tensor_nchw();
        assert_eq!(t.shape(), &[1, 4, 3, 4]); // [1, C=4, H=3, W=4]
    }

    #[test]
    fn test_rgba8_to_tensor_values() {
        let mut img = Image::<Rgba8>::new(2, 1);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 255, g: 0, b: 128, a: 255 };
        let t = img.to_tensor_nchw();
        // NCHW: channel 0 (R), row 0, col 0
        assert!((t.get(&[0, 0, 0, 0]) - 1.0).abs() < 0.001); // R=255 → 1.0
        assert!((t.get(&[0, 1, 0, 0]) - 0.0).abs() < 0.001); // G=0 → 0.0
        assert!((t.get(&[0, 2, 0, 0]) - 128.0 / 255.0).abs() < 0.01); // B=128
        assert!((t.get(&[0, 3, 0, 0]) - 1.0).abs() < 0.001); // A=255 → 1.0
    }

    #[test]
    fn test_rgba8_tensor_round_trip() {
        let mut img = Image::<Rgba8>::new(3, 2);
        *img.pixel_mut(0, 0) = Rgba8Pixel { r: 128, g: 64, b: 32, a: 255 };
        *img.pixel_mut(2, 1) = Rgba8Pixel { r: 10, g: 20, b: 30, a: 40 };
        let t = img.to_tensor_nchw();
        let back = Image::<Rgba8>::from_tensor_nchw(&t);
        assert_eq!(back.width(), 3);
        assert_eq!(back.height(), 2);
        // u8 round-trip error at most 1 step
        let eps = 1;
        let p = back.pixel(0, 0);
        assert!((p.r as i16 - 128).abs() <= eps);
        assert!((p.g as i16 - 64).abs() <= eps);
        assert!((p.b as i16 - 32).abs() <= eps);
        assert!((p.a as i16 - 255).abs() <= eps);
        let p2 = back.pixel(2, 1);
        assert!((p2.r as i16 - 10).abs() <= eps);
        assert!((p2.g as i16 - 20).abs() <= eps);
    }

    #[test]
    fn test_rgbaf32_tensor_round_trip_exact() {
        let mut img = Image::<RgbaF32>::new(2, 2);
        *img.pixel_mut(0, 0) = RgbaF32Pixel { r: 0.1, g: 0.2, b: 0.3, a: 0.4 };
        *img.pixel_mut(1, 1) = RgbaF32Pixel { r: 0.9, g: 0.8, b: 0.7, a: 0.6 };
        let t = img.to_tensor_nchw();
        let back = Image::<RgbaF32>::from_tensor_nchw(&t);
        // f32 round-trip should be exact
        assert_eq!(back.pixel(0, 0), &RgbaF32Pixel { r: 0.1, g: 0.2, b: 0.3, a: 0.4 });
        assert_eq!(back.pixel(1, 1), &RgbaF32Pixel { r: 0.9, g: 0.8, b: 0.7, a: 0.6 });
    }

    #[test]
    fn test_y8_to_tensor() {
        let mut img = Image::<Y8>::new(2, 2);
        *img.pixel_mut(0, 0) = Y8Pixel { y: 128 };
        let t = img.to_tensor_nchw();
        assert_eq!(t.shape(), &[1, 1, 2, 2]); // [1, C=1, H=2, W=2]
        assert!((t.get(&[0, 0, 0, 0]) - 128.0 / 255.0).abs() < 0.01);
    }

    #[test]
    fn test_from_tensor_3d_shape() {
        // Accept [C, H, W] without batch dimension
        let t = Tensor::from_shape_data(&[4, 2, 3], vec![0.5f32; 24]);
        let img = Image::<Rgba8>::from_tensor_nchw(&t);
        assert_eq!(img.width(), 3);
        assert_eq!(img.height(), 2);
    }

    #[test]
    #[should_panic(expected = "channel mismatch")]
    fn test_from_tensor_channel_mismatch() {
        let t = Tensor::from_shape_data(&[1, 3, 2, 2], vec![0.0f32; 12]);
        Image::<Rgba8>::from_tensor_nchw(&t);
    }
}
