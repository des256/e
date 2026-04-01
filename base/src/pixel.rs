use crate::F16;

// -- GPU format enum --

/// Abstract GPU texture format, not tied to any specific API.
///
/// Maps to equivalent formats in wgpu, Vulkan, Metal, etc.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GpuFormat {
    /// 8-bit RGBA, unsigned normalized linear.
    Rgba8Unorm,
    /// 8-bit RGBA, sRGB encoded.
    Rgba8Srgb,
    /// 8-bit BGRA, unsigned normalized linear.
    Bgra8Unorm,
    /// 8-bit BGRA, sRGB encoded.
    Bgra8Srgb,
    /// 16-bit RGBA, half-float.
    Rgba16Float,
    /// 32-bit RGBA, full-float.
    Rgba32Float,
    /// 8-bit single-channel, unsigned normalized.
    R8Unorm,
    /// 8-bit two-channel, unsigned normalized.
    Rg8Unorm,
}

// -- trait hierarchy --

/// Base trait for all pixel formats.
///
/// Every pixel format (linear, packed, planar, compressed) implements this.
/// Format types are zero-sized — all information is in associated constants.
pub trait PixelFormat: 'static {
    /// Human-readable format name.
    const NAME: &'static str;
    /// Corresponding GPU texture format, if one exists.
    const GPU_FORMAT: Option<GpuFormat>;
}

/// Uncompressed pixel format with deterministic size.
///
/// Provides stride and buffer size computation. Format types are ZSTs,
/// so these are associated functions, not methods.
pub trait UncompressedFormat: PixelFormat {
    /// Minimum byte stride for a row of `width` pixels.
    fn min_stride(width: u32) -> u32;
    /// Total buffer size in bytes for the given dimensions and stride.
    fn data_size(width: u32, height: u32, stride: u32) -> usize;
}

/// Standard row-major pixel format with direct per-pixel access.
///
/// Each pixel occupies a fixed number of contiguous bytes.
pub trait LinearFormat: UncompressedFormat {
    /// The typed pixel representation. Must be `#[repr(C)]` with
    /// `size_of::<Pixel>() == BYTES_PER_PIXEL`.
    type Pixel: Copy + 'static;
    /// Bytes per pixel.
    const BYTES_PER_PIXEL: usize;
}

/// Macropixel format where multiple pixels share encoded data.
///
/// For example, YUYV packs two horizontally adjacent pixels into a
/// single 4-byte macropixel.
pub trait PackedFormat: UncompressedFormat {
    /// The typed macropixel representation (`#[repr(C)]`).
    type Macropixel: Copy + 'static;
    /// Bytes per macropixel.
    const BYTES_PER_MACROPIXEL: usize;
    /// Number of pixels encoded in one macropixel.
    const PIXELS_PER_MACROPIXEL: usize;
}

/// Planar pixel format with separate data planes.
///
/// Plane accessors are inherent methods on `Image<ConcreteFormat>`,
/// not on this trait, because plane structure is format-specific.
pub trait PlanarFormat: UncompressedFormat {}

// ============================================================
// Linear format types
// ============================================================

// -- Rgba8 --

/// 8-bit RGBA pixel (memory order: R, G, B, A). Unspecified color space.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Rgba8Pixel {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
    /// Alpha channel.
    pub a: u8,
}

/// 8-bit RGBA format, unspecified color space.
pub struct Rgba8;

impl PixelFormat for Rgba8 {
    const NAME: &'static str = "Rgba8";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Rgba8Unorm);
}

impl UncompressedFormat for Rgba8 {
    fn min_stride(width: u32) -> u32 { width * 4 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Rgba8 {
    type Pixel = Rgba8Pixel;
    const BYTES_PER_PIXEL: usize = 4;
}

// -- Argb8 --

/// 8-bit ARGB pixel (memory order: A, R, G, B).
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Argb8Pixel {
    /// Alpha channel.
    pub a: u8,
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

/// 8-bit ARGB format. No standard GPU equivalent.
pub struct Argb8;

impl PixelFormat for Argb8 {
    const NAME: &'static str = "Argb8";
    const GPU_FORMAT: Option<GpuFormat> = None;
}

impl UncompressedFormat for Argb8 {
    fn min_stride(width: u32) -> u32 { width * 4 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Argb8 {
    type Pixel = Argb8Pixel;
    const BYTES_PER_PIXEL: usize = 4;
}

// -- Bgra8 --

/// 8-bit BGRA pixel (memory order: B, G, R, A).
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Bgra8Pixel {
    /// Blue channel.
    pub b: u8,
    /// Green channel.
    pub g: u8,
    /// Red channel.
    pub r: u8,
    /// Alpha channel.
    pub a: u8,
}

/// 8-bit BGRA format. Vulkan swapchain / Metal native on many GPUs.
pub struct Bgra8;

impl PixelFormat for Bgra8 {
    const NAME: &'static str = "Bgra8";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Bgra8Unorm);
}

impl UncompressedFormat for Bgra8 {
    fn min_stride(width: u32) -> u32 { width * 4 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Bgra8 {
    type Pixel = Bgra8Pixel;
    const BYTES_PER_PIXEL: usize = 4;
}

// -- Srgba8 --

/// 8-bit sRGB-encoded RGBA pixel (memory order: R, G, B, A).
///
/// Distinct type from [`Rgba8Pixel`] to enforce color space safety.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Srgba8Pixel {
    /// Red channel (sRGB encoded).
    pub r: u8,
    /// Green channel (sRGB encoded).
    pub g: u8,
    /// Blue channel (sRGB encoded).
    pub b: u8,
    /// Alpha channel (linear).
    pub a: u8,
}

/// 8-bit sRGB RGBA format. Perceptually uniform encoding for display.
pub struct Srgba8;

impl PixelFormat for Srgba8 {
    const NAME: &'static str = "Srgba8";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Rgba8Srgb);
}

impl UncompressedFormat for Srgba8 {
    fn min_stride(width: u32) -> u32 { width * 4 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Srgba8 {
    type Pixel = Srgba8Pixel;
    const BYTES_PER_PIXEL: usize = 4;
}

// -- RgbaF16 --

/// Half-float RGBA pixel using [`F16`].
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RgbaF16Pixel {
    /// Red channel.
    pub r: F16,
    /// Green channel.
    pub g: F16,
    /// Blue channel.
    pub b: F16,
    /// Alpha channel.
    pub a: F16,
}

/// Half-float RGBA format. HDR textures, GPU compute.
pub struct RgbaF16;

impl PixelFormat for RgbaF16 {
    const NAME: &'static str = "RgbaF16";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Rgba16Float);
}

impl UncompressedFormat for RgbaF16 {
    fn min_stride(width: u32) -> u32 { width * 8 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for RgbaF16 {
    type Pixel = RgbaF16Pixel;
    const BYTES_PER_PIXEL: usize = 8;
}

// -- RgbaF32 --

/// Full-float RGBA pixel.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct RgbaF32Pixel {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

/// Full-float RGBA format. HDR processing, render targets, ML pipelines.
pub struct RgbaF32;

impl PixelFormat for RgbaF32 {
    const NAME: &'static str = "RgbaF32";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Rgba32Float);
}

impl UncompressedFormat for RgbaF32 {
    fn min_stride(width: u32) -> u32 { width * 16 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for RgbaF32 {
    type Pixel = RgbaF32Pixel;
    const BYTES_PER_PIXEL: usize = 16;
}

// ============================================================
// Helper format types (for NV12 plane views)
// ============================================================

// -- Y8 --

/// Single-channel 8-bit luminance pixel.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Y8Pixel {
    /// Luminance value.
    pub y: u8,
}

/// Single-channel 8-bit luminance format.
pub struct Y8;

impl PixelFormat for Y8 {
    const NAME: &'static str = "Y8";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::R8Unorm);
}

impl UncompressedFormat for Y8 {
    fn min_stride(width: u32) -> u32 { width }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Y8 {
    type Pixel = Y8Pixel;
    const BYTES_PER_PIXEL: usize = 1;
}

// -- Uv8 --

/// Two-channel interleaved 8-bit chroma pixel.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Uv8Pixel {
    /// Blue-difference chroma (Cb).
    pub u: u8,
    /// Red-difference chroma (Cr).
    pub v: u8,
}

/// Two-channel interleaved 8-bit chroma format.
pub struct Uv8;

impl PixelFormat for Uv8 {
    const NAME: &'static str = "Uv8";
    const GPU_FORMAT: Option<GpuFormat> = Some(GpuFormat::Rg8Unorm);
}

impl UncompressedFormat for Uv8 {
    fn min_stride(width: u32) -> u32 { width * 2 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl LinearFormat for Uv8 {
    type Pixel = Uv8Pixel;
    const BYTES_PER_PIXEL: usize = 2;
}

// ============================================================
// Packed format types
// ============================================================

// -- Yuyv8 --

/// YUYV macropixel: two horizontally adjacent pixels packed into 4 bytes.
///
/// `y0` and `y1` are luminance for the left and right pixel;
/// `u` and `v` are shared chroma.
#[derive(Copy, Clone, Debug, PartialEq)]
#[repr(C)]
pub struct Yuyv8Macropixel {
    /// Luminance of the left pixel.
    pub y0: u8,
    /// Shared blue-difference chroma (Cb).
    pub u: u8,
    /// Luminance of the right pixel.
    pub y1: u8,
    /// Shared red-difference chroma (Cr).
    pub v: u8,
}

/// YUYV 4:2:2 packed format (V4L2 webcam output).
///
/// Width must be even. Each macropixel encodes 2 pixels in 4 bytes.
pub struct Yuyv8;

impl PixelFormat for Yuyv8 {
    const NAME: &'static str = "Yuyv8";
    const GPU_FORMAT: Option<GpuFormat> = None;
}

impl UncompressedFormat for Yuyv8 {
    fn min_stride(width: u32) -> u32 { width * 2 }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize
    }
}

impl PackedFormat for Yuyv8 {
    type Macropixel = Yuyv8Macropixel;
    const BYTES_PER_MACROPIXEL: usize = 4;
    const PIXELS_PER_MACROPIXEL: usize = 2;
}

// ============================================================
// Planar format types
// ============================================================

// -- Nv12 --

/// NV12 YUV 4:2:0 semi-planar format.
///
/// Y plane (width×height, stride per row) followed by interleaved UV plane
/// (width/2 × height/2 samples, each 2 bytes; UV row = width bytes = same
/// stride as Y). Total buffer = stride × height × 1.5.
///
/// Width and height must be even.
pub struct Nv12;

impl PixelFormat for Nv12 {
    const NAME: &'static str = "Nv12";
    const GPU_FORMAT: Option<GpuFormat> = None;
}

impl UncompressedFormat for Nv12 {
    fn min_stride(width: u32) -> u32 { width }
    fn data_size(_width: u32, height: u32, stride: u32) -> usize {
        stride as usize * height as usize + stride as usize * (height as usize / 2)
    }
}

impl PlanarFormat for Nv12 {}

// ============================================================
// Compile-time size assertions
// ============================================================

const _: () = assert!(std::mem::size_of::<Rgba8Pixel>() == 4);
const _: () = assert!(std::mem::size_of::<Argb8Pixel>() == 4);
const _: () = assert!(std::mem::size_of::<Bgra8Pixel>() == 4);
const _: () = assert!(std::mem::size_of::<Srgba8Pixel>() == 4);
const _: () = assert!(std::mem::size_of::<RgbaF16Pixel>() == 8);
const _: () = assert!(std::mem::size_of::<RgbaF32Pixel>() == 16);
const _: () = assert!(std::mem::size_of::<Y8Pixel>() == 1);
const _: () = assert!(std::mem::size_of::<Uv8Pixel>() == 2);
const _: () = assert!(std::mem::size_of::<Yuyv8Macropixel>() == 4);

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- trait hierarchy (dummy types) --

    #[test]
    fn test_gpu_format_variants() {
        assert_eq!(GpuFormat::Rgba8Unorm, GpuFormat::Rgba8Unorm);
        assert_ne!(GpuFormat::Rgba8Unorm, GpuFormat::Rgba8Srgb);
        let _all = [
            GpuFormat::Rgba8Unorm, GpuFormat::Rgba8Srgb,
            GpuFormat::Bgra8Unorm, GpuFormat::Bgra8Srgb,
            GpuFormat::Rgba16Float, GpuFormat::Rgba32Float,
            GpuFormat::R8Unorm, GpuFormat::Rg8Unorm,
        ];
    }

    // -- linear format pixel struct construction --

    #[test]
    fn test_rgba8_pixel() {
        let p = Rgba8Pixel { r: 255, g: 128, b: 64, a: 32 };
        assert_eq!(p.r, 255);
        assert_eq!(p.g, 128);
        assert_eq!(p.b, 64);
        assert_eq!(p.a, 32);
        assert_eq!(std::mem::size_of::<Rgba8Pixel>(), 4);
    }

    #[test]
    fn test_argb8_pixel() {
        let p = Argb8Pixel { a: 255, r: 128, g: 64, b: 32 };
        assert_eq!(p.a, 255);
        assert_eq!(p.r, 128);
        assert_eq!(std::mem::size_of::<Argb8Pixel>(), 4);
    }

    #[test]
    fn test_bgra8_pixel() {
        let p = Bgra8Pixel { b: 10, g: 20, r: 30, a: 40 };
        assert_eq!(p.b, 10);
        assert_eq!(p.r, 30);
        assert_eq!(std::mem::size_of::<Bgra8Pixel>(), 4);
    }

    #[test]
    fn test_srgba8_pixel() {
        let p = Srgba8Pixel { r: 1, g: 2, b: 3, a: 4 };
        assert_eq!(p.r, 1);
        assert_eq!(std::mem::size_of::<Srgba8Pixel>(), 4);
    }

    #[test]
    fn test_rgba_f16_pixel() {
        let p = RgbaF16Pixel {
            r: F16::from_f32(1.0),
            g: F16::from_f32(0.5),
            b: F16::from_f32(0.25),
            a: F16::from_f32(1.0),
        };
        assert!((p.r.to_f32() - 1.0).abs() < 0.001);
        assert!((p.g.to_f32() - 0.5).abs() < 0.001);
        assert_eq!(std::mem::size_of::<RgbaF16Pixel>(), 8);
    }

    #[test]
    fn test_rgba_f32_pixel() {
        let p = RgbaF32Pixel { r: 1.0, g: 0.5, b: 0.25, a: 1.0 };
        assert_eq!(p.r, 1.0);
        assert_eq!(p.g, 0.5);
        assert_eq!(std::mem::size_of::<RgbaF32Pixel>(), 16);
    }

    // -- GPU format mapping --

    #[test]
    fn test_gpu_format_mapping() {
        assert_eq!(Rgba8::GPU_FORMAT, Some(GpuFormat::Rgba8Unorm));
        assert_eq!(Argb8::GPU_FORMAT, None);
        assert_eq!(Bgra8::GPU_FORMAT, Some(GpuFormat::Bgra8Unorm));
        assert_eq!(Srgba8::GPU_FORMAT, Some(GpuFormat::Rgba8Srgb));
        assert_eq!(RgbaF16::GPU_FORMAT, Some(GpuFormat::Rgba16Float));
        assert_eq!(RgbaF32::GPU_FORMAT, Some(GpuFormat::Rgba32Float));
        assert_eq!(Y8::GPU_FORMAT, Some(GpuFormat::R8Unorm));
        assert_eq!(Uv8::GPU_FORMAT, Some(GpuFormat::Rg8Unorm));
        assert_eq!(Yuyv8::GPU_FORMAT, None);
        assert_eq!(Nv12::GPU_FORMAT, None);
    }

    // -- linear format trait constants --

    #[test]
    fn test_linear_format_bytes_per_pixel() {
        assert_eq!(Rgba8::BYTES_PER_PIXEL, 4);
        assert_eq!(Argb8::BYTES_PER_PIXEL, 4);
        assert_eq!(Bgra8::BYTES_PER_PIXEL, 4);
        assert_eq!(Srgba8::BYTES_PER_PIXEL, 4);
        assert_eq!(RgbaF16::BYTES_PER_PIXEL, 8);
        assert_eq!(RgbaF32::BYTES_PER_PIXEL, 16);
        assert_eq!(Y8::BYTES_PER_PIXEL, 1);
        assert_eq!(Uv8::BYTES_PER_PIXEL, 2);
    }

    #[test]
    fn test_linear_format_min_stride() {
        assert_eq!(Rgba8::min_stride(640), 2560);
        assert_eq!(RgbaF32::min_stride(640), 10240);
        assert_eq!(Y8::min_stride(640), 640);
        assert_eq!(Uv8::min_stride(320), 640);
    }

    // -- packed format --

    #[test]
    fn test_yuyv8_macropixel() {
        let m = Yuyv8Macropixel { y0: 100, u: 128, y1: 120, v: 130 };
        assert_eq!(m.y0, 100);
        assert_eq!(m.u, 128);
        assert_eq!(m.y1, 120);
        assert_eq!(m.v, 130);
        assert_eq!(std::mem::size_of::<Yuyv8Macropixel>(), 4);
    }

    #[test]
    fn test_yuyv8_constants() {
        assert_eq!(Yuyv8::BYTES_PER_MACROPIXEL, 4);
        assert_eq!(Yuyv8::PIXELS_PER_MACROPIXEL, 2);
        assert_eq!(Yuyv8::min_stride(640), 1280);
    }

    // -- planar format --

    #[test]
    fn test_nv12_data_size() {
        // 640x480: Y=640*480=307200, UV=640*240=153600, total=460800
        assert_eq!(Nv12::data_size(640, 480, 640), 460800);
        assert_eq!(Nv12::min_stride(640), 640);
    }

    #[test]
    fn test_nv12_data_size_with_padding() {
        // stride=768 (256-byte aligned), 640x480
        // Y=768*480=368640, UV=768*240=184320, total=552960
        assert_eq!(Nv12::data_size(640, 480, 768), 552960);
    }

    // -- format names --

    #[test]
    fn test_format_names() {
        assert_eq!(Rgba8::NAME, "Rgba8");
        assert_eq!(Argb8::NAME, "Argb8");
        assert_eq!(Bgra8::NAME, "Bgra8");
        assert_eq!(Srgba8::NAME, "Srgba8");
        assert_eq!(RgbaF16::NAME, "RgbaF16");
        assert_eq!(RgbaF32::NAME, "RgbaF32");
        assert_eq!(Y8::NAME, "Y8");
        assert_eq!(Uv8::NAME, "Uv8");
        assert_eq!(Yuyv8::NAME, "Yuyv8");
        assert_eq!(Nv12::NAME, "Nv12");
    }

    // -- repr(C) byte layout verification --

    #[test]
    fn test_rgba8_pixel_byte_layout() {
        let p = Rgba8Pixel { r: 0xAA, g: 0xBB, b: 0xCC, a: 0xDD };
        let bytes: &[u8; 4] = unsafe { &*(&p as *const Rgba8Pixel as *const [u8; 4]) };
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_argb8_pixel_byte_layout() {
        let p = Argb8Pixel { a: 0xAA, r: 0xBB, g: 0xCC, b: 0xDD };
        let bytes: &[u8; 4] = unsafe { &*(&p as *const Argb8Pixel as *const [u8; 4]) };
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_bgra8_pixel_byte_layout() {
        let p = Bgra8Pixel { b: 0xAA, g: 0xBB, r: 0xCC, a: 0xDD };
        let bytes: &[u8; 4] = unsafe { &*(&p as *const Bgra8Pixel as *const [u8; 4]) };
        assert_eq!(bytes, &[0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_yuyv8_macropixel_byte_layout() {
        let m = Yuyv8Macropixel { y0: 0x11, u: 0x22, y1: 0x33, v: 0x44 };
        let bytes: &[u8; 4] = unsafe { &*(&m as *const Yuyv8Macropixel as *const [u8; 4]) };
        assert_eq!(bytes, &[0x11, 0x22, 0x33, 0x44]);
    }
}
