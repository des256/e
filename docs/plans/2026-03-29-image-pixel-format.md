# Image\<P\> Pixel Format Infrastructure Plan

Created: 2026-03-29
Status: COMPLETE
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Implement a type-parameterized `Image<P>` system in the base crate where each pixel format is a distinct Rust type, with a trait hierarchy covering linear, packed, planar, and compressed format categories, plus `ImageView`/`ImageViewMut` for zero-copy borrows. Include format conversions (channel reorder, sRGB↔linear, YUYV→RGB, NV12→RGB), an abstract GPU format enum, and tensor interop.

**Architecture:** Two new files in `base/src/`: `pixel.rs` (trait hierarchy, GPU format enum, all format zero-sized types, typed pixel/macropixel structs) and `image.rs` (Image/ImageView/ImageViewMut containers, pixel access, conversions, tensor interop). Follows the existing flat-file-per-concept pattern.

**Tech Stack:** Pure Rust, no external dependencies. Uses existing `F16` and `Tensor<T>` types from the base crate.

## Scope

### In Scope

- PixelFormat trait hierarchy: PixelFormat → UncompressedFormat → {LinearFormat, PackedFormat, PlanarFormat}
- 8 pixel formats: Rgba8, Argb8, Bgra8, Srgba8, RgbaF16, RgbaF32, Yuyv8, Nv12
- 2 helper formats for NV12 plane views: Y8, Uv8
- Typed pixel structs per format (separate types even when memory layout matches, e.g. Rgba8Pixel vs Srgba8Pixel)
- Image\<P\>, ImageView\<'a, P\>, ImageViewMut\<'a, P\> with stride support
- Pixel access: pixel()/pixel_mut() for linear, macropixel()/macropixel_mut() for packed, y_plane()/uv_plane() for Nv12
- Format conversions: channel reorder (Rgba8↔Argb8↔Bgra8), sRGB↔linear (Srgba8↔RgbaF32), precision (Rgba8↔RgbaF32, RgbaF16↔RgbaF32), video (Yuyv8→Rgba8, Nv12→Rgba8)
- Abstract GpuFormat enum (not tied to wgpu/vulkan)
- Tensor interop: to_tensor_nchw() / from_tensor_nchw() for linear formats
- sub_image() for zero-copy cropping

### Out of Scope

- CompressedFormat trait and Jpeg/Png types (deferred to follow-up)
- LinearRgba8 format (dropped — linear-light work uses RgbaF32/RgbaF16)
- Additional formats beyond the 8+2 listed (extensible later)
- wgpu/Vulkan-specific integration (GpuFormat is abstract)
- SIMD-optimized conversion paths
- Image I/O (file loading/saving)

## Approach

**Chosen:** Type-level pixel formats (Option A) with category-specific trait methods

**Why:** Compile-time safety prevents mixing incompatible formats. Monomorphization eliminates runtime dispatch overhead. The cost is that "image of unknown format" requires an enum or trait object — acceptable for this codebase.

**Alternatives considered:**
- *Runtime format field:* `Image<u8>` with a format enum — flexible but loses compile-time safety, runtime errors for format mismatches.
- *Channel-type parameterization:* `Image<u8>`, `Image<f32>` — doesn't capture channel layout or color space, insufficient for YUYV/NV12.

## Context for Implementer

> Write for an implementer who has never seen the codebase.

- **Patterns to follow:** All types in `base/src/` use inline `#[cfg(test)] mod tests` (see `tensor.rs:997`, `f16.rs`). Structs use `#[derive(Copy, Clone, Debug, PartialEq)]`. Imports use `crate::*` style (see `vec2.rs:3`). No `#[repr(C)]` on existing types, but pixel structs need it for safe byte↔struct transmutation.
- **Conventions:** Free-standing constructor functions alongside types (e.g. `pub const fn vec4<T>(...) -> Vec4<T>` in `vec4.rs:41`). Module files re-exported via `pub use` in `lib.rs`.
- **Key files:**
  - `base/src/lib.rs` — module declarations and re-exports
  - `base/src/tensor.rs` — `Tensor<T>`, `TensorView<'a, T>`, `TensorElement` trait — similar owned/view pattern to follow
  - `base/src/f16.rs` — `F16` type used by `RgbaF16` pixel format, `#[repr(transparent)]` wrapping u16
  - `base/src/zero.rs` — `Zero` trait pattern for trait design reference
- **Gotchas:**
  - `F16` arithmetic promotes to f32 and rounds back (see `f16.rs`). Use `F16::from_f32()` and `.to_f32()` for conversions.
  - No external deps — all math (sRGB gamma, YUV↔RGB matrix) must be inline.
  - `Tensor<T>` uses element strides (not byte strides). Image uses byte strides. Don't confuse them during tensor interop.
  - `TensorElement` is the bound for `Tensor<T>` — only `f32`, `f64`, `F16`, `i8`, `i32`, `i64`, `u8` are supported.
  - **Alignment:** `Vec<u8>` is only 1-byte aligned. Pixel types with u8 fields (align=1) are always safe to reference via pointer cast. Float pixel types (RgbaF32Pixel has align=4, RgbaF16Pixel has align=2) require aligned allocations. The `new*` constructors should use `Vec::with_capacity` + zeroing which typically gives higher alignment, but this is not guaranteed. Add runtime alignment assertions in pixel access for safety.
- **Domain context:**
  - **sRGB transfer function:** Linear→sRGB: `if L ≤ 0.0031308 then 12.92*L else 1.055*L^(1/2.4) - 0.055`. Inverse for sRGB→linear.
  - **YUV→RGB (BT.601):** `R = Y + 1.402*(V-128)`, `G = Y - 0.344136*(U-128) - 0.714136*(V-128)`, `B = Y + 1.772*(U-128)`. Clamp to [0,255].
  - **NV12 layout:** Y plane (width×height bytes, stride per row), followed by UV plane (interleaved U,V pairs, width/2 × height/2 samples, each sample 2 bytes, so UV row = width bytes = same stride as Y).
  - **YUYV layout:** Each 4-byte macropixel encodes 2 horizontally adjacent pixels: [Y0, U, Y1, V]. Width must be even.
  - **Stride:** Bytes per row, always ≥ minimum for the format. GPU textures commonly require 256-byte alignment. The stride field accommodates this padding.

## Assumptions

- `F16::from_f32()` and `F16::to_f32()` are correct and performant for bulk pixel conversion — supported by `f16.rs` implementation. Tasks 2, 6 depend on this.
- NV12 UV plane stride equals Y plane stride (standard for NV12). Task 3, 5 depend on this.
- `Tensor::from_shape_data` and `Tensor::as_slice()` work correctly for interop — supported by `tensor.rs` tests. Task 7 depends on this.
- Pixel structs with `#[repr(C)]` can be safely transmuted from aligned byte slices — standard Rust guarantee for `#[repr(C)]` types with no padding. Tasks 2, 3, 5 depend on this.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Pixel struct alignment/padding surprises on exotic platforms | Low | High | Use `#[repr(C)]` and add `const_assert!(size_of::<Rgba8Pixel>() == 4)` compile-time checks for every pixel struct |
| sRGB gamma math precision | Low | Medium | Test against known reference values (e.g. sRGB(0.5) ≈ 0.735) with epsilon tolerance |
| NV12 stride assumption wrong for some hardware | Medium | Medium | Document that stride applies to both planes; validate in constructor that data size matches expected layout |
| Tensor interop shape mismatch | Low | Medium | Assert tensor shape [1,C,H,W] or [C,H,W] in from_tensor_nchw, test round-trip |

## Goal Verification

### Truths

1. All 10 format types (Rgba8, Argb8, Bgra8, Srgba8, RgbaF16, RgbaF32, Yuyv8, Nv12, Y8, Uv8) compile and implement their respective trait hierarchy
2. `Image<Rgba8>` can be created, pixels read/written by (x,y), and bytes accessed
3. `ImageView` provides zero-copy borrowing and sub_image() returns a cropped view without allocation
4. Channel reorder conversions produce correct byte output (e.g. Rgba8 [R,G,B,A] → Argb8 [A,R,G,B])
5. sRGB↔linear round-trip preserves values within ±1/255 tolerance
6. YUYV and NV12 to Rgba8 conversion produces correct RGB values for known test vectors
7. Image↔Tensor round-trip preserves pixel values (normalized to [0,1] in tensor)

### Artifacts

1. `base/src/pixel.rs` — all trait definitions, format types, pixel structs
2. `base/src/image.rs` — Image/ImageView/ImageViewMut with all methods
3. `base/src/lib.rs` — updated with module declarations and re-exports
4. Inline test modules in both pixel.rs and image.rs

## Progress Tracking

- [x] Task 1: Trait hierarchy + GpuFormat enum
- [x] Task 2: Linear format types + pixel structs
- [x] Task 3: Packed + planar format types
- [x] Task 4: Image, ImageView, ImageViewMut core
- [x] Task 5: Pixel access + sub_image
- [x] Task 6: Format conversions
- [x] Task 7: Tensor interop
      **Total Tasks:** 7 | **Completed:** 7 | **Remaining:** 0

## Implementation Tasks

### Task 1: Trait Hierarchy + GpuFormat Enum

**Objective:** Define the PixelFormat trait hierarchy and abstract GPU format enum.
**Dependencies:** None

**Files:**

- Create: `base/src/pixel.rs`
- Modify: `base/src/lib.rs` (add `mod pixel; pub use pixel::*;`)

**Key Decisions / Notes:**

- Traits: `PixelFormat` (base) → `UncompressedFormat` (adds size/stride computation) → `LinearFormat` (associated `Pixel` type + `BYTES_PER_PIXEL`), `PackedFormat` (associated `Macropixel` type + `BYTES_PER_MACROPIXEL` + `PIXELS_PER_MACROPIXEL`), `PlanarFormat` (marker — plane accessors are inherent methods on `Image<ConcreteFormat>`)
- `UncompressedFormat` provides `fn min_stride(width: u32) -> u32` and `fn data_size(width: u32, height: u32, stride: u32) -> usize` as associated functions (not methods — format types are ZSTs)
- `GpuFormat` enum variants: `Rgba8Unorm`, `Rgba8Srgb`, `Bgra8Unorm`, `Bgra8Srgb`, `Rgba16Float`, `Rgba32Float`, `R8Unorm`, `Rg8Unorm` (covers all formats; Argb8 has no standard GPU equivalent → `GPU_FORMAT = None`)
- No implementations yet — just trait definitions and the enum

**Definition of Done:**

- [ ] All 5 traits compile with their associated types/constants/functions
- [ ] GpuFormat enum defined with all variants
- [ ] `pixel` module declared and re-exported in lib.rs — `cargo build -p base` succeeds with no ambiguous glob re-export errors
- [ ] Tests verify trait definitions compile (marker impls on dummy types)

**Verify:**

- `cargo test -p base --lib`

---

### Task 2: Linear Format Types + Pixel Structs

**Objective:** Implement Rgba8, Argb8, Bgra8, Srgba8, RgbaF16, RgbaF32 format types with their pixel structs and full trait implementations.
**Dependencies:** Task 1

**Files:**

- Modify: `base/src/pixel.rs`

**Key Decisions / Notes:**

- Each format is a zero-sized type (ZST): `pub struct Rgba8;`
- Each has an associated `#[repr(C)]` pixel struct: `pub struct Rgba8Pixel { pub r: u8, pub g: u8, pub b: u8, pub a: u8 }`
- Srgba8Pixel is a separate type from Rgba8Pixel even though layout matches — enforces color space distinction at the type level
- RgbaF16Pixel fields are `F16` (from `crate::F16`), RgbaF32Pixel fields are `f32`
- Add compile-time size assertions: `const _: () = assert!(std::mem::size_of::<Rgba8Pixel>() == 4);`
- Argb8 has `GPU_FORMAT = None` (no standard GPU equivalent). All others map to GpuFormat variants.
- `min_stride` = `width * BYTES_PER_PIXEL as u32`, `data_size` = `stride as usize * height as usize`

**Definition of Done:**

- [ ] 6 format ZSTs implement PixelFormat + UncompressedFormat + LinearFormat
- [ ] 6 pixel structs are `#[repr(C)]`, `Copy`, `Clone`, `Debug`, `PartialEq`
- [ ] Compile-time size assertions pass for all pixel structs (4, 4, 4, 4, 8, 16 bytes)
- [ ] GPU format mapping correct for each type
- [ ] Tests: construct each pixel struct, verify byte layout via `std::mem::size_of`, verify GPU_FORMAT values

**Verify:**

- `cargo test -p base --lib`

---

### Task 3: Packed + Planar Format Types

**Objective:** Implement Yuyv8, Nv12, Y8, and Uv8 format types with their associated structs.
**Dependencies:** Task 1

**Files:**

- Modify: `base/src/pixel.rs`

**Key Decisions / Notes:**

- **Y8:** Single-channel 8-bit luminance. `LinearFormat` with `Y8Pixel { pub y: u8 }`. BYTES_PER_PIXEL = 1. GPU_FORMAT = Some(GpuFormat::R8Unorm).
- **Uv8:** Two-channel interleaved 8-bit chroma. `LinearFormat` with `Uv8Pixel { pub u: u8, pub v: u8 }`. BYTES_PER_PIXEL = 2. GPU_FORMAT = Some(GpuFormat::Rg8Unorm).
- **Yuyv8:** `PackedFormat` with `Yuyv8Macropixel { pub y0: u8, pub u: u8, pub y1: u8, pub v: u8 }`. BYTES_PER_MACROPIXEL = 4, PIXELS_PER_MACROPIXEL = 2. min_stride = width * 2 (2 bytes per pixel average). GPU_FORMAT = None.
- **Nv12:** `PlanarFormat`. min_stride = width (Y plane). data_size = stride * height + stride * (height / 2) (Y plane + UV plane; UV stride = Y stride). GPU_FORMAT = None.
- Yuyv8 constructor must enforce even width (assert/panic if odd)

**Definition of Done:**

- [ ] Y8, Uv8 implement LinearFormat with correct pixel types
- [ ] Yuyv8 implements PackedFormat with Yuyv8Macropixel
- [ ] Nv12 implements PlanarFormat with correct data_size (1.5× height factor)
- [ ] Compile-time size assertions: Y8Pixel=1, Uv8Pixel=2, Yuyv8Macropixel=4
- [ ] Tests: construct macropixel struct, verify Nv12 data_size math, verify min_stride for each format

**Verify:**

- `cargo test -p base --lib`

---

### Task 4: Image, ImageView, ImageViewMut Core

**Objective:** Implement the core image container structs with constructors, dimension queries, byte access, row access, and view creation.
**Dependencies:** Task 1 (needs PixelFormat/UncompressedFormat traits)

**Files:**

- Create: `base/src/image.rs`
- Modify: `base/src/lib.rs` (add `mod image; pub use image::*;`)

**Key Decisions / Notes:**

- `Image<P: PixelFormat>` has `width: u32, height: u32, stride: u32, data: Vec<u8>`
- `ImageView<'a, P: PixelFormat>` has `width: u32, height: u32, stride: u32, data: &'a [u8]`
- `ImageViewMut<'a, P: PixelFormat>` has `width: u32, height: u32, stride: u32, data: &'a mut [u8]`
- Constructors on `Image<P> where P: UncompressedFormat`:
  - `new(width, height) -> Self` — zero-filled, stride = min_stride
  - `new_with_stride(width, height, stride) -> Self` — custom stride (asserts stride >= min_stride)
  - `from_raw(width, height, stride, data: Vec<u8>) -> Self` — takes ownership (asserts data.len() >= data_size)
  - `from_data(width, height, data: Vec<u8>) -> Self` — stride = min_stride
- Dimension queries: `width()`, `height()`, `stride()`, `data_size()`
- Byte access: `as_bytes() -> &[u8]`, `as_bytes_mut() -> &mut [u8]`, `into_bytes(self) -> Vec<u8>`
- Row access: `row(y) -> &[u8]`, `row_mut(y) -> &mut [u8]` (returns stride-wide slice)
- View creation: `view(&self) -> ImageView<P>`, `view_mut(&mut self) -> ImageViewMut<P>`
- Same dimension/byte/row methods on ImageView and ImageViewMut
- Follow Tensor/TensorView pattern: no Deref, explicit view() method

**Definition of Done:**

- [ ] All three structs compile and construct for any UncompressedFormat type
- [ ] `cargo build -p base` succeeds with no ambiguous glob re-export errors after adding `pub use image::*` to lib.rs
- [ ] Constructors validate stride and data size with asserts
- [ ] Byte and row access return correct slices
- [ ] view()/view_mut() create borrows without copying
- [ ] Tests: construct Image\<Rgba8\>, verify dimensions, byte access, row access, view creation

**Verify:**

- `cargo test -p base --lib`

---

### Task 5: Pixel Access + sub_image

**Objective:** Add typed pixel access for linear/packed formats, plane accessors for Nv12, and zero-copy sub-image cropping.
**Dependencies:** Task 2, Task 3, Task 4

**Files:**

- Modify: `base/src/image.rs`

**Key Decisions / Notes:**

- **LinearFormat pixel access** on Image/ImageView/ImageViewMut where `P: LinearFormat`:
  - `pixel(x, y) -> &P::Pixel` — pointer cast from byte slice using `#[repr(C)]` guarantee
  - `pixel_mut(x, y) -> &mut P::Pixel` — on Image and ImageViewMut only
  - Bounds check: assert x < width, y < height
  - Byte offset: `y * stride + x * BYTES_PER_PIXEL`
  - **Alignment safety:** For u8-field pixel structs (Rgba8Pixel, etc.), alignment is 1 — always safe. For f32-field structs (RgbaF32Pixel, align=4) and F16-field structs (RgbaF16Pixel, align=2), assert pointer alignment at runtime before casting. The `Image::new*` constructors use `Vec<u8>` which may only be 1-byte aligned; add a debug_assert on alignment in pixel(). For from_raw(), document that callers must provide suitably aligned data for float pixel types.
- **PackedFormat macropixel access** where `P: PackedFormat`:
  - `macropixel(mx, y) -> &P::Macropixel` — mx is macropixel index (0..width/PIXELS_PER_MACROPIXEL)
  - `macropixel_mut(mx, y) -> &mut P::Macropixel`
  - Byte offset: `y * stride + mx * BYTES_PER_MACROPIXEL`
- **Nv12 plane access** as inherent methods on `Image<Nv12>`, `ImageView<Nv12>`, `ImageViewMut<Nv12>`:
  - `y_plane(&self) -> ImageView<Y8>` — view into first `stride * height` bytes, width=self.width, height=self.height, stride=self.stride
  - `uv_plane(&self) -> ImageView<Uv8>` — view into remaining `stride * (height/2)` bytes. Constructed with width=self.width/2, height=self.height/2, stride=self.stride (same stride as Y plane — correct because NV12 UV rows occupy the same byte width as Y rows: width/2 samples × 2 bytes = width bytes)
  - Mutable variants: `y_plane_mut`, `uv_plane_mut` on Image and ImageViewMut
- **sub_image** on ImageView where `P: LinearFormat` (zero-copy crop):
  - `sub_image(x, y, width, height) -> ImageView<P>`
  - Adjusts data pointer and dimensions, keeps same stride
  - Restricted to LinearFormat only — PackedFormat macropixels can't be split at arbitrary x, and PlanarFormat has non-contiguous planes

**Definition of Done:**

- [ ] pixel()/pixel_mut() return correct typed references for Rgba8, RgbaF32
- [ ] pixel() asserts pointer alignment for float pixel types (RgbaF32, RgbaF16)
- [ ] macropixel()/macropixel_mut() work for Yuyv8
- [ ] y_plane()/uv_plane() return correctly-sized ImageView\<Y8\>/ImageView\<Uv8\> for Nv12
- [ ] uv_plane().stride() == y_plane().stride() for Nv12
- [ ] sub_image() returns a cropped view with correct dimensions and offset (LinearFormat only)
- [ ] Tests: write pixel then read back, macropixel round-trip, Nv12 plane dimensions and stride equality, sub_image bounds

**Verify:**

- `cargo test -p base --lib`

---

### Task 6: Format Conversions

**Objective:** Implement the ConvertTo trait and all pixel format conversions.
**Dependencies:** Task 2, Task 3, Task 4, Task 5

**Files:**

- Modify: `base/src/image.rs`

**Key Decisions / Notes:**

- `ConvertTo<Q: PixelFormat>` trait defined in `image.rs` (where `Image<P>` is defined). All `ConvertTo` impls also live in `image.rs`. `pixel.rs` has no knowledge of `image.rs`.
  ```
  pub trait ConvertTo<Q: PixelFormat>: PixelFormat {
      fn convert(src: &Image<Self>) -> Image<Q>;
  }
  ```
- `Image<P>` gets `pub fn convert<Q>(&self) -> Image<Q> where P: ConvertTo<Q>`
- **Channel reorder** (per-pixel byte shuffle, no color math):
  - Rgba8 → Argb8: [R,G,B,A] → [A,R,G,B]
  - Rgba8 → Bgra8: [R,G,B,A] → [B,G,R,A]
  - Argb8 → Rgba8, Argb8 → Bgra8
  - Bgra8 → Rgba8, Bgra8 → Argb8
  - (6 impls total among the 3 types)
- **sRGB ↔ linear float:**
  - Srgba8 → RgbaF32: decode sRGB gamma per channel (u8→f32 via /255.0, then inverse gamma), alpha stays linear
  - RgbaF32 → Srgba8: apply sRGB gamma per channel, quantize (×255 + round + clamp), alpha quantized linearly
- **Precision conversions:**
  - Rgba8 → RgbaF32: straight u8→f32 (/255.0), no gamma (Rgba8 is unspecified color space)
  - RgbaF32 → Rgba8: f32→u8 (×255 + round + clamp)
  - RgbaF16 → RgbaF32: F16::to_f32() per channel
  - RgbaF32 → RgbaF16: F16::from_f32() per channel
- **Video format conversions:**
  - Yuyv8 → Rgba8: unpack macropixels, BT.601 YUV→RGB per pixel, clamp to [0,255]
  - Nv12 → Rgba8: iterate pixels, sample Y plane directly, sample UV with nearest-neighbor (half-res), BT.601 YUV→RGB, clamp
- **sRGB gamma functions** (private helpers):
  - `fn srgb_to_linear(s: f32) -> f32`
  - `fn linear_to_srgb(l: f32) -> f32`

**Definition of Done:**

- [ ] ConvertTo trait defined and Image::convert() dispatches to it
- [ ] Channel reorder: Rgba8→Argb8→Bgra8→Rgba8 round-trip preserves bytes exactly
- [ ] sRGB: known test values match reference (sRGB normalized 0.5 → linear ≈ 0.2140; sRGB u8=188 (normalized ≈ 0.737) → linear ≈ 0.502)
- [ ] Precision: Rgba8(128,64,32,255) → RgbaF32 → Rgba8 round-trip matches original
- [ ] YUYV: known Y,U,V → expected R,G,B (test vector from BT.601 spec)
- [ ] NV12: known Y plane + UV plane → expected RGB output
- [ ] All tests pass

**Verify:**

- `cargo test -p base --lib`

---

### Task 7: Tensor Interop

**Objective:** Convert between Image\<P\> (for LinearFormat types) and Tensor\<f32\> in NCHW layout.
**Dependencies:** Task 2, Task 4, Task 5, Task 6

**Files:**

- Modify: `base/src/image.rs`

**Key Decisions / Notes:**

- `to_tensor_nchw(&self) -> Tensor<f32>` on `Image<P> where P: LinearFormat`:
  - Output shape: [1, C, H, W] where C = BYTES_PER_PIXEL / size_of channel element... actually this needs thought.
  - For Rgba8: C=4, values normalized to [0.0, 1.0] (divide by 255)
  - For RgbaF32: C=4, values taken as-is (already float)
  - For RgbaF16: C=4, values converted via F16::to_f32()
  - For Y8: C=1, values normalized to [0.0, 1.0]
  - For Uv8: C=2, values normalized to [0.0, 1.0]
  - Use existing `Tensor::from_shape_data`
- Need a way to determine channel count and extract channel values as f32.
- Define `TensorConvertible` helper trait in `image.rs` with `const CHANNELS: usize`, `fn pixel_to_channels(pixel: &Self::Pixel, out: &mut [f32])`, and `fn channels_to_pixel(channels: &[f32]) -> Self::Pixel`. Implement it for all LinearFormat types in `image.rs` (alongside ConvertTo impls — `pixel.rs` has no knowledge of `image.rs`).
- `from_tensor_nchw(tensor: &Tensor<f32>) -> Image<P>`:
  - Accepts [1, C, H, W] or [C, H, W]
  - Asserts C matches format channel count
  - Constructs image, fills pixel-by-pixel

**Definition of Done:**

- [ ] to_tensor_nchw produces [1, C, H, W] tensor with correct values
- [ ] from_tensor_nchw reconstructs image from tensor
- [ ] Round-trip: Image\<Rgba8\> → Tensor → Image\<Rgba8\> preserves pixel values (u8 round-trip error at most 1 step; tests use epsilon = 1.0/255.0)
- [ ] Round-trip: Image\<RgbaF32\> → Tensor → Image\<RgbaF32\> preserves pixel values exactly
- [ ] Tests: shape verification, value verification, both directions

**Verify:**

- `cargo test -p base --lib`

## Open Questions

None — all design decisions resolved.

### Deferred Ideas

- CompressedFormat trait + Jpeg/Png types (no pixel access, decode-only)
- Additional linear formats: Rgb8 (3-byte, no alpha), R8, Rg8, Rgba16Unorm
- Additional video formats: I420, P010, UYVY
- SIMD-optimized conversion paths (SSE4/AVX2/NEON)
- Alignment-aware constructors (`new_aligned(width, height, alignment)`)
- DynamicImage enum for runtime format dispatch
- Image I/O (PNG, JPEG, BMP encode/decode)
