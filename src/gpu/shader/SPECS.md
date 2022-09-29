# All Languages

## Execution Model

Vertex
TessellationControl
TessellationEvaluation
Geometry
Fragment
Compute
Kernel
(no ray tracing)


## Types

### Scalar

| Rust   | SPIR-V     | GLSL     | HLSL     | MSL      | WGSL   |
| ------ | ---------- | -------- | -------- | -------- | ------ |
| `void` | 19,id      | `void`   | ?        | `void`   | `void` |
| `bool` | 20,id      | `bool`   | `bool`   | `bool`   | `bool` |
| `u8`   | 21,id,8,0  | `uint`   | `uint`   | `uchar`  | `u32`  |
| `i8`   | 21,id,8,1  | `int`    | `int`    | `char`   | `i32`  |
| `u16`  | 21,id,16,0 | `uint`   | `uint`   | `ushort` | `u32`  |
| `i16`  | 21,id,16,1 | `int`    | `int`    | `short`  | `i32`  |
| `u32`  | 21,id,32,0 | `uint`   | `uint`   | `uint`   | `u32`  |
| `i32`  | 21,id,32,1 | `int`    | `int`    | `int`    | `i32`  |
| `u64`  | 21,id,64,0 | ?        | ?        | `ulong`  | ?      |
| `i64`  | 21,id,64,1 | ?        | ?        | `long`   | ?      |
| `f16`  | 22,id,16   | `float`  | `half`   | `half`   | `f16`  |
| `f32`  | 22,id,32   | `float`  | `float`  | `float`  | `f32`  |
| `f64`  | 22,id,64   | `double` | `double` | ?        | `f64`  |

### Vector

| Rust         | SPIR-V        | GLSL    | HLSL      | MSL       | WGSL         |
| ------------ | ------------- | ------- | --------- | --------- | ------------ |
| `Vec2<bool>` | 23,id,&bool,2 | `bvec2` | `bool2`   | `bool2`   | `vec2<bool>` |
| `Vec2<u8>`   | 23,id,&u8,2   | `uvec2` | `uint2`   | `uchar2`  | `vec2<u32>`  |
| `Vec2<i8>`   | 23,id,&i8,2   | `ivec2` | `int2`    | `char2`   | `vec2<i32>`  |
| `Vec2<u16>`  | 23,id,&u16,2  | `uvec2` | `uint2`   | `ushort2` | `vec2<u32>`  |
| `Vec2<i16>`  | 23,id,&i16,2  | `ivec2` | `int2`    | `short2`  | `vec2<i32>`  |
| `Vec2<u32>`  | 23,id,&u32,2  | `uvec2` | `uint2`   | `uint2`   | `vec2<u32>`  |
| `Vec2<i32>`  | 23,id,&i32,2  | `ivec2` | `int2`    | `int2`    | `vec2<i32>`  |
| `Vec2<u64>`  | 23,id,&u64,2  | ?       | ?         | `ulong2`  | ?            |
| `Vec2<i64>`  | 23,id,&i64,2  | ?       | ?         | `long2`   | ?            |
| `Vec2<f16>`  | 23,id,&f16,2  | `vec2`  | `half2`   | `half2`   | `vec2<f16>`  |
| `Vec2<f32>`  | 23,id,&f32,2  | `vec2`  | `float2`  | `float2`  | `vec2<f32>`  |
| `Vec2<f64>`  | 23,id,&f64,2  | `dvec2` | `double2` | ?         | ?            |
| `Vec3<bool>` | 23,id,&bool,3 | `bvec3` | `bool3`   | `bool3`   | `vec3<bool>` |
| `Vec3<u8>`   | 23,id,&u8,3   | `uvec3` | `uint3`   | `uchar3`  | `vec3<u32>`  |
| `Vec3<i8>`   | 23,id,&i8,3   | `ivec3` | `int3`    | `char3`   | `vec3<i32>`  |
| `Vec3<u16>`  | 23,id,&u16,3  | `uvec3` | `uint3`   | `ushort3` | `vec3<u32>`  |
| `Vec3<i16>`  | 23,id,&i16,3  | `ivec3` | `int3`    | `short3`  | `vec3<i32>`  |
| `Vec3<u32>`  | 23,id,&u32,3  | `uvec3` | `uint3`   | `uint3`   | `vec3<u32>`  |
| `Vec3<i32>`  | 23,id,&i32,3  | `ivec3` | `int3`    | `int3`    | `vec3<i32>`  |
| `Vec3<u64>`  | 23,id,&u64,3  | ?       | ?         | `ulong3`  | ?            |
| `Vec3<i64>`  | 23,id,&i64,3  | ?       | ?         | `long3`   | ?            |
| `Vec3<f16>`  | 23,id,&f16,3  | `vec3`  | `half3`   | `half3`   | `vec3<f16>`  |
| `Vec3<f32>`  | 23,id,&f32,3  | `vec3`  | `float3`  | `float3`  | `vec3<f32>`  |
| `Vec3<f64>`  | 23,id,&f64,3  | `dvec3` | `double3` | ?         | ?            |
| `Vec4<bool>` | 23,id,&bool,4 | `bvec4` | `bool4`   | `bool4`   | `vec4<bool>` |
| `Vec4<u8>`   | 23,id,&u8,4   | `uvec4` | `uint4`   | `uchar4`  | `vec4<u32>`  |
| `Vec4<i8>`   | 23,id,&i8,4   | `ivec4` | `int4`    | `char4`   | `vec4<i32>`  |
| `Vec4<u16>`  | 23,id,&u16,4  | `uvec4` | `uint4`   | `ushort4` | `vec4<u32>`  |
| `Vec4<i16>`  | 23,id,&i16,4  | `ivec4` | `int4`    | `short4`  | `vec4<i32>`  |
| `Vec4<u32>`  | 23,id,&u32,4  | `uvec4` | `uint4`   | `uint4`   | `vec4<u32>`  |
| `Vec4<i32>`  | 23,id,&i32,4  | `ivec4` | `int4`    | `int4`    | `vec4<i32>`  |
| `Vec4<u64>`  | 23,id,&u64,4  | ?       | ?         | `ulong4`  | ?            |
| `Vec4<i64>`  | 23,id,&i64,4  | ?       | ?         | `long4`   | ?            |
| `Vec4<f16>`  | 23,id,&f16,4  | `vec4`  | `half4`   | `half4`   | `vec4<f16>`  |
| `Vec4<f32>`  | 23,id,&f32,4  | `vec4`  | `float4`  | `float4`  | `vec4<f32>`  |
| `Vec4<f64>`  | 23,id,&f64,4  | `dvec4` | `double4` | ?         | ?            |

### Matrix

| Rust          | SPIR-V             | GLSL      | HLSL        | MSL        | WGSL          |
| ------------- | ------------------ | --------- | ----------- | ---------- | ------------- |
| `Mat2x2<f16>` | 24,id,&Vec2<f16>,2 | `mat2x2`  | `half2x2`   | `half2x2`  | `mat2x2<f16>` |
| `Mat2x2<f32>` | 24,id,&Vec2<f32>,2 | `mat2x2`  | `float2x2`  | `float2x2` | `mat2x2<f32>` |
| `Mat2x2<f64>` | 24,id,&Vec2<f64>,2 | `dmat2x2` | `double2x2` | ?          | ?             |
| `Mat2x3<f16>` | 24,id,&Vec3<f16>,2 | `mat2x3`  | `half2x3`   | `half2x3`  | `mat2x3<f16>` |
| `Mat2x3<f32>` | 24,id,&Vec3<f32>,2 | `mat2x3`  | `float2x3`  | `float2x3` | `mat2x3<f32>` |
| `Mat2x3<f64>` | 24,id,&Vec3<f64>,2 | `dmat2x3` | `double2x3` | ?          | ?             |
| `Mat2x4<f16>` | 24,id,&Vec4<f16>,2 | `mat2x4`  | `half2x4`   | `half2x4`  | `mat2x4<f16>` |
| `Mat2x4<f32>` | 24,id,&Vec4<f32>,2 | `mat2x4`  | `float2x4`  | `float2x4` | `mat2x4<f32>` |
| `Mat2x4<f64>` | 24,id,&Vec4<f64>,2 | `dmat2x4` | `double2x4` | ?          | ?             |
| `Mat3x2<f16>` | 24,id,&Vec2<f16>,3 | `mat3x2`  | `half3x2`   | `half3x2`  | `mat3x2<f16>` |
| `Mat3x2<f32>` | 24,id,&Vec2<f32>,3 | `mat3x2`  | `float3x2`  | `float3x2` | `mat3x2<f32>` |
| `Mat3x2<f64>` | 24,id,&Vec2<f64>,3 | `dmat3x2` | `double3x2` | ?          | ?             |
| `Mat3x3<f16>` | 24,id,&Vec3<f16>,3 | `mat3x3`  | `half3x3`   | `half3x3`  | `mat3x3<f16>` |
| `Mat3x3<f32>` | 24,id,&Vec3<f32>,3 | `mat3x3`  | `float3x3`  | `float3x3` | `mat3x3<f32>` |
| `Mat3x3<f64>` | 24,id,&Vec3<f64>,3 | `dmat3x3` | `double3x3` | ?          | ?             |
| `Mat3x4<f16>` | 24,id,&Vec4<f16>,3 | `mat3x4`  | `half3x4`   | `half3x4`  | `mat3x4<f16>` |
| `Mat3x4<f32>` | 24,id,&Vec4<f32>,3 | `mat3x4`  | `float3x4`  | `float3x4` | `mat3x4<f32>` |
| `Mat3x4<f64>` | 24,id,&Vec4<f64>,3 | `dmat3x4` | `double3x4` | ?          | ?             |
| `Mat4x2<f16>` | 24,id,&Vec2<f16>,4 | `mat4x2`  | `half4x2`   | `half4x2`  | `mat4x2<f16>` |
| `Mat4x2<f32>` | 24,id,&Vec2<f32>,4 | `mat4x2`  | `float4x2`  | `float4x2` | `mat4x2<f32>` |
| `Mat4x2<f64>` | 24,id,&Vec2<f64>,4 | `dmat4x2` | `double4x2` | ?          | ?             |
| `Mat4x3<f16>` | 24,id,&Vec3<f16>,4 | `mat4x3`  | `half4x3`   | `half4x3`  | `mat4x3<f16>` |
| `Mat4x3<f32>` | 24,id,&Vec3<f32>,4 | `mat4x3`  | `float4x3`  | `float4x3` | `mat4x3<f32>` |
| `Mat4x3<f64>` | 24,id,&Vec3<f64>,4 | `dmat4x3` | `double4x3` | ?          | ?             |
| `Mat4x4<f16>` | 24,id,&Vec4<f16>,4 | `mat4x4`  | `half4x4`   | `half4x4`  | `mat4x4<f16>` |
| `Mat4x4<f32>` | 24,id,&Vec4<f32>,4 | `mat4x4`  | `float4x4`  | `float4x4` | `mat4x4<f32>` |
| `Mat4x4<f64>` | 24,id,&Vec4<f64>,4 | `dmat4x4` | `double4x4` | ?          | ?             |

### SIMD Groups?

### Atomic Types?

### Pixel Formats (Color)

GLSL: 

| Rust        | SPIR-V | GLSL             | HLSL | MSL               | WGSL          |
| ----------- | ------ | ---------------- | ---- | ----------------- | ------------- |
| `R8UN`      | 15     | `r8`             |      | `R8Unorm`         |               |
| `R8IN`      | 20     | `r8_snorm`       |      | `R8Snorm`         |               |
| `R8U`       | 39     | `r8ui`           |      | `R8Uint`          |               |
| `R8I`       | 29     | `r8i`            |      | `R8Sint`          |               |
| `R16UN`     | 14     | `r16`            |      | `R16Unorm`        |               |
| `R16IN`     | 19     | `r16_snorm`      |      | `R16Snorm`        |               |
| `R16U`      | 38     | `r16ui`          |      | `R16Uint`         |               |
| `R16I`      | 28     | `r16i`           |      | `R16Sint`         |               |
| `R16F`      | 9      | `r16f`           |      | `R16Float`        |               |
| `R32U`      | 33     | `r32ui`          |      | `R32Uint`         | `r32uint`     |
| `R32I`      | 24     | `r32i`           |      | `R32Sint`         | `r32sint`     |
| `R32F`      | 3      | `r32f`           |      | `R32Float`        | `r32float`    |
| `R64U`      | 40     | ?                |      | ?                 |               |
| `R64I`      | 41     | ?                |      | ?                 |               |
| `R64F`      | ?      | ?                |      | ?                 |               |
| `RG8UN`     | 13     | `rg8`            |      | `RG8Unorm`        |               |
| `RG8IN`     | 18     | `rg8_snorm`      |      | `RG8Snorm`        |               |
| `RG8U`      | 37     | `rg8ui`          |      | `RG8Uint`         |               |
| `RG8I`      | 27     | `rg8i`           |      | `RG8Sint`         |               |
| `RG16UN`    | 12     | `rg16`           |      | `RG16Unorm`       |               |
| `RG16IN`    | 19     | `rg16_snorm`     |      | `RG16Snorm`       |               |
| `RG16U`     | 36     | `rg16ui`         |      | `RG16Uint`        |               |
| `RG16I`     | 26     | `rg16i`          |      | `RG16Sint`        |               |
| `RG16F`     | 7      | `rg16f`          |      | `RG16Float`       |               |
| `RG32U`     | 35     | `rg32ui`         |      | `RG32Uint`        | `rg32uint`    |
| `RG32I`     | 25     | `rg32i`          |      | `RG32Sint`        | `rg32sint`    |
| `RG32F`     | 6      | `rg32f`          |      | `RG32Float`       | `rg32float`   |
| `RG64U`     | ?      | ?                |      | ?                 |               |
| `RG64I`     | ?      | ?                |      | ?                 |               |
| `RG64F`     | ?      | ?                |      | ?                 |               |
| `RGBA8UN`   | 4      | `rgba8`          |      | `RGBA8Unorm`      | `rgba8unorm`  |
| `SRGBA8UN`  | ?      | ?                |      | `RGBA8Unorm_sRGB` |               |
| `RGBA8IN`   | 5      | `rgba8_snorm`    |      | `RGBA8Snorm`      | `rgba8snorm`  |
| `RGBA8U`    | 32     | `rgba8ui`        |      | `RGBA8Uint`       |               |
| `RGBA8I`    | 23     | `rgba8i`         |      | `RGBA8Sint`       |               |
| `RGBA16UN`  | 10     | `rgba16`         |      | `RGBA16Unorm`     |               |
| `RGBA16IN`  | 16     | `rgba16_snorm`   |      | `RGBA16Snorm`     |               |
| `RGBA16U`   | 31     | `rgba16ui`       |      | `RGBA16Uint`      | `rgba16uint`  |
| `RGBA16I`   | 22     | `rgba16i`        |      | `RGBA16Sint`      | `rgba16sint`  |
| `RGBA16F`   | 2      | `rgba16f`        |      | `RGBA16Float`     | `rgba16float` |
| `RGBA32U`   | 30     | `rgba32ui`       |      | `RGBA32Uint`      | `rgba32uint`  |
| `RGBA32I`   | 21     | `rgba32i`        |      | `RGBA32Sint`      | `rgba32sint`  |
| `RGBA32F`   | 1      | `rgba32f`        |      | `RGBA32Float`     | `rgba32float` |
| `RGBA64U`   | ?      | ?                |      | ?                 |               |
| `RGBA64I`   | ?      | ?                |      | ?                 |               |
| `RGBA64F`   | ?      | ?                |      | ?                 |               |
| `BGRA8UN`   | (4)    | (rgba8)          |      | `BGRA8Unorm`      |               |
| `SBGRA8UN`  | ?      | ?                |      | `BGRA8Unorm_sRGB` |               |
| `BGRA8IN`   | (5)    | (rgba8_snorm)    |      | `BGRA8Snorm`      |               |
| `BGRA8U`    | (32)   | (rgba8ui)        |      | ?                 |               |
| `BGRA8I`    | (23)   | (rgba8i)         |      | ?                 |               |
| `BGRA16UN`  | (10)   | (rgba16)         |      | ?                 |               |
| `BGRA16IN`  | (16)   | (rgba16_snorm)   |      | ?                 |               |
| `BGRA16U`   | (31)   | (rgba16ui)       |      | ?                 |               |
| `BGRA16I`   | (22)   | (rgba16i)        |      | ?                 |               |
| `BGRA16F`   | (2)    | (rgba16f)        |      | ?                 |               |
| `BGRA32U`   | (30)   | (rgba32ui)       |      | ?                 |               |
| `BGRA32I`   | (21)   | (rgba32i)        |      | ?                 |               |
| `BGRA32F`   | (1)    | (rgba32f)        |      | ?                 |               |
| `BGRA64U`   | ?      | ?                |      | ?                 |               |
| `BGRA64I`   | ?      | ?                |      | ?                 |               |
| `BGRA64F`   | ?      | ?                |      | ?                 |               |
| `ABGR8UN`   | (4)    | (rgba8)          |      | ?                 |               |
| `ABGR8IN`   | (5)    | (rgba8_snorm)    |      | ?                 |               |
| `ABGR8U`    | (32)   | (rgba8ui)        |      | ?                 |               |
| `ABGR8I`    | (23)   | (rgba8i)         |      | ?                 |               |
| `ABGR16UN`  | (10)   | (rgba16)         |      | ?                 |               |
| `ABGR16IN`  | (16)   | (rgba16_snorm)   |      | ?                 |               |
| `ABGR16U`   | (31)   | (rgba16ui)       |      | ?                 |               |
| `ABGR16I`   | (22)   | (rgba16i)        |      | ?                 |               |
| `ABGR16F`   | (2)    | (rgba16f)        |      | ?                 |               |
| `ABGR32U`   | (30)   | (rgba32ui)       |      | ?                 |               |
| `ABGR32I`   | (21)   | (rgba32i)        |      | ?                 |               |
| `ABGR32F`   | (1)    | (rgba32f)        |      | ?                 |               |
| `ABGR64U`   | ?      | ?                |      | ?                 |               |
| `ABGR64I`   | ?      | ?                |      | ?                 |               |
| `ABGR64F`   | ?      | ?                |      | ?                 |               |
| `ARGB8UN`   | (4)    | (rgba8)          |      | ?                 |               |
| `ARGB8IN`   | (5)    | (rgba8_snorm)    |      | ?                 |               |
| `ARGB8U`    | (32)   | (rgba8ui)        |      | ?                 |               |
| `ARGB8I`    | (23)   | (rgba8i)         |      | ?                 |               |
| `ARGB16UN`  | (10)   | (rgba16)         |      | ?                 |               |
| `ARGB16IN`  | (16)   | (rgba16_snorm)   |      | ?                 |               |
| `ARGB16U`   | (31)   | (rgba16ui)       |      | ?                 |               |
| `ARGB16I`   | (22)   | (rgba16i)        |      | ?                 |               |
| `ARGB16F`   | (2)    | (rgba16f)        |      | ?                 |               |
| `ARGB32U`   | (30)   | (rgba32ui)       |      | ?                 |               |
| `ARGB32I`   | (21)   | (rgba32i)        |      | ?                 |               |
| `ARGB32F`   | (1)    | (rgba32f)        |      | ?                 |               |
| `ARGB64U`   | ?      | ?                |      | ?                 |               |
| `ARGB64I`   | ?      | ?                |      | ?                 |               |
| `ARGB64F`   | ?      | ?                |      | ?                 |               |
| `A1RGB5`    | (4)    | ?                |      | ?                 |               |
| `RGB5A1`    | (4)    | ?                |      | ?                 |               |
| `R5G6B5`    | (4)    | ?                |      | ?                 |               |
| `RGB10A2UN` | (10)   | `rgb10_a2`       |      | `RGB10A2Unorm`    |               |
| `RGB10A2U`  | (31)   | `rgb10_a2ui`     |      | ?                 |               |
| `RGB9E5F`   | (2)    | ?                |      | `RGB9E5Float`     |               |
| `RG11B10F`  | (2)    | `r11f_g11f_b10f` |      | `RG11B10Float`    |               |

### Pixel Formats (Depth/Stencil)

### Samplers

| Rust                      | SPIR-V | GLSL     | HLSL         | MSL     |
| ------------------------- | ------ | -------- | ------------ | ------- |
| Sampler<T>                | 26,id  | sampler* | SamplerState | sampler |

### Textures

| Rust | SPIR-V | GLSL | HLSL | MSL |
| ---- | ------ | ---- | ---- | --- |
| Texture1D
| Texture1DArray
| Texture2D
| Texture2DArray
| Texture3D
| TextureCube
| TextureCubeArray
| Texture2DMS
| Texture2DMSArray
| Depth2D
| Depth2DArray
| DepthCubeArray
| Depth2DMS
| Depth2DMSArray

OpTypeSampledImage?
texture*

### Images

| Rust | SPIR-V | GLSL | HLSL | MSL |
| ---- | ------ | ---- | ---- | --- |

OpTypeImage?
image*

### Buffers

### Texture Buffers?

### Arrays

### Structs

### Unions

### Enums

## Functions

### Built-in Functions

#### Boolean Scalar and Vector

| Rust            | GLSL         | HLSL | MSL             | WGSL |
| --------------- | ------------ | ---- | --------------- | ---- |
| `x.select(a,b)` | `mix(y,z,x)` |      | `select(y,z,x)` |      |
| `x.all()`       | `all(x)`     |      | `all(x)`        |      |
| `x.any()`       | `any(x)`     |      | `any(x)`        |      |
| `x.not()`       | `not(x)`     |      | `not(x)`        |      |

#### Integer and Float Scalar and Vector

| Rust            | GLSL           | HLSL           | MSL            | WGSL           |
| --------------- | -------------- | -------------- | -------------- | -------------- |
| `x.abs()`       | `abs(x)`       | `abs(x)`       | `abs(x)`       | `abs(x)`       |
| `x.signum()`    | `sign(x)`      | `sign(x)`      | `sign(x)`      | `sign(x)`      |
| `x.min(y)`      | `min(x,y)`     | `min(x,y)`     | `min(x,y)`     | `min(x,y)`     |
| `x.max(y)`      | `max(x,y)`     | `max(x,y)`     | `max(x,y)`     | `max(x,y)`     |
| `x.clamp(l,h)`  | `clamp(x,l,h)` | `clamp(x,l,h)` | `clamp(x,l,h)` | `clamp(x,l,h)` |
| `x.sclamp(l,h)` | `clamp(x,l,h)` | `clamp(x,l,h)` | `clamp(x,l,h)` | `clamp(x,l,h)` |

#### Float Scalar and Vector

| Rust                 | GLSL                | HLSL                   | MSL                 | WGSL                |
| -------------------- | ------------------- | ---------------------- | ------------------- | ------------------- |
| `x.to_radians()`     | `radians(x)`        | `radians(x)`           | `(M_PI/180.0)*x`    | `radians(x)`        |
| `x.to_degrees()`     | `degrees(x)`        | `degrees(x)`           | `(180.0/M_PI)*x`    | `degrees(x)`        |
| `x.sin()`            | `sin(x)`            | `sin(x)`               | `sin(x)`            | `sin(x)`            |
| `x.cos()`            | `cos(x)`            | `cos(x)`               | `cos(x)`            | `cos(x)`            |
| `x.tan()`            | `tan(x)`            | `tan(x)`               | `tan(x)`            | `tan(x)`            |
| `x.sinh()`           | `sinh(x)`           | `sinh(x)`              | `sinh(x)`           | `sinh(x)`           |
| `x.cosh()`           | `cosh(x)`           | `cosh(x)`              | `cosh(x)`           | `cosh(x)`           |
| `x.tanh()`           | `tanh(x)`           | `tanh(x)`              | `tanh(x)`           | `tanh(x)`           |
| `x.asin()`           | `asin(x)`           | `asin(x)`              | `asin(x)`           | `asin(x)`           |
| `x.acos()`           | `acos(x)`           | `acos(x)`              | `acos(x)`           | `acos(x)`           |
| `x.atan()`           | `atan(x)`           | `atan(x)`              | `atan(x)`           | `atan(x)`           |
| `x.atan2(y)`         | `atan2(x,y)`        | `atan2(x,y)`           | `atan2(x,y)`        | `atan2(x,y)`        |
| `x.asinh()`          | `asinh(x)`          | `log(x+sqrt(1+x*x))`   | `asinh(x)`          | `asinh(x)`          |
| `x.acosh()`          | `acosh(x)`          | `log(x+sqrt(x*x-1))`   | `acosh(x)`          | `acosh(x)`          |
| `x.atanh()`          | `atanh(x)`          | `0.5*log((1+x)/(1-x))` | `atanh(x)`          | `atanh(x)`          |
| `x.powf(y)`          | `pow(x,y)`          | `pow(x,y)`             | `pow(x,y)`          | `pow(x,y)`          |
| `x.spowf(y)`         | `pow(x,y)`          | `pow(x,y)`             | `pow(x,y)`          | `pow(x,y)`          |
| `x.exp()`            | `exp(x)`            | `exp(x)`               | `exp(x)`            | `exp(x)`            |
| `x.ln()`             | `log(x)`            | `log(x)`               | `log(x)`            | `log(x)`            |
| `x.exp2()`           | `exp2(x)`           | `exp2(x)`              | `exp2(x)`           | `exp2(x)`           |
| `x.log2()`           | `log2(x)`           | `log2(x)`              | `log2(x)`           | `log2(x)`           |
| `x.sqrt()`           | `sqrt(x)`           | `sqrt(x)`              | `sqrt(x)`           | `sqrt(x)`           |
| `x.invsqrt()`        | `inversesqrt(x)`    | `rsqrt(x)`             | `rsqrt(x)`          | `inverseSqrt(x)`    |
| `x.floor()`          | `floor(x)`          | `floor(x)`             | `floor(x)`          | `floor(x)`          |
| `x.trunc()`          | `trunc(x)`          | `trunc(x)`             | `trunc(x)`          | `trunc(x)`          |
| `x.round()`          | `round(x)`          | `round(x)`             | `round(x)`          | `round(x)`          |
| `x.ceil()`           | `ceil(x)`           | `ceil(x)`              | `ceil(x)`           | `ceil(x)`           |
| `x.fract()`          | `fract(x)`          | `frac(x)`              | `fract(x)`          | `fract(x)`          |
| `x.rem_euclid(y)`    | `mod(x,y)`          | `fmod(x,y)`            | `fmod(x,y)`         | `mod(x,y)`          |
| `x.modf(y)`          | `modf(x,y)`         | `modf(x,y)`            | `modf(x,y)`         | `modf(x,y)`         |
| `x.mix(y,a)`         | `mix(x,y,a)`        | `lerp(x,y,a)`          | `mix(x,y,a)`        | `mix(x,y,a)`        |
| `x.smix(y,a)`        | `mix(x,y,a)`        | `lerp(x,y,a)`          | `mix(x,y,a)`        | `mix(x,y,a)`        |
| `x.step(y)`          | `step(x,y)`         | `step(x,y)`            | `step(x,y)`         | `step(x,y)`         |
| `x.sstep(y)`         | `step(x,y)`         | `step(x,y)`            | `step(x,y)`         | `step(x,y)`         |
| `x.smoothstep(y,z)`  | `smoothstep(y,z,x)` | `smoothstep(y,z,x)`    | `smoothstep(y,z,x)` | `smoothstep(y,z,x)` |
| `x.ssmoothstep(y,z)` | `smoothstep(y,z,x)` | `smoothstep(y,z,x)`    | `smoothstep(y,z,x)` | `smoothstep(y,z,x)` |
| `x.is_nan()`         | `isnan(x)`          | `isnan(x)`             | `isnan(x)`          |                     |
| `x.is_infinite()`    | `isinf(x)`          | `isinf(x)`             | `isinf(x)`          |                     |
| `x.fma(y,z)`         | `fma(x,y,z)`        | `fma(x,y,z)`           | `fma(x,y,z)`        | `fma(x,y,z)`        |

not supported: roundEven, frexp, ldexp, packUnorm2x16, packSnorm2x16, packUnorm4x8, packSnorm4x8, unpackUnorm2x16, unpackSnorm2x16, unpackUnorm4x8, unpackSnorm4x8, packHalf2x16, unpackHalf2x16, packDouble2x32, unpackDouble2x32, 

#### Vector

| Rust                      | GLSL                    | HLSL                 | MSL                | WGSL                 |
| ------------------------- | ----------------------- | -------------------- | ------------------ | -------------------- |
| `x.length()`              | `length(x)`             | `length(x)`          |                    | `length(x)`          |
| `x.distance(y)`           | `distance(x,y)`         | `distance(x,y)`      |                    | `distance(x,y)`      |
| `x.dot(y)`                | `dot(x,y)`              | `dot(x,y)`           |                    | `dot(x,y)`           |
| `x.cross(y)`              | `cross(x,y)`            | `cross(x,y)`         |                    | `cross(x,y)`         |
| `x.normalize()`           | `normalize(x)`          | `normalize(x)`       |                    | `normalize(x)`       |
| `x.faceforward(i,n)`      | `faceforward(x,i,n)`    | `faceforward(x,i,n)` |                    | `faceForward(x,i,n)` |
| `x.reflect(n)`            | `reflect(x,n)`          | `reflect(x,n)`       |                    | `reflect(x,n)`       |
| `x.refract(n,e)`          | `refract(x,n,e)`        | `refract(x,n,e)`     |                    | `refract(x,n,e)`     |
| `x.outer(y)`              | `outerProduct(x,y)`     |
| `x.less_than(y)`          | `lessThan(x,y)`         |
| `x.less_than_equal(y)`    | `lessThanEqual(x,y)`    |
| `x.greater_than(y)`       | `greaterThan(x,y)`      |
| `x.greater_than_equal(y)` | `greaterThanEqual(x,y)` |
| `x.equal(y)`              | `equal(x,y)`            |
| `x.not_equal(y)`          | `notEqual(x,y)`         |

#### Matrix

| Rust              | GLSL                  | HLSL               | MSL                | WGSL               |
| ----------------- | --------------------- | ------------------ | ------------------ | ------------------ |
| `x.compmul(y)`    | `matrixCompMult(x,y)` |
| `x.transpose()`   | `transpose(x)`        |
| `x.determinant()` | `determinant(x)`      |
| `x.inverse()`     | `inverse(x)`          |



fn uaddCarry(x: U,y: U,carry: &mut U);
fn usubBorrow(x: U,y: U,carry: &mut U);
fn umulExtended(x: U,y: U,mut msb: U,lsb: &mut U);
fn imulExtended(x: I,y: I,mut msb: I,lsb: &mut I);
fn bitfieldExtract(value: T,offset: T,bits: T);
fn bitfieldInsert(base: T,insert: T,offset: T,bits: T);
fn bitfieldReverse(value: T);
fn bitCount(value: T);
fn findLSB(value: T);
fn findMSB(value: T);
fn textureSize(sampler: ?,lod: T) -> ?;
fn textureQueryLod(sampler: ?,p: ?) -> ?;
fn textureQueryLevels(sampler: ?) -> I;
texture
textureProj
textureLod
textureOffset
texelFetch
texelFetchOffset
textureProjOffset
textureLodOffset
textureProjLod
textureProjLodOffset
textureGrad
textureGradOffset
textureProjGrad
textureProjGradOffset
textureGather
textureGatherOffset
atomicCounterIncrement
atomicCounterDecrement
atomicCounter
atomicCounterAdd
atomucCounterSubtract
atomicCounterMin
atomicCounterMax
atomicCounterAnd
atomicCounterOr
atomicCounterXor
atomicCounterExchange
atomicCounterCompSwap
atomicAdd
atomicMin
atomicMax
atomicAnd
atomicOr
atomicXor
atomicExchange
atomicCompSwap
imageSize
imageSamples
imageLoad
imageStore
imageAtomicAdd
imageAtomicMin
imageAtomicMax
imageAtomicAnd
imageAtomicOr
imageAtomicXor
imageAtomicExchange
imageAtomicCompSwap
emitStreamVertex
endStreamPrimitive
emitVertex
endPrimitive
dFdx
dFdy
dFdxFine
dFdyFine
dFdxCoarse
dFdyCoarse
fwidth
fwidthFine
fwidthCoarse
interpolateAtCentroid
interpolateAtSample
interpolateAtOffset
noise1
noise2
noise3
noise4
barrier
memoryBarrier
memoryBarrierAtomicCounter
memoryBarrierBuffer
memoryBarrierShared
memoryBarrierImage
groupMemoryBarrier
subpassLoad
anyInvocation
allInvocations
allInvocationsEqual

## Statements

### If

### While

### Switch

### For

### Break

### Continue

### Return

## Expressions

### Field

### Index

### Swizzling
