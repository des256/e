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

| Rust | SPIR-V     | GLSL   | HLSL   | MSL    | WGSL |
| ---- | ---------- | ------ | ------ | ------ | ---- |
| void | 19,id      | void   | ?      | void   | void |
| bool | 20,id      | bool   | bool   | bool   | bool |
| u8   | 21,id,8,0  | uint   | uint   | uchar  | :
| i8   | 21,id,8,1  | int    | int    | char   |
| u16  | 21,id,16,0 | uint   | uint   | ushort |
| i16  | 21,id,16,1 | int    | int    | short  |
| u32  | 21,id,32,0 | uint   | uint   | uint   |
| i32  | 21,id,32,1 | int    | int    | int    |
| u64  | 21,id,64,0 | ?      | ?      | ulong  |
| i64  | 21,id,64,1 | ?      | ?      | long   |
| f16  | 22,id,16   | float  | half   | half   |
| f32  | 22,id,32   | float  | float  | float  |
| f64  | 22,id,64   | double | double | ?      |

### Vector

| Rust       | SPIR-V        | GLSL  | HLSL    | MSL     |
| ---------- | ------------- | ----- | ------- | ------- |
| Vec2<bool> | 23,id,&bool,2 | bvec2 | bool2   | bool2   |
| Vec2<u8>   | 23,id,&u8,2   | uvec2 | uint2   | uchar2  |
| Vec2<i8>   | 23,id,&i8,2   | ivec2 | int2    | char2   |
| Vec2<u16>  | 23,id,&u16,2  | uvec2 | uint2   | ushort2 |
| Vec2<i16>  | 23,id,&i16,2  | ivec2 | int2    | short2  |
| Vec2<u32>  | 23,id,&u32,2  | uvec2 | uint2   | uint2   |
| Vec2<i32>  | 23,id,&i32,2  | ivec2 | int2    | int2    |
| Vec2<u64>  | 23,id,&u64,2  | ?     | ?       | ulong2  |
| Vec2<i64>  | 23,id,&i64,2  | ?     | ?       | long2   |
| Vec2<f16>  | 23,id,&f16,2  | vec2  | half2   | half2   |
| Vec2<f32>  | 23,id,&f32,2  | vec2  | float2  | float2  |
| Vec2<f64>  | 23,id,&f64,2  | dvec2 | double2 | ?       |
| Vec3<bool> | 23,id,&bool,3 | bvec3 | bool3   | bool3   |
| Vec3<u8>   | 23,id,&u8,3   | uvec3 | uint3   | uchar3  |
| Vec3<i8>   | 23,id,&i8,3   | ivec3 | int3    | char3   |
| Vec3<u16>  | 23,id,&u16,3  | uvec3 | uint3   | ushort3 |
| Vec3<i16>  | 23,id,&i16,3  | ivec3 | int3    | short3  |
| Vec3<u32>  | 23,id,&u32,3  | uvec3 | uint3   | uint3   |
| Vec3<i32>  | 23,id,&i32,3  | ivec3 | int3    | int3    |
| Vec3<u64>  | 23,id,&u64,3  | ?     | ?       | ulong3  |
| Vec3<i64>  | 23,id,&i64,3  | ?     | ?       | long3   |
| Vec3<f16>  | 23,id,&f16,3  | vec3  | half3   | half3   |
| Vec3<f32>  | 23,id,&f32,3  | vec3  | float3  | float3  |
| Vec3<f64>  | 23,id,&f64,3  | dvec3 | double3 | ?       |
| Vec4<bool> | 23,id,&bool,4 | bvec4 | bool4   | bool4   |
| Vec4<u8>   | 23,id,&u8,4   | uvec4 | uint4   | uchar4  |
| Vec4<i8>   | 23,id,&i8,4   | ivec4 | int4    | char4   |
| Vec4<u16>  | 23,id,&u16,4  | uvec4 | uint4   | ushort4 |
| Vec4<i16>  | 23,id,&i16,4  | ivec4 | int4    | short4  |
| Vec4<u32>  | 23,id,&u32,4  | uvec4 | uint4   | uint4   |
| Vec4<i32>  | 23,id,&i32,4  | ivec4 | int4    | int4    |
| Vec4<u64>  | 23,id,&u64,4  | ?     | ?       | ulong4  |
| Vec4<i64>  | 23,id,&i64,4  | ?     | ?       | long4   |
| Vec4<f16>  | 23,id,&f16,4  | vec4  | half4   | half4   |
| Vec4<f32>  | 23,id,&f32,4  | vec4  | float4  | float4  |
| Vec4<f64>  | 23,id,&f64,4  | dvec4 | double4 | ?       |

### Matrix

| Rust        | SPIR-V             | GLSL    | HLSL      | MSL      |
| ----------- | ------------------ | ------- | --------- | -------- |
| Mat2x2<f16> | 24,id,&Vec2<f16>,2 | mat2x2  | half2x2   | half2x2  |
| Mat2x2<f32> | 24,id,&Vec2<f32>,2 | mat2x2  | float2x2  | float2x2 |
| Mat2x2<f64> | 24,id,&Vec2<f64>,2 | dmat2x2 | double2x2 | ?        |
| Mat2x3<f16> | 24,id,&Vec3<f16>,2 | mat2x3  | half2x3   | half2x3  |
| Mat2x3<f32> | 24,id,&Vec3<f32>,2 | mat2x3  | float2x3  | float2x3 |
| Mat2x3<f64> | 24,id,&Vec3<f64>,2 | dmat2x3 | double2x3 | ?        |
| Mat2x4<f16> | 24,id,&Vec4<f16>,2 | mat2x4  | half2x4   | half2x4  |
| Mat2x4<f32> | 24,id,&Vec4<f32>,2 | mat2x4  | float2x4  | float2x4 |
| Mat2x4<f64> | 24,id,&Vec4<f64>,2 | dmat2x4 | double2x4 | ?        |
| Mat3x2<f16> | 24,id,&Vec2<f16>,3 | mat3x2  | half3x2   | half3x2  |
| Mat3x2<f32> | 24,id,&Vec2<f32>,3 | mat3x2  | float3x2  | float3x2 |
| Mat3x2<f64> | 24,id,&Vec2<f64>,3 | dmat3x2 | double3x2 | ?        |
| Mat3x3<f16> | 24,id,&Vec3<f16>,3 | mat3x3  | half3x3   | half3x3  |
| Mat3x3<f32> | 24,id,&Vec3<f32>,3 | mat3x3  | float3x3  | float3x3 |
| Mat3x3<f64> | 24,id,&Vec3<f64>,3 | dmat3x3 | double3x3 | ?        |
| Mat3x4<f16> | 24,id,&Vec4<f16>,3 | mat3x4  | half3x4   | half3x4  |
| Mat3x4<f32> | 24,id,&Vec4<f32>,3 | mat3x4  | float3x4  | float3x4 |
| Mat3x4<f64> | 24,id,&Vec4<f64>,3 | dmat3x4 | double3x4 | ?        |
| Mat4x2<f16> | 24,id,&Vec2<f16>,4 | mat4x2  | half4x2   | half4x2  |
| Mat4x2<f32> | 24,id,&Vec2<f32>,4 | mat4x2  | float4x2  | float4x2 |
| Mat4x2<f64> | 24,id,&Vec2<f64>,4 | dmat4x2 | double4x2 | ?        |
| Mat4x3<f16> | 24,id,&Vec3<f16>,4 | mat4x3  | half4x3   | half4x3  |
| Mat4x3<f32> | 24,id,&Vec3<f32>,4 | mat4x3  | float4x3  | float4x3 |
| Mat4x3<f64> | 24,id,&Vec3<f64>,4 | dmat4x3 | double4x3 | ?        |
| Mat4x4<f16> | 24,id,&Vec4<f16>,4 | mat4x4  | half4x4   | half4x4  |
| Mat4x4<f32> | 24,id,&Vec4<f32>,4 | mat4x4  | float4x4  | float4x4 |
| Mat4x4<f64> | 24,id,&Vec4<f64>,4 | dmat4x4 | double4x4 | ?        |

### SIMD Groups?

### Atomic Types?

### Pixel Formats (Color)

GLSL: 

| Rust      | SPIR-V | GLSL           | HLSL | MSL             |
| --------- | ------ | -------------- | ---- | --------------- |
| R8UN      | 15     | r8             |      | R8Unorm         |
| R8IN      | 20     | r8_snorm       |      | R8Snorm         |
| R8U       | 39     | r8ui           |      | R8Uint          |
| R8I       | 29     | r8i            |      | R8Sint          |
| R16UN     | 14     | r16            |      | R16Unorm        |
| R16IN     | 19     | r16_snorm      |      | R16Snorm        |
| R16U      | 38     | r16ui          |      | R16Uint         |
| R16I      | 28     | r16i           |      | R16Sint         |
| R16F      | 9      | r16f           |      | R16Float        |
| R32U      | 33     | r32ui          |      | R32Uint         |
| R32I      | 24     | r32i           |      | R32Sint         |
| R32F      | 3      | r32f           |      | R32Float        |
| R64U      | 40     | ?              |      | ?               |
| R64I      | 41     | ?              |      | ?               |
| R64F      | ?      | ?              |      | ?               |
| A8UN      | 15     | r8             |      | R8Unorm         |
| A8IN      | 20     | r8_snorm       |      | R8Snorm         |
| A8U       | 39     | r8ui           |      | R8Uint          |
| A8I       | 29     | r8i            |      | R8Sint          |
| A16UN     | 14     | r16            |      | R16Unorm        |
| A16IN     | 19     | r16_snorm      |      | R16Snorm        |
| A16U      | 38     | r16ui          |      | R16Uint         |
| A16I      | 28     | r16i           |      | R16Sint         |
| A16F      | 9      | r16f           |      | R16Float        |
| A32U      | 33     | r32ui          |      | R32Uint         |
| A32I      | 24     | r32i           |      | R32Sint         |
| A32F      | 3      | r32f           |      | R32Float        |
| A64U      | 40     | ?              |      | ?               |
| A64I      | 41     | ?              |      | ?               |
| A64F      | ?      | ?              |      | ?               |
| I8UN      | 15     | r8             |      | R8Unorm         |
| I8IN      | 20     | r8_snorm       |      | R8Snorm         |
| I8U       | 39     | r8ui           |      | R8Uint          |
| I8I       | 29     | r8i            |      | R8Sint          |
| I16UN     | 14     | r16            |      | R16Unorm        |
| I16IN     | 19     | r16_snorm      |      | R16Snorm        |
| I16U      | 38     | r16ui          |      | R16Uint         |
| I16I      | 28     | r16i           |      | R16Sint         |
| I16F      | 9      | r16f           |      | R16Float        |
| I32U      | 33     | r32ui          |      | R32Uint         |
| I32I      | 24     | r32i           |      | R32Sint         |
| I32F      | 3      | r32f           |      | R32Float        |
| I64U      | 40     | ?              |      | ?               |
| I64I      | 41     | ?              |      | ?               |
| I64F      | ?      | ?              |      | ?               |
| L8UN      | 15     | r8             |      | R8Unorm         |
| L8IN      | 20     | r8_snorm       |      | R8Snorm         |
| L8U       | 39     | r8ui           |      | R8Uint          |
| L8I       | 29     | r8i            |      | R8Sint          |
| L16UN     | 14     | r16            |      | R16Unorm        |
| L16IN     | 19     | r16_snorm      |      | R16Snorm        |
| L16U      | 38     | r16ui          |      | R16Uint         |
| L16I      | 28     | r16i           |      | R16Sint         |
| L16F      | 9      | r16f           |      | R16Float        |
| L32U      | 33     | r32ui          |      | R32Uint         |
| L32I      | 24     | r32i           |      | R32Sint         |
| L32F      | 3      | r32f           |      | R32Float        |
| L64U      | 40     | ?              |      | ?               |
| L64I      | 41     | ?              |      | ?               |
| L64F      | ?      | ?              |      | ?               |
| RG8UN     | 13     | rg8            |      | RG8Unorm        |
| RG8IN     | 18     | rg8_snorm      |      | RG8Snorm        |
| RG8U      | 37     | rg8ui          |      | RG8Uint         |
| RG8I      | 27     | rg8i           |      | RG8Sint         |
| RG16UN    | 12     | rg16           |      | RG16Unorm       |
| RG16IN    | 19     | rg16_snorm     |      | RG16Snorm       |
| RG16U     | 36     | rg16ui         |      | RG16Uint        |
| RG16I     | 26     | rg16i          |      | RG16Sint        |
| RG16F     | 7      | rg16f          |      | RG16Float       |
| RG32U     | 35     | rg32ui         |      | RG32Uint        |
| RG32I     | 25     | rg32i          |      | RG32Sint        |
| RG32F     | 6      | rg32f          |      | RG32Float       |
| RG64U     | ?      | ?              |      | ?               |
| RG64I     | ?      | ?              |      | ?               |
| RG64F     | ?      | ?              |      | ?               |
| RGBA8UN   | 4      | rgba8          |      | RGBA8Unorm      |
| SRGBA8UN  | ?      | ?              |      | RGBA8Unorm_sRGB |
| RGBA8IN   | 5      | rgba8_snorm    |      | RGBA8Snorm      |
| RGBA8U    | 32     | rgba8ui        |      | RGBA8Uint       |
| RGBA8I    | 23     | rgba8i         |      | RGBA8Sint       |
| RGBA16UN  | 10     | rgba16         |      | RGBA16Unorm     |
| RGBA16IN  | 16     | rgba16_snorm   |      | RGBA16Snorm     |
| RGBA16U   | 31     | rgba16ui       |      | RGBA16Uint      |
| RGBA16I   | 22     | rgba16i        |      | RGBA16Sint      |
| RGBA16F   | 2      | rgba16f        |      | RGBA16Float     |
| RGBA32U   | 30     | rgba32ui       |      | RGBA32Uint      |
| RGBA32I   | 21     | rgba32i        |      | RGBA32Sint      |
| RGBA32F   | 1      | rgba32f        |      | RGBA32Float     |
| RGBA64U   | ?      | ?              |      | ?               |
| RGBA64I   | ?      | ?              |      | ?               |
| RGBA64F   | ?      | ?              |      | ?               |
| BGRA8UN   | (4)    | (rgba8)        |      | BGRA8Unorm      |
| SBGRA8UN  | ?      | ?              |      | BGRA8Unorm_sRGB |
| BGRA8IN   | (5)    | (rgba8_snorm)  |      | BGRA8Snorm      |
| BGRA8U    | (32)   | (rgba8ui)      |      | ?               |
| BGRA8I    | (23)   | (rgba8i)       |      | ?               |
| BGRA16UN  | (10)   | (rgba16)       |      | ?               |
| BGRA16IN  | (16)   | (rgba16_snorm) |      | ?               |
| BGRA16U   | (31)   | (rgba16ui)     |      | ?               |
| BGRA16I   | (22)   | (rgba16i)      |      | ?               |
| BGRA16F   | (2)    | (rgba16f)      |      | ?               |
| BGRA32U   | (30)   | (rgba32ui)     |      | ?               |
| BGRA32I   | (21)   | (rgba32i)      |      | ?               |
| BGRA32F   | (1)    | (rgba32f)      |      | ?               |
| BGRA64U   | ?      | ?              |      | ?               |
| BGRA64I   | ?      | ?              |      | ?               |
| BGRA64F   | ?      | ?              |      | ?               |
| ABGR8UN   | (4)    | (rgba8)        |      | ?               |
| ABGR8IN   | (5)    | (rgba8_snorm)  |      | ?               |
| ABGR8U    | (32)   | (rgba8ui)      |      | ?               |
| ABGR8I    | (23)   | (rgba8i)       |      | ?               |
| ABGR16UN  | (10)   | (rgba16)       |      | ?               |
| ABGR16IN  | (16)   | (rgba16_snorm) |      | ?               |
| ABGR16U   | (31)   | (rgba16ui)     |      | ?               |
| ABGR16I   | (22)   | (rgba16i)      |      | ?               |
| ABGR16F   | (2)    | (rgba16f)      |      | ?               |
| ABGR32U   | (30)   | (rgba32ui)     |      | ?               |
| ABGR32I   | (21)   | (rgba32i)      |      | ?               |
| ABGR32F   | (1)    | (rgba32f)      |      | ?               |
| ABGR64U   | ?      | ?              |      | ?               |
| ABGR64I   | ?      | ?              |      | ?               |
| ABGR64F   | ?      | ?              |      | ?               |
| ARGB8UN   | (4)    | (rgba8)        |      | ?               |
| ARGB8IN   | (5)    | (rgba8_snorm)  |      | ?               |
| ARGB8U    | (32)   | (rgba8ui)      |      | ?               |
| ARGB8I    | (23)   | (rgba8i)       |      | ?               |
| ARGB16UN  | (10)   | (rgba16)       |      | ?               |
| ARGB16IN  | (16)   | (rgba16_snorm) |      | ?               |
| ARGB16U   | (31)   | (rgba16ui)     |      | ?               |
| ARGB16I   | (22)   | (rgba16i)      |      | ?               |
| ARGB16F   | (2)    | (rgba16f)      |      | ?               |
| ARGB32U   | (30)   | (rgba32ui)     |      | ?               |
| ARGB32I   | (21)   | (rgba32i)      |      | ?               |
| ARGB32F   | (1)    | (rgba32f)      |      | ?               |
| ARGB64U   | ?      | ?              |      | ?               |
| ARGB64I   | ?      | ?              |      | ?               |
| ARGB64F   | ?      | ?              |      | ?               |
| A1RGB5    | (4)    | ?              |      | ?               |
| RGB5A1    | (4)    | ?              |      | ?               |
| R5G6B5    | (4)    | ?              |      | ?               |
| RGB10A2UN | (10)   | rgb10_a2       |      | RGB10A2Unorm    |
| RGB10A2U  | (31)   | rgb10_a2ui     |      | ?               |
| RGB9E5F   | (2)    | ?              |      | RGB9E5Float     |
| RG11B10F  | (2)    | r11f_g11f_b10f |      | RG11B10Float    |

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

fn radians(x: F) -> F;
fn degrees(x: F) -> F;
fn sin(x: F) -> F;
fn cos(x: F) -> F;
fn tan(x: F) -> F;
fn asin(x: F) -> F;
fn acos(x: F) -> F;
fn atan(y: F,x: F) -> F;
fn sinh(x: F) -> F;
fn cosh(x: F) -> F;
fn tanh(x: F) -> F;
fn asinh(x: F) -> F;
fn acosh(x: F) -> F;
fn atanh(x: F) -> F;
fn pow(x: F,y: F) -> F;
fn exp(x: F) -> F;
fn log(x: F) -> F;
fn exp2(x: F) -> F;
fn log2(x: F) -> F;
fn sqrt(x: F) -> F;
fn isqrt(x: F) -> F;
fn abs(x: T) -> T;
fn sign(x: T) -> T;
fn floor(x: F) -> F;
fn trunc(x: F) -> F;
fn round(x: F) -> F;
fn roundEven(x: F) -> F;
fn ceil(x: F) -> F;
fn fract(x: F) -> F;
fn mod(x: T,y: T) -> T;
fn modf(x: T,y: T) -> T;
fn min(x: T,y: T) -> T;
fn max(x: T,y: T) -> T;
fn clamp(x: T,min: T,max: T) -> T;
fn mix(x: T,y: T,a: T) -> T;
fn step(edge: T,x: T) -> T;
fn smoothstep(edge0: T,edge1: T,x: T) -> T;
fn isnan(x: F) -> bool;
fn isinf(x: F) -> bool;
fn floatBitsToInt(x: F) -> I;
fn floatBitsToUint(x: F) -> U;
fn intBitsToFloat(x: I) -> F;
fn uintBitsToFloat(x: U) -> F;
fn fma(a: F,b: F,c: F) -> F;
fn frexp(x: F,exp: I) -> F;
fn ldexp(x: F,exp: I) -> F;
fn packUnorm2x16(v: Vec2<T>) -> U;
fn packSnorm2x16(v: Vec2<T>) -> U;
fn packUnorm4x8(v: Vec4<T>) -> U;
fn packSnorm4x8(v: Vec4<T>) -> U;
fn unpackUnorm2x16(p: U) -> Vec2<T>;
fn unpackSnorm2x16(p: U) -> Vec2<T>;
fn unpackUnorm4x8(p: U) -> Vec4<T>;
fn unpackSnorm4x8(p: U) -> Vec4<T>;
fn packHalf2x16(v: Vec2<T>) -> U;
fn unpackHalf2x16(p: U) -> Vec2<T>;
fn packDouble2x32(v: Vec2<U>) -> f64;
fn unpackDouble2x32(p: f64) -> Vec2<U>;
fn length(x: F) -> F;
fn distance(p0: F,p1: F) -> F;
fn dot(x: F,y: F) -> F;
fn cross(x: Vec3<F>,y: Vec3<F>) -> Vec3<F>;
fn normalize(x: F) -> F;
fn faceforward(n: F,i: F,nref: F) -> F;
fn reflect(i: F,n: F) -> F;
fn refract(i: F,n: F,e: F) -> F;
fn matrixCompMult(x: M,y: M) -> M;
fn outerProduct(c: VecA<F>,r: VecB<F>) -> MatAxB<F>;
fn transpose(m: MatAxB<F>) -> MatBxA<F>;
fn determinant(m: MatAxA<F>) -> F;
fn inverse(m: MatAxA<F>) -> MatAxA<F>;
fn lessThan(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn lessThanEqual(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn greaterThan(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn greaterThanEqual(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn equal(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn notEqual(x: VecA<T>,y: VecA<T>) -> VecA<bool>;
fn any(x: VecA<bool>) -> bool;
fn all(x: VecA<bool>) -> bool;
fn not(x: VecA<bool>) -> bool;
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
