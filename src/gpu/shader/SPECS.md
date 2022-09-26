# Output Languages

## GLSL

## SPIR-V

## HLSL

## MSL

### Types

#### Scalar

bool
char, int8_t
unsigned char, uchar, uint8_t
short, int16_t
unsigned short, ushort, uint16_t
int, int32_t
unsigned int, uint, uint32_t
long, int64_t
unsigned long, uint64_t
half
float
size_t
ptrdiff_t
void

#### Vector

bool2, bool3, bool4
char2, char3, char4
packed_char2, packed_char3, packed_char4
uchar2, uchar3, uchar4
packed_uchar2, packed_uchar3, packed_uchar4
short2, short3, short4
packed_short2, packed_short3, packed_short4
ushort2, ushort3, ushort4
packed_ushort2, packed_ushort3, packed_ushort4
int2, int3, int4
packed_int2, packed_int3, packed_int4
uint2, uint3, uint4
packed_uint2, packed_uint3, packed_uint4
long2, long3, long4
packed_long2, packed_long3, packed_long4
ulong2, ulong3, ulong4
packed_ulong2, packed_ulong3, packed_ulong4
half2, half3, half4
packed_half2, packed_half3, packed_half4
float2, float3, float4
packed_float2, packed_float3, packed_float4

constructors exist for each of these with all composition permutations

#### Matrix

half2x2, half2x3, half2x4
half3x2, half3x3, half3x4
half4x2, half4x3, half4x4
float2x2, float2x3, float2x4
float3x2, float3x3, float3x4
float4x2, float4x3, float4x4

simdgroup_half8x8
simdgroup_float8x8

#### Atomic Types

atomic_int
atomic_uint
atomic_bool
atomic_ulong
atomic_float

#### Pixel Data Types

r8unorm
r8snorm
r16unorm
r16snorm
rg8unorm
rg8snorm
rg16unorm
rg16snorm
rgba8unorm
srgba8unorm
rgba8snorm
rgba16unorm
rgba16snorm
rgb10a2
rg11b10f
rgb9e5

#### Textures

also indicating intended access here

texture1d
texture1d_array
texture2d
texture2d_array
texture3d
texturecube
texturecube_array
texture2d_ms
texture2d_ms_array
depth2d
depth2d_array
depthcube
depthcube_array
depth2d_ms
depth2d_ms_array

#### Texture Buffer

texture_buffer

#### Samplers

describe how a texture can be accessed with:

- coord::normalized/pixel
- address::repeat/mirrored_repeat/clamp_to_edge/clamp_to_zero/clamp_to_border
- or separately with s_address, t_address and r_address
- border_color::transparent_black/opaque_black/opaque_white
- filter::nearest/linear
- or separately with mag_filter and min_filter
- mip_filter::nearest/linear
- compare_func::never/less/less_equal/greater/greater_equal/equal/not_equal/always

#### Aggregate Types

array
