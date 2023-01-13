use std::{
    rc::Rc,
    collections::HashMap,
};

pub struct Field {
    ident: String,
    type_: Type,
}

pub struct Struct {
    ident: String,
    fields: Vec<Field>,
}

pub enum Type {
    Void,
    Float,Double,
    Int,Uint,
    Bool,
    Vec2,Vec3,Vec4,
    DVec2,DVec3,DVec4,
    BVec2,BVec3,BVec4,
    IVec2,IVec3,IVec4,
    UVec2,UVec3,UVec4,
    Mat2,Mat2x3,Mat2x4,
    Mat3x2,Mat3,Mat3x4,
    Mat4x2,Mat4x3,Mat4,
    DMat2,DMat2x3,DMat2x4,
    DMat3x2,DMat3,DMat3x4,
    DMat4x2,DMat4x3,DMat4,
    AtomicUint,
    Sampler1D,Texture1D,Image1D,Sampler1DShadow,
    Sampler1DArray,Texture1DArray,Image1DArray,Sampler1DArrayShadow,
    Sampler2D,Texture2D,Image2D,Sampler2DShadow,
    Sampler2DArray,Image2DArray,Sampler2DArrayShadow,
    Sampler2DMS,Texture2DMS,Image2DMS,
    Sampler2DMSArray,Texture2DMSArray,Image2DMSArray,
    Sampler2DRect,Texture2DRect,Image2DRect,Sampler2DRectShadow,
    Sampler3D,Texture3D,Image3D,
    SamplerCube,TextureCube,ImageCube,SamplerCubeShadow,
    SamplerCubeArray,TextureCubeArray,ImageCubeArray,SamplerCubeArrayShadow,
    SamplerBuffer,TextureBuffer,ImageBuffer,
    SubPassInput,SubPassInputMS,
    ISampler1D,ITexture1D,IImage1D,
    ISampler1DArray,ITexture1DArray,IImage1DArray,
    ISampler2D,ITexture2D,IImage2D,
    ISampler2DArray,ITexture2DArray,IImage2DArray,
    ISampler2DMS,ITexture2DMS,IImage2DMS,
    ISampler2DMSArray,ITexture2DMSArray,IImage2DMSArray,
    ISampler2DRect,ITexture2DRect,IImage2DRect,
    ISampler3D,ITexture3D,IImage3D,
    ISamplerCube,ITextureCube,IImageCube,
    ISamplerCubeArray,ITextureCubeArray,IImageCubeArray,
    ISamplerBuffer,ITextureBuffer,IImageBuffer,
    ISubPassInput,ISubPassInputMS,
    USampler1D,UTexture1D,UImage1D,
    USampler1DArray,UTexture1DArray,UImage1DArray,
    USampler2D,UTexture2D,UImage2D,
    USampler2DArray,UTexture2DArray,UImage2DArray,
    USampler2DMS,UTexture2DMS,UImage2DMS,
    USampler2DMSArray,UTexture2DMSArray,UImage2DMSArray,
    USampler2DRect,UTexture2DRect,UImage2DRect,
    USampler3D,UTexture3D,UImage3D,
    USamplerCube,UTextureCube,UImageCube,
    USamplerCubeArray,UTextureCubeArray,UImageCubeArray,
    USamplerBuffer,UTextureBuffer,UImageBuffer,
    USubPassInput,USubPassInputMS,
    Struct(Rc<Struct>),
    Array(Box<Type>,usize),
}

pub enum StorageQual {
    Const,
    In,
    Out,
    Attribute,
    Uniform,
    Varying,
    Buffer,
    Shared,
}

pub enum LayoutQual {
    Shared,
    Packed,
    Std140,
    Std430,
    RowMajor,
    ColumnMajor,
    Binding(usize),
    Offset(usize),
    Align(usize),
    Location(usize),
    Component(usize),
    Index(usize),
    Triangles,
    Quads,
    IsoLines,
    EqualSpacing,
    FracEvenSpacing,
    FracOddSpacing,
    CW,
    CCW,
    PointMode,
    Points,
    Lines,
    LinesAdjacency,
    Triangles,
    TrianglesAdjacency,
    OriginUpperLeft,
    PixelCenterInteger,
    EarlyFragmentTests,
    LocalSizeX(f64),
    LocalSizeY(f64),
    LocalSizeZ(f64),
    XfbBuffer(usize),
    XfbStride(usize),
    XfbOffset(usize),
    Vertices(usize),
    LineStrip,
    TriangleStrip,
    MaxVertices(usize),
    Stream(usize),
    DepthAny,
    DepthGreater,
    DepthLess,
    DepthUnchanged,
}

pub enum InterpQual {

}

pub enum FormatQual {
    RGBA32F,
    RGBA16F,
    RG32F,
    RG16F,
    R11G11B10F,
    R32F,
    R16F,
    RGBA16,
    RGB10A2,
    RGBA8,
    RG16,
    RG8,
    R16,
    R8,
    RGBA16SN,
    RGBA8SN,
    RG16SN,
    RG8SN,
    R16SN,
    R8SN,
    RGBA32I,
    RGBA16I,
    RGBA8I,
    RG32I,
    RG16I,
    RG8I,
    R32I,
    R16I,
    R8I,
    RGBA32UI,
    RGBA16UI,
    RGB10A2UI,
    RGBA8UI,
    RG32UI,
    RG16UI,
    RG8UI,
    R32UI,
    R16UI,
    R8UI,
}

enum InterpQual {
    Smooth,
    Flat,
    NoPerspective,
}

enum ParamQual {
    Const,
    In,
    Out,
    InOut,
}

enum MemQual {
    Coherent,
    Volatile,
    Restrict,
    ReadOnly,
    WriteOnly,
}

pub enum Qual {
    In,
    Out,
    Attribute,
    Uniform,
    Varying,
    Buffer,
    Shared,
    Invariant,
    Centroid,
    Sample,
    Patch,
    Const,
    Layout(usize),  // layout(location = ...)
}

pub struct Decl {
    quals: Vec<Qual>,
    type_: Type,
    ident: String,
    expr: Option<Expr>,
}

pub struct DeclBlock {
    quals: Vec<Qual>,
    ident: String,
    decls: Vec<Decl>,
}

enum Expr {
    Struct(Rc<Struct>,Vec<Expr>),
    Array(Type,Vec<Expr>),

}

enum Stat {
    Compound(Vec<Stat>),
    Decl(Decl),
    Expr(Expr),
    If(Expr,Stat,Option<Stat>),
    Switch(Expr,Vec<Stat>),
    For(Decl,Expr,Stat,Vec<Stat>),
    While(Expr,Vec<Stat>),
    Do(Vec<Stat>,Expr),
    Discard,
    Return(Option<Expr>),
    Break,
    Continue,
    Case(Expr),
    Default,
}

struct Function {
    ident: String,
    args: Vec<Decl>,
    type_: Type,
}

pub struct Module {
    decls: HashMap<String,Decl>,
    declblocks: HashMap<String,DeclBlock>,
    functions: HashMap<String,Function>,
}

/*
STDLIB:

compute:
    in uvec3 gl_NumWorkGroups;
    const uvec3 gl_WorkGroupSize;
    in uvec3 gl_WorkGroupID;
    in uvec3 gl_LocalInvocationIndex;
    in uvec3 gl_GlobalInvocationID;
    in uint gl_LocalInvocationIndex;

vertex:
    in int gl_VertexID;
    in int gl_InstanceID;
    in int gl_VertexIndex;
    in int gl_InstanceIndex;
    in int gl_DrawID;
    in int gl_BaseVertex;
    in int gl_BaseInstance;
    out gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    };

geometry:
    in gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    };
    in int gl_PrimitiveIDIn;
    in int gl_InvocationID;
    out gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    };
    out int gl_PrimitiveID;
    out int gl_Layer;
    out int gl_ViewportIndex;

tesselation:
    in gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    };
    in int gl_PatchVerticesIn;
    in int gl_PrimitiveID;
    in int gl_InvocationID;
    out gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    } gl_out[];
    patch out float gl_TessLevelOuter[4];
    patch out float gl_TessLevelInner[2];

tesselation evaluation:
    in gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    } gl_in[gl_MaxPatchVertices];
    in int gl_PatchVerticesIn;
    in int gl_PrimitiveID;
    in vec3 gl_TessCoord;
    patch in float gl_TessLevelOuter[4];
    patch in float gl_TessLevelInner[2];
    out gl_PerVertex {
        vec4 gl_Position;
        float gl_PointSize;
        float gl_ClipDistance[];
        float gl_CullDistance[];
    };

fragment:
    in vec4 gl_FragCoord;
    in bool gl_FrontFacing;
    in float gl_ClipDistance[];
    in float gl_CullDistance[];
    in vec2 gl_PointCoord;
    in int gl_PrimitiveID;
    in int gl_SampleID;
    in vec2 gl_SamplePosition;
    in int gl_SampleMaskIn[];
    in int gl_Layer;
    in int gl_ViewportIndex;
    in bool gl_HelperInvocation;
    out float gl_FragDepth;
    out int gl_SampleMask[];
 */