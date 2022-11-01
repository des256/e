use std::{
    rc::Rc,
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

pub struct Module {
    decls: HashMap<String,Decl>,
    declblocks: HashMap<String,DeclBlock>,
    functions: HashMap<String,Function>,
}
