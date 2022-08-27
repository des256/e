use sr;

pub trait Vertex where Self: Sized {
    fn get_fields() -> Vec<(String,sr::BaseType)>;
}

pub trait Uniform where Self: Sized {

}

pub trait Varying where Self: Sized {

}

pub trait Sample where Self: Sized {

}

pub enum PrimitiveTopology {
    Points,
    Lines,
    LineStrip,
    Triangles,
    TriangleStrip,
    TriangleFan,
    LinesAdjacency,
    LineStripAdjacency,
    TrianglesAdjacency,
    TriangleStripAdjacency,
    Patches,
}

pub enum PrimitiveRestart {
    Disabled,
    Enabled,
}

pub enum DepthClamp {
    Disabled,
    Enabled,
}

pub enum PrimitiveDiscard {
    Disabled,
    Enabled,
}

pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

pub enum CullMode {
    None,
    Front(FrontFace),
    Back(FrontFace),
    FrontAndBack(FrontFace),
}

pub enum DepthBias {
    Disabled,
    Enabled(f32,f32,f32),
}

pub enum SampleShading {
    Disabled,
    Enabled(f32),
}

pub enum AlphaToCoverage {
    Disabled,
    Enabled,
}

pub enum AlphaToOne {
    Disabled,
    Enabled,
}

pub enum CompareOp {
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

pub enum DepthBounds {
    Disabled,
    Enabled(f32,f32),
}

pub enum DepthTest {
    Disabled,
    Enabled(CompareOp,DepthBounds),
}

pub enum DepthWrite {
    Disabled,
    Enabled,
}

pub enum StencilTest {
    Disabled,
    Enabled(
        (StencilOp,StencilOp,StencilOp,CompareOp,u32,u32,u32),
        (StencilOp,StencilOp,StencilOp,CompareOp,u32,u32,u32),
    ),
}

pub enum StencilOp {
    Keep,
    Zero,
    Replace,
    IncClamp,
    DecClamp,
    Invert,
    IncWrap,
    DecWrap,
}

pub enum LogicOp {
    Disabled,
    Clear,
    And,
    AndReverse,
    Copy,
    AndInverted,
    NoOp,
    Xor,
    Or,
    Nor,
    Equivalent,
    Invert,
    OrReverse,
    CopyInverted,
    OrInverted,
    Nand,
    Set,
}

pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
    Src1Color,
    OneMinusSrc1Color,
    Src1Alpha,
    OneMinusSrc1Alpha,
}

pub enum Blend {
    Disabled,
    Enabled((BlendOp,BlendFactor,BlendFactor),(BlendOp,BlendFactor,BlendFactor)),
}

#[derive(Clone,Copy)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Clone,Copy)]
pub enum TextureWrap {
    Black,
    Edge,
    Repeat,
    Mirror,
}

#[derive(Clone,Copy)]
pub enum BlendMode {
    _Replace,
    _Over,
}

#[cfg(gpu="vulkan")]
mod vulkan;
#[cfg(gpu="vulkan")]
pub use vulkan::*;
#[cfg(gpu="vulkan")]
mod spirv;
#[cfg(gpu="vulkan")]
pub use spirv::*;

#[cfg(gpu="opengl")]
mod opengl;
#[cfg(gpu="opengl")]
pub use opengl::*;
#[cfg(gpu="opengl")]
mod glsl;
#[cfg(gpu="opengl")]
pub use glsl::*;
