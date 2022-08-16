use {
    crate::*,
};

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

mod commandcontext;
pub use commandcontext::*;

mod commandbuffer;
pub use commandbuffer::*;

mod pipelinelayout;
pub use pipelinelayout::*;

mod shadermodule;
pub use shadermodule::*;

mod graphicspipeline;
pub use graphicspipeline::*;

mod semaphore;
pub use semaphore::*;

mod vertexbuffer;
pub use vertexbuffer::*;

mod indexbuffer;
pub use indexbuffer::*;