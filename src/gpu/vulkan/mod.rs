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

pub enum CullMode {
    None,
    Front,
    Back,
    FrontAndBack,
}

pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

pub enum DepthBias {
    Disabled,
    Enabled,
}

pub enum SampleShading {
    Disabled,
    Enabled,
}

pub enum AlphaToCoverage {
    Disabled,
    Enabled,
}

pub enum AlphaToOne {
    Disabled,
    Enabled,
}

pub enum DepthTest {
    Disabled,
    Enabled,
}

pub enum DepthWrite {
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
    Enabled,
}

pub enum StencilTest {
    Disabled,
    Enabled,
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

pub enum Blend {
    Disabled,
    Enabled,
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

pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

mod commandcontext;
pub use commandcontext::*;

mod commandbuffer;
pub use commandbuffer::*;

mod pipelinelayout;
pub use pipelinelayout::*;

mod shader;
pub use shader::*;

mod graphicspipeline;
pub use graphicspipeline::*;

mod semaphore;
pub use semaphore::*;

mod vertexbuffer;
pub use vertexbuffer::*;

mod indexbuffer;
pub use indexbuffer::*;