use {
    crate::*,
    ast::*,
};

pub trait Vertex where Self: Sized {
    fn ast() -> ast::Struct;
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

pub(crate) fn type_to_size(type_: &ast::Type) -> Result<usize,String> {
    match type_ {
        Type::Inferred | Type::Void | Type::Integer | Type::Float | Type::USize | Type::ISize => Err(format!("Vertex field cannot be {}",type_)),
        Type::Bool | Type::AnonTuple(_) | Type::Array(_,_) | Type::UnknownIdent(_) => Err(format!("TODO: Vertex field {}",type_)),
        Type::U8 | Type::I8 => Ok(1),
        Type::U16 | Type::I16 | Type::F16 => Ok(2),
        Type::U32 | Type::I32 | Type::F32 => Ok(4),
        Type::U64 | Type::I64 | Type::F64 => Ok(8),
        Type::Generic(ident,types) => {
            if types.len() == 1 {
                let vc = match ident.as_str() {
                    "Vec2" => Ok(2),
                    "Vec3" => Ok(3),
                    "Vec4" => Ok(4),
                    _ => Err(format!("Vertex field cannot be a {}",ident)),
                }?;
                match types[0] {
                    Type::Inferred | Type::Void | Type::Integer | Type::Float | Type::USize | Type::ISize | Type::Generic(_,_) => Err(format!("Vertex field cannot be Vec{}<{}>",vc,type_)),
                    Type::Bool | Type::AnonTuple(_) | Type::Array(_,_) | Type::UnknownIdent(_) => Err(format!("TODO: Vertex field Vec{}<{}>",vc,type_)),
                    Type::U8 | Type::I8 => match vc {
                        2 => Ok(2),
                        3 => Ok(3),
                        4 => Ok(4),
                        _ => Err("NOPE!".to_string()),
                    },
                    Type::U16 | Type::I16 | Type::F16 => match vc {
                        2 => Ok(4),
                        3 => Ok(6),
                        4 => Ok(8),
                        _ => Err("NOPE!".to_string()),
                    },
                    Type::U32 | Type::I32 | Type::F32 => match vc {
                        2 => Ok(8),
                        3 => Ok(12),
                        4 => Ok(16),
                        _ => Err("NOPE!".to_string()),
                    },
                    Type::U64 | Type::I64 | Type::F64 => match vc {
                        2 => Ok(16),
                        3 => Ok(24),
                        4 => Ok(32),
                        _ => Err("NOPE!".to_string()),
                    },
                }
            }
            else {
                Err(format!("Vertex field cannot be {}",type_))
            }
        },
    }
}
