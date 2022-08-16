pub enum FieldType {
    U8,
    I8,
    U8N,
    I8N,
    U16,
    I16,
    U16N,
    I16N,
    F16,
    U32,
    I32,
    F32,
    U64,
    I64,
    F64,
    U8XY,
    I8XY,
    U8NXY,
    I8NXY,
    U16XY,
    I16XY,
    U16NXY,
    I16NXY,
    F16XY,
    U32XY,
    I32XY,
    F32XY,
    U64XY,
    I64XY,
    F64XY,
    U8XYZ,
    I8XYZ,
    U8NXYZ,
    I8NXYZ,
    U16XYZ,
    I16XYZ,
    U16NXYZ,
    I16NXYZ,
    F16XYZ,
    U32XYZ,
    I32XYZ,
    F32XYZ,
    U64XYZ,
    I64XYZ,
    F64XYZ,
    U8XYZW,
    I8XYZW,
    U8NXYZW,
    I8NXYZW,
    U16XYZW,
    I16XYZW,
    U16NXYZW,
    I16NXYZW,
    F16XYZW,
    U32XYZW,
    I32XYZW,
    F32XYZW,
    U64XYZW,
    I64XYZW,
    F64XYZW,
    U8NRGBA,
    U16NRGBA,
    F16RGBA,
    F32RGBA,
    F64RGBA,
}

pub struct VertexAttribute {
    pub location: usize,
    pub binding: usize,
    pub field_type: FieldType,
    pub offset: usize,
}

pub trait VertexFormat where Self: Sized {
    fn stride() -> usize;
    fn attributes() -> usize;
    fn attribute(index: usize) -> VertexAttribute;
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

#[cfg(gpu="opengl")]
mod opengl;
#[cfg(gpu="opengl")]
pub use opengl::*;
