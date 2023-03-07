use {
    crate::*,
    std::rc::Rc,
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

#[derive(Debug,Clone)]
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

#[derive(Debug,Clone)]
pub enum PrimitiveRestart {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
pub enum DepthClamp {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
pub enum PrimitiveDiscard {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

#[derive(Debug,Clone)]
pub enum FrontFace {
    CounterClockwise,
    Clockwise,
}

#[derive(Debug,Clone)]
pub enum CullMode {
    None,
    Front(FrontFace),
    Back(FrontFace),
    FrontAndBack(FrontFace),
}

#[derive(Debug,Clone)]
pub enum DepthBias {
    Disabled,
    Enabled(f32,f32,f32),
}

#[derive(Debug,Clone)]
pub enum SampleShading {
    Disabled,
    Enabled(f32),
}

#[derive(Debug,Clone)]
pub enum AlphaToCoverage {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
pub enum AlphaToOne {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
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

#[derive(Debug,Clone)]
pub enum DepthBounds {
    Disabled,
    Enabled(f32,f32),
}

#[derive(Debug,Clone)]
pub enum DepthTest {
    Disabled,
    Enabled(CompareOp,DepthBounds),
}

#[derive(Debug,Clone)]
pub enum DepthWrite {
    Disabled,
    Enabled,
}

#[derive(Debug,Clone)]
pub enum StencilTest {
    Disabled,
    Enabled(
        (StencilOp,StencilOp,StencilOp,CompareOp,u32,u32,u32),
        (StencilOp,StencilOp,StencilOp,CompareOp,u32,u32,u32),
    ),
}

#[derive(Debug,Clone)]
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

#[derive(Debug,Clone)]
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

#[derive(Debug,Clone)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

#[derive(Debug,Clone)]
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

#[derive(Debug,Clone)]
pub enum Blend {
    Disabled,
    Enabled((BlendOp,BlendFactor,BlendFactor),(BlendOp,BlendFactor,BlendFactor)),
}

#[derive(Debug,Clone,Copy)]
pub enum TextureFilter {
    Nearest,
    Linear,
}

#[derive(Debug,Clone,Copy)]
pub enum TextureWrap {
    Black,
    Edge,
    Repeat,
    Mirror,
}

#[derive(Debug,Clone,Copy)]
pub enum BlendMode {
    _Replace,
    _Over,
}

pub trait GraphicsPipeline {

}

pub trait ComputePipeline {

}

pub trait PipelineLayout {

}

pub trait VertexBuffer {

}

pub trait IndexBuffer {

}

pub trait Framebuffer {

}

pub trait VertexShader {

}

pub trait FragmentShader {

}

pub trait CommandBuffer {
    type Surface : Surface;
    type GraphicsPipeline : GraphicsPipeline;
    type ComputePipeline : ComputePipeline;
    type VertexBuffer : VertexBuffer;
    type IndexBuffer : IndexBuffer;
    fn begin(&self) -> Result<(),String>;
    fn end(&self) -> bool;
    fn begin_render_pass(&self,surface: &Self::Surface,index: usize,r: Rect<i32>);
    fn end_render_pass(&self);
    fn bind_graphics_pipeline(&self,pipeline: &Rc<Self::GraphicsPipeline>);
    fn bind_compute_pipeline(&self,pipeline: &Rc<Self::ComputePipeline>);
    fn bind_vertex_buffer(&self,vertex_buffer: &Rc<Self::VertexBuffer>);
    fn bind_index_buffer(&self,index_buffer: &Rc<Self::IndexBuffer>);
    fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize);
    fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: usize,first_instance: usize);
    fn set_viewport(&self,r: Rect<i32>,min_depth: f32,max_depth: f32);
    fn set_scissor(&self,r: Rect<i32>);
}

pub trait Surface {
    fn set_rect(&mut self,r: Rect<i32>) -> Result<(),String>;
    fn get_swapchain_count(&self) -> usize;
    fn acquire(&self) -> Result<usize,String>;
    fn present(&self,index: usize) -> Result<(),String>;
}

pub trait Gpu {
    type CommandBuffer : CommandBuffer;
    type VertexShader : VertexShader;
    type FragmentShader : FragmentShader;
    type PipelineLayout : PipelineLayout;
    fn open(system: &Rc<System>) -> Result<Rc<Self>,String>;
    fn create_surface(self: &Rc<Self>,window: &Rc<Window>,r: Rect<i32>) -> Result<<Self::CommandBuffer as CommandBuffer>::Surface,String>;
    fn create_command_buffer(self: &Rc<Self>) -> Result<Self::CommandBuffer,String>;
    fn submit_command_buffer(&self,command_buffer: &Self::CommandBuffer) -> Result<(),String>;
    fn create_vertex_shader(self: &Rc<Self>,ast: &ast::Module) -> Result<Self::VertexShader,String>;
    fn create_fragment_shader(self: &Rc<Self>,ast: &ast::Module) -> Result<Self::FragmentShader,String>;
    fn create_graphics_pipeline<T: Vertex>(self: &Rc<Self>,
        surface: &<Self::CommandBuffer as CommandBuffer>::Surface,
        pipeline_layout: &Rc<Self::PipelineLayout>,
        vertex_shader: &Rc<Self::VertexShader>,
        fragment_shader: &Rc<Self::FragmentShader>,
        topology: gpu::PrimitiveTopology,
        restart: gpu::PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: gpu::DepthClamp,
        primitive_discard: gpu::PrimitiveDiscard,
        polygon_mode: gpu::PolygonMode,
        cull_mode: gpu::CullMode,
        depth_bias: gpu::DepthBias,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: gpu::SampleShading,
        alpha_to_coverage: gpu::AlphaToCoverage,
        alpha_to_one: gpu::AlphaToOne,
        depth_test: gpu::DepthTest,
        depth_write_mask: bool,
        stencil_test: gpu::StencilTest,
        logic_op: gpu::LogicOp,
        blend: gpu::Blend,
        write_mask: (bool,bool,bool,bool),
        blend_constant: Vec4<f32>,
    ) -> Result<<Self::CommandBuffer as CommandBuffer>::GraphicsPipeline,String>;
    fn create_vertex_buffer<T: Vertex>(self: &Rc<Self>,vertices: &Vec<T>) -> Result<<Self::CommandBuffer as CommandBuffer>::VertexBuffer,String>;
    fn create_index_buffer<T>(self: &Rc<Self>,indices: &Vec<T>) -> Result<<Self::CommandBuffer as CommandBuffer>::IndexBuffer,String>;
    fn create_pipeline_layout(self: &Rc<Self>) -> Result<Self::PipelineLayout,String>;
}

// anything past this level requires qualifiers vulkan::, opengl::, etc.
#[cfg(vulkan)]
pub mod vulkan;

#[cfg(opengl)]
pub mod opengl;

#[cfg(gles)]
pub mod gles;

#[cfg(metal)]
pub mod metal;

#[cfg(directx)]
pub mod directx;

#[cfg(webgl)]
pub mod webgl;

#[cfg(webgpu)]
pub mod webgpu;

pub(crate) fn type_to_size(type_: &ast::Type) -> Result<usize,String> {
    match type_ {
        ast::Type::Bool => Err("TODO: bool vertex field".to_string()),
        ast::Type::U8 => Ok(1),
        ast::Type::I8 => Ok(1),
        ast::Type::U16 => Ok(2),
        ast::Type::I16 => Ok(2),
        ast::Type::U32 => Ok(4),
        ast::Type::I32 => Ok(4),
        ast::Type::U64 => Ok(8),
        ast::Type::I64 => Ok(8),
        ast::Type::F16 => Ok(2),
        ast::Type::F32 => Ok(4),
        ast::Type::F64 => Ok(8),
        ast::Type::Struct(ident) => {
            let parts: Vec<&str> = ident.split(&['<','>']).collect();
            if parts.len() == 2 {
                let ss = match parts[1] {
                    "bool" => { return Err(format!("TODO: {} vertex field",ident)); },
                    "u8" | "i8" => 1,
                    "u16" | "i16" | "f16" => 2,
                    "u32" | "i32" | "f32" => 4,
                    "u64" | "i64" | "f64" => 8,
                    _ => { return Err(format!("Vertex field cannot be {}",type_)) },
                };
                match parts[0] {
                    "Vec2" => Ok(2 * ss),
                    "Vec3" => Ok(3 * ss),
                    "Vec4" => Ok(4 * ss),
                    _ => Err(format!("Vertex field cannot be {}",type_)),
                }
            }
            else {
                Err(format!("Vertex field cannot be {}",type_))
            }
        },
        _ => Err(format!("Vertex field cannot be {}",type_)),
    }
}
