// Graphics Pipeline
// shader stages: vertex, tesselation control, tesselation evaluation, geometry, fragment, compute stages
// vertex input: from the vertex buffer memory, which data of what type describes which attributes to the vertex
// input assembly: primitive topology and restart enable
// tesselation: patch control points
// rasterization: depth clamp enable, rasterizer discard enable, polygon mode, cull mode, front face, depth bias enable, depth bias constant, clamp, slope factor, line width
// multisample: samples, sample shading enable, min sample shading, sample mask, alpha to coverage, alpha to one
// depth/stencil: depth test enable, depth write enable, depth compare op, depth bounds test enable, stencil test enable, front, back, min depth bounds, max depth bounds
// color blend: logic op enable, logic op, [blend enable, src/dst color/alpha factors, color/alpha blend op, color write mask] for each attachment

use {
    crate::*,
    std::rc::Rc,
};

pub struct GraphicsPipeline {
    pub(crate) system: Rc<System>,
    pub(crate) pipeline_layout: Rc<PipelineLayout>,
    pub(crate) shader_program: sys::GLuint,
    pub(crate) topology: sys::GLenum,
    pub(crate) restart: PrimitiveRestart,
    pub(crate) patch_control_points: usize,
    pub(crate) depth_clamp: bool,
    pub(crate) primitive_discard: bool,
    pub(crate) polygon_mode: sys::GLenum,
    pub(crate) cull_mode: Option<(sys::GLenum,sys::GLenum)>,
    pub(crate) polygon_offset: Option<(f32,f32)>,
    pub(crate) line_width: f32,
    pub(crate) rasterization_samples: usize,
    pub(crate) sample_shading: SampleShading,
    pub(crate) sample_alpha_to_coverage: bool,
    pub(crate) sample_alpha_to_one: bool,
    pub(crate) depth_test: Option<sys::GLenum>,
    pub(crate) depth_mask: u8,
    pub(crate) stencil_test: Option<(u32,sys::GLenum,i32,u32,sys::GLenum,sys::GLenum,sys::GLenum)>,
    pub(crate) logic_op: Option<sys::GLenum>,
    pub(crate) blend: Option<(sys::GLenum,sys::GLenum,sys::GLenum,sys::GLenum,sys::GLenum)>,
    pub(crate) color_mask: (sys::GLboolean,sys::GLboolean,sys::GLboolean,sys::GLboolean),
    pub(crate) blend_constant: Color<f32>,
}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { sys::glDeleteProgram(self.shader_program) };
    }
}