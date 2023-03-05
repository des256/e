use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::rc::Rc,
};


#[derive(Debug)]
pub struct GraphicsPipeline {
    pub gpu: Rc<Gpu>,
    pub pipeline_layout: Rc<PipelineLayout>,
    pub shader_program: sys::GLuint,
    pub topology: sys::GLenum,
    pub restart: gpu::PrimitiveRestart,
    pub patch_control_points: usize,
    pub depth_clamp: bool,
    pub rasterizer_discard: bool,
    pub polygon_mode: sys::GLenum,
    pub cull_mode: Option<(sys::GLenum,sys::GLenum)>,
    pub polygon_offset: Option<(f32,f32)>,
    pub line_width: f32,
    pub rasterization_samples: usize,
    pub sample_shading: gpu::SampleShading,
    pub sample_alpha_to_coverage: bool,
    pub sample_alpha_to_one: bool,
    pub depth_test: Option<sys::GLenum>,
    pub depth_mask: u8,
    pub stencil_test: Option<(u32,sys::GLenum,i32,u32,sys::GLenum,sys::GLenum,sys::GLenum)>,
    pub logic_op: Option<sys::GLenum>,
    pub blend: Option<(sys::GLenum,sys::GLenum,sys::GLenum,sys::GLenum,sys::GLenum,sys::GLenum)>,
    pub color_mask: (sys::GLboolean,sys::GLboolean,sys::GLboolean,sys::GLboolean),
    pub blend_constant: Vec4<f32>,
}

impl gpu::GraphicsPipeline for GraphicsPipeline {

}

impl Drop for GraphicsPipeline {
    fn drop(&mut self) {
        unsafe { checkgl!(sys::glDeleteProgram(self.shader_program)) };
    }
}