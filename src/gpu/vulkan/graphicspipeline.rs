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
    super::*,
    crate::gpu,
    std::{
        ptr::null_mut,
        rc::Rc,
    },
};


#[derive(Debug)]
pub struct GraphicsPipeline {
    pub(crate) gpu: Rc<Gpu>,
    pub(crate) vk_pipeline: sys::VkPipeline,
    pub(crate) vertex_shader: Rc<VertexShader>,
    pub(crate) fragment_shader: Rc<FragmentShader>,
    pub(crate) pipeline_layout: Rc<PipelineLayout>,
}

impl gpu::GraphicsPipeline for GraphicsPipeline {

}

impl Drop for GraphicsPipeline {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipeline(self.gpu.vk_device,self.vk_pipeline,null_mut()) };
    }
}
