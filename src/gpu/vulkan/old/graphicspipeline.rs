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
    std::{
        ptr::null_mut,
        rc::Rc,
    },
};

pub struct GraphicsPipeline {
    pub(crate) system: Rc<System>,
    pub(crate) vk_graphics_pipeline: sys::VkPipeline,
#[allow(dead_code)]    
    pub(crate) vertex_shader: Rc<VertexShader>,
#[allow(dead_code)]    
    pub(crate) fragment_shader: Rc<FragmentShader>,
#[allow(dead_code)]    
    pub(crate) pipeline_layout: Rc<PipelineLayout>,
}

impl Drop for GraphicsPipeline {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipeline(self.system.gpu.vk_device,self.vk_graphics_pipeline,null_mut()) };
    }

    /*
    Cannot call vkDestroyPipeline on VkPipeline 0x120000000012[] that is
    currently in use by a command buffer. The Vulkan spec states: All submitted
    commands that refer to pipeline must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroyPipeline-pipeline-00765)
    */
}
