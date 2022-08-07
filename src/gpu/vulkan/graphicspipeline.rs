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
    std::ptr::null_mut,
};

pub struct GraphicsPipeline {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_graphics_pipeline: sys::VkPipeline,
}

impl Drop for GraphicsPipeline {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipeline(self.vk_device,self.vk_graphics_pipeline,null_mut()) };
    }
}
