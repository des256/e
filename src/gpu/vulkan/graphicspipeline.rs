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
        mem::MaybeUninit,
    },
};

fn base_type_format(base_type: &sr::BaseType) -> sys::VkFormat {
    match base_type {
        sr::BaseType::U8 => sys::VK_FORMAT_R8_UINT,
        sr::BaseType::U16 => sys::VK_FORMAT_R16_UINT,
        sr::BaseType::U32 => sys::VK_FORMAT_R32_UINT,
        sr::BaseType::U64 => sys::VK_FORMAT_R64_UINT,
        sr::BaseType::I8 => sys::VK_FORMAT_R8_SINT,
        sr::BaseType::I16 => sys::VK_FORMAT_R16_SINT,
        sr::BaseType::I32 => sys::VK_FORMAT_R32_SINT,
        sr::BaseType::I64 => sys::VK_FORMAT_R64_SINT,
        sr::BaseType::F16 => sys::VK_FORMAT_R16_SFLOAT,
        sr::BaseType::F32 => sys::VK_FORMAT_R32_SFLOAT,
        sr::BaseType::F64 => sys::VK_FORMAT_R64_SFLOAT,
        sr::BaseType::Vec2U8 => sys::VK_FORMAT_R8G8_UINT,
        sr::BaseType::Vec2U16 => sys::VK_FORMAT_R16G16_UINT,
        sr::BaseType::Vec2U32 => sys::VK_FORMAT_R32G32_UINT,
        sr::BaseType::Vec2U64 => sys::VK_FORMAT_R64G64_UINT,
        sr::BaseType::Vec2I8 => sys::VK_FORMAT_R8G8_SINT,
        sr::BaseType::Vec2I16 => sys::VK_FORMAT_R16G16_SINT,
        sr::BaseType::Vec2I32 => sys::VK_FORMAT_R32G32_SINT,
        sr::BaseType::Vec2I64 => sys::VK_FORMAT_R64G64_SINT,
        sr::BaseType::Vec2F16 => sys::VK_FORMAT_R16G16_SFLOAT,
        sr::BaseType::Vec2F32 => sys::VK_FORMAT_R32G32_SFLOAT,
        sr::BaseType::Vec2F64 => sys::VK_FORMAT_R64G64_SFLOAT,
        sr::BaseType::Vec3U8 => sys::VK_FORMAT_R8G8B8_UINT,
        sr::BaseType::Vec3U16 => sys::VK_FORMAT_R16G16B16_UINT,
        sr::BaseType::Vec3U32 => sys::VK_FORMAT_R32G32B32_UINT,
        sr::BaseType::Vec3U64 => sys::VK_FORMAT_R64G64B64_UINT,
        sr::BaseType::Vec3I8 => sys::VK_FORMAT_R8G8B8_SINT,
        sr::BaseType::Vec3I16 => sys::VK_FORMAT_R16G16B16_SINT,
        sr::BaseType::Vec3I32 => sys::VK_FORMAT_R32G32B32_SINT,
        sr::BaseType::Vec3I64 => sys::VK_FORMAT_R64G64B64_SINT,
        sr::BaseType::Vec3F16 => sys::VK_FORMAT_R16G16B16_SFLOAT,
        sr::BaseType::Vec3F32 => sys::VK_FORMAT_R32G32B32_SFLOAT,
        sr::BaseType::Vec3F64 => sys::VK_FORMAT_R64G64B64_SFLOAT,
        sr::BaseType::Vec4U8 => sys::VK_FORMAT_R8G8B8A8_UINT,
        sr::BaseType::Vec4U16 => sys::VK_FORMAT_R16G16B16A16_UINT,
        sr::BaseType::Vec4U32 => sys::VK_FORMAT_R32G32B32A32_UINT,
        sr::BaseType::Vec4U64 => sys::VK_FORMAT_R64G64B64A64_UINT,
        sr::BaseType::Vec4I8 => sys::VK_FORMAT_R8G8B8A8_SINT,
        sr::BaseType::Vec4I16 => sys::VK_FORMAT_R16G16B16A16_SINT,
        sr::BaseType::Vec4I32 => sys::VK_FORMAT_R32G32B32A32_SINT,
        sr::BaseType::Vec4I64 => sys::VK_FORMAT_R64G64B64A64_SINT,
        sr::BaseType::Vec4F16 => sys::VK_FORMAT_R16G16B16A16_SFLOAT,
        sr::BaseType::Vec4F32 => sys::VK_FORMAT_R32G32B32A32_SFLOAT,
        sr::BaseType::Vec4F64 => sys::VK_FORMAT_R64G64B64A64_SFLOAT,
        sr::BaseType::ColorU8 => sys::VK_FORMAT_R8G8B8A8_UNORM,
        sr::BaseType::ColorU16 => sys::VK_FORMAT_R16G16B16A16_UNORM,
        sr::BaseType::ColorF16 => sys::VK_FORMAT_R16G16B16A16_SFLOAT,
        sr::BaseType::ColorF32 => sys::VK_FORMAT_R32G32B32A32_SFLOAT,
        sr::BaseType::ColorF64 => sys::VK_FORMAT_R64G64B64A64_SFLOAT,
    }
}

pub struct GraphicsPipeline<'system> {
    pub(crate) system: &'system System,
    pub(crate) vk_graphics_pipeline: sys::VkPipeline,
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
