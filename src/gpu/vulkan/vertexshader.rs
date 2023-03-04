use {
    super::*,
    crate::gpu,
    std::{
        rc::Rc,
        ptr::null_mut,
    },
};

/// Vertex shader.
#[derive(Debug)]
pub struct VertexShader {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl gpu::VertexShader for VertexShader {

}

impl Drop for VertexShader {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.gpu.vk_device,self.vk_shader_module,null_mut()) };
    }
}
