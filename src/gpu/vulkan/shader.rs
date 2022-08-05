// E - GPU (Vulkan) - Shader
// Desmond Germans, 2020

use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Shader<'system> {
    pub system: &'system System,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl<'system> Drop for Shader<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.system.vk_device,self.vk_shader_module,null_mut()) };
    }
}
