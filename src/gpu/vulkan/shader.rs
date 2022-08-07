// E - GPU (Vulkan) - Shader
// Desmond Germans, 2020

use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Shader {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl Drop for Shader {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.vk_device,self.vk_shader_module,null_mut()) };
    }
}
