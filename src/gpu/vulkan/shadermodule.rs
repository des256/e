use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct ShaderModule<'system> {
    pub system: &'system System,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl System {

    /// Create a shader.
    pub fn create_shader_module(&self,code: &[u8]) -> Option<ShaderModule> {

#[cfg(gpu="vulkan")]
        {
            let create_info = sys::VkShaderModuleCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                codeSize: code.len() as u64,
                pCode: code.as_ptr() as *const u32,
            };

            let mut vk_shader_module = MaybeUninit::uninit();
            match unsafe { sys::vkCreateShaderModule(self.vk_device,&create_info,null_mut(),vk_shader_module.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create shader (error {})",code);
                    return None;
                },
            }
            let vk_shader_module = unsafe { vk_shader_module.assume_init() };

            Some(ShaderModule {
                system: &self,
                vk_shader_module: vk_shader_module,
            })
        }

#[cfg(not(gpu="vulkan"))]
        None
    }    
}

impl<'system> Drop for ShaderModule<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.system.vk_device,self.vk_shader_module,null_mut()) };
    }
}
