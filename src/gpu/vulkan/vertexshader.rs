use {
    crate::*,
    std::{
        rc::Rc,
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

/// Vertex shader.
#[derive(Debug)]
pub struct VertexShader {
    pub system: Rc<System>,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl VertexShader {

    pub fn new(system: &Rc<System>,code: &[u8]) -> Result<VertexShader,String> {
        let create_info = sys::VkShaderModuleCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            codeSize: code.len() as u64,
            pCode: code.as_ptr() as *const u32,
        };
        let mut vk_shader_module = MaybeUninit::uninit();
        match unsafe { sys::vkCreateShaderModule(system.gpu_system.vk_device,&create_info,null_mut(),vk_shader_module.as_mut_ptr()) } {
            sys::VK_SUCCESS => Ok(VertexShader {
                system: Rc::clone(system),
                vk_shader_module: unsafe { vk_shader_module.assume_init() },
            }),
            code => Err(format!("Unable to create vertex shader ({})",vk_code_to_string(code))),
        }
    }
}

impl Drop for VertexShader {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.system.gpu_system.vk_device,self.vk_shader_module,null_mut()) };
    }
}
