use {
    crate::*,
    std::{
        rc::Rc,
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct FragmentShader {
    pub system: Rc<System>,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl Drop for FragmentShader {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.system.gpu.vk_device,self.vk_shader_module,null_mut()) };
    }
}