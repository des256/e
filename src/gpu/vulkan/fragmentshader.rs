use {
    super::*,
    crate::gpu,
    std::{
        rc::Rc,
        ptr::null_mut,
    },
};

#[derive(Debug)]
pub struct FragmentShader {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_shader_module: sys::VkShaderModule,
}

impl gpu::FragmentShader for FragmentShader {

}

impl Drop for FragmentShader {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyShaderModule(self.gpu.vk_device,self.vk_shader_module,null_mut()) };
    }
}
