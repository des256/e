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
