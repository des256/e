use {
    crate::*,
    std::ptr::null_mut,
};

pub struct GraphicsPipeline<'system> {
    pub system: &'system System,
    pub(crate) vk_graphics_pipeline: sys::VkPipeline,
}

impl<'system> Drop for GraphicsPipeline<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipeline(self.system.vk_device,self.vk_graphics_pipeline,null_mut()) };
    }
}
