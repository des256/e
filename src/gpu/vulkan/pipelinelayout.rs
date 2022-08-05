use {
    crate::*,
    std::ptr::null_mut,
};

pub struct PipelineLayout<'system> {
    pub system: &'system System,
    pub(crate) vk_pipeline_layout: sys::VkPipelineLayout,
}

impl<'system> Drop for PipelineLayout<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.system.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
