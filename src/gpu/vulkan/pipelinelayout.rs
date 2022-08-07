use {
    crate::*,
    std::ptr::null_mut,
};

pub struct PipelineLayout {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_pipeline_layout: sys::VkPipelineLayout,
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
