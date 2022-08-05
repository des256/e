use {
    crate::*,
    std::ptr::null_mut,
};

pub struct RenderPass<'system> {
    pub system: &'system System,
    pub(crate) vk_renderpass: sys::VkRenderPass,
}

impl<'system> Drop for RenderPass<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyRenderPass(self.system.vk_device,self.vk_renderpass,null_mut()) };
    }
}
