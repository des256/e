use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Framebuffer<'system> {
    pub system: &'system System,
    pub owned: bool,
    pub(crate) vk_framebuffer: sys::VkFramebuffer,
}

impl<'system> Framebuffer<'system> {
}

impl<'system> Drop for Framebuffer<'system> {

    fn drop(&mut self) {
        if self.owned {
            unsafe { sys::vkDestroyFramebuffer(self.system.vk_device,self.vk_framebuffer,null_mut()) };
        }
    }
}
