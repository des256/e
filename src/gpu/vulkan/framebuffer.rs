use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Framebuffer {
    pub owned: bool,
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_framebuffer: sys::VkFramebuffer,
}

impl Drop for Framebuffer {

    fn drop(&mut self) {
        if self.owned {
            unsafe { sys::vkDestroyFramebuffer(self.vk_device,self.vk_framebuffer,null_mut()) };
        }
    }
}
