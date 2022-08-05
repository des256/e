use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Semaphore<'system> {
    pub system: &'system System,
    pub(crate) vk_semaphore: sys::VkSemaphore,
}

impl<'system> Drop for Semaphore<'system> {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.system.vk_device,self.vk_semaphore,null_mut()) };
    }
}
