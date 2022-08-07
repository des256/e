use {
    crate::*,
    std::ptr::null_mut,
};

pub struct Semaphore {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_semaphore: sys::VkSemaphore,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.vk_device,self.vk_semaphore,null_mut()) };
    }
}
