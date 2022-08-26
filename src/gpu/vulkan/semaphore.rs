use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct Semaphore<'system> {
    pub system: &'system System,
    pub(crate) vk_semaphore: sys::VkSemaphore,
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.system.gpu.vk_device,self.vk_semaphore,null_mut()) };
    }

    /*
    Cannot call vkDestroySemaphore on VkSemaphore 0x140000000014[] that is
    currently in use by a command buffer. The Vulkan spec states: All submitted
    batches that refer to semaphore must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroySemaphore-semaphore-01137)
    */
}
