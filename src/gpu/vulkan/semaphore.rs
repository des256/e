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

impl System {

    /// Create a semaphore.
    pub fn create_semaphore(&self) -> Option<Semaphore> {

        let info = sys::VkSemaphoreCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
        };
        let mut vk_semaphore = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSemaphore(self.vk_device,&info,null_mut(),vk_semaphore.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create semaphore (error {})",code);
                return None;
            },
        }
        Some(Semaphore {
            system: &self,
            vk_semaphore: unsafe { vk_semaphore.assume_init() },
        })
    }
}

impl<'system> Drop for Semaphore<'system> {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.system.vk_device,self.vk_semaphore,null_mut()) };
    }
}
