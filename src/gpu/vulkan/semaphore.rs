// E - GPU (Vulkan) - Semaphore
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct Semaphore<'system,'gpu,'screen,'session> {
    pub session: &'session Session<'system,'gpu,'screen>,
    pub(crate) vk_semaphore: sys::VkSemaphore,
}

impl<'system,'gpu,'screen> Session<'system,'gpu,'screen> {

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
                println!("Unable to create semaphore (error {}).",code);
                return None;
            },
        }
        Some(Semaphore {
            session: &self,
            vk_semaphore: unsafe { vk_semaphore.assume_init() },
        })
    }
}

impl<'system,'gpu,'screen,'session> Drop for Semaphore<'system,'gpu,'screen,'session> {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.session.vk_device,self.vk_semaphore,null_mut()) };
    }
}
