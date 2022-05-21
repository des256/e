// E - GPU (Vulkan) - Semaphore
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct Semaphore {
    pub session: Rc<Session>,
    pub(crate) vk_semaphore: VkSemaphore,
}

impl Session {

    pub fn create_semaphore(self: &Rc<Self>) -> Option<Rc<Semaphore>> {

        let info = VkSemaphoreCreateInfo {
            sType: VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
        };
        let mut vk_semaphore = MaybeUninit::uninit();
        match unsafe { vkCreateSemaphore(self.vk_device,&info,null_mut(),vk_semaphore.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create semaphore (error {}).",code);
                return None;
            },
        }
        Some(Rc::new(Semaphore {
            session: Rc::clone(self),
            vk_semaphore: unsafe { vk_semaphore.assume_init() },
        }))
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { vkDestroySemaphore(self.session.vk_device,self.vk_semaphore,null_mut()) };
    }
}
