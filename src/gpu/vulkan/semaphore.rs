use {
    crate::*,
    std::{
        ptr::null_mut,
        rc::Rc,
        mem::MaybeUninit,
    },
};

pub struct Semaphore {
    pub system: Rc<System>,
    pub(crate) vk_semaphore: sys::VkSemaphore,
}

impl Semaphore {
    pub fn new(system: &Rc<System>) -> Result<Semaphore,String> {
        let info = sys::VkSemaphoreCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
        };
        let mut vk_semaphore = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSemaphore(system.gpu_system.vk_device,&info,null_mut(),vk_semaphore.as_mut_ptr()) } {
            sys::VK_SUCCESS => Ok(Semaphore { system: Rc::clone(system),vk_semaphore: unsafe { vk_semaphore.assume_init() }, }),
            code => Err(format!("Unable to create semaphore ({})",code)),
        }
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { sys::vkDestroySemaphore(self.system.gpu_system.vk_device,self.vk_semaphore,null_mut()) };
    }
}
