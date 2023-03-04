use {
    super::*,
    crate::gpu,
    std::{
        ptr::null_mut,
        rc::Rc,
    },
};

#[derive(Debug)]
pub struct IndexBuffer {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_memory: sys::VkDeviceMemory,
}

impl gpu::IndexBuffer for IndexBuffer {
    
}

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroyBuffer(self.gpu.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.gpu.vk_device,self.vk_memory,null_mut());
        }
    }
}