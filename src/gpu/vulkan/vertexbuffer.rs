use {
    super::*,
    crate::gpu,
    std::{
        ptr::null_mut,
        rc::Rc,
    },
};

#[derive(Debug)]
pub struct VertexBuffer {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_device_memory: sys::VkDeviceMemory,
}

impl gpu::VertexBuffer for VertexBuffer {
    
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroyBuffer(self.gpu.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.gpu.vk_device,self.vk_device_memory,null_mut());
        }
    }
}