use {
    crate::*,
};

pub struct VertexBuffer {
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_memory: sys::VkDeviceMemory,
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        // TODO: drop buffer and memory
    }
}