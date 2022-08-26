use {
    crate::*,
    std::{
        ptr::{
            null_mut,
            copy_nonoverlapping,
        },
        mem::MaybeUninit,
        ffi::c_void,
    },
};

pub struct VertexBuffer<'system> {
    pub system: &'system System,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_memory: sys::VkDeviceMemory,
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroyBuffer(self.system.gpu.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.system.gpu.vk_device,self.vk_memory,null_mut());
        }
    }

    /*
    Cannot free VkBuffer 0xd000000000d[] that is in use by a command buffer.
    The Vulkan spec states: All submitted commands that refer to buffer,
    either directly or via a VkBufferView, must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroyBuffer-buffer-00922)
    */

    /*
    Cannot call vkFreeMemory on VkDeviceMemory 0xe000000000e[] that is
    currently in use by a command buffer. The Vulkan spec states: All
    submitted commands that refer to memory (via images or buffers) must have
    completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkFreeMemory-memory-00677)
    */
}