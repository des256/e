use {
    crate::*,
};

pub struct WindowVulkan {
    pub instance: VkInstance,
    pub physical_device: VkPhysicalDevice,
    pub device: VkDevice,
    pub queue: VkQueue,
    pub surface: VkSurface,
    pub render_pass: VkRenderPass,
    pub swapchain: VkSwapchain,
    pub image_views: Vec<VkImageView>,
    pub framebuffers: Vec<VkFramebuffer>,
}

impl Window {

    pub fn acquire_next(&self,semaphore: &Semaphore) -> Option<CommandContext> {
        let index = self.vulkan.device.acquire_next(self.vulkan.swapchain,semaphore);
        Some(CommandContext {
            device: self.vulkan.device,
            framebuffer: self.vulkan.framebuffers[index],
            render_pass: self.vulkan.render_pass,
            command_pool: self.vulkan.command_pool,
            command_buffer: self.vulkan.command_buffer[index],
        })
    }

    pub fn present(&self,index: usize,semaphore: &Semaphore) {
        self.queue.present(index,&self.resources.borrow().swapchain,semaphore)
    }
}
