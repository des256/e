use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandContext<'system,'window> {
    pub window: &'window Window<'system>,
    pub index: usize,
}

impl<'system> Window<'system> {

    /// Acquire command context for next available frame in the window's swap chain. Also indicate to trigger signal_semaphore when this frame is ready to be drawn to.
    pub fn acquire_next(&self,signal_semaphore: &Semaphore) -> CommandContext {
        let mut index = 0u32;
        unsafe { sys::vkAcquireNextImageKHR(
            self.system.vk_device,
            self.vk_swapchain,
            0xFFFFFFFFFFFFFFFF,
            signal_semaphore.vk_semaphore,
            null_mut(),
            &mut index,
        ) };
        CommandContext {
            window: &self,
            index: index as usize,
        }
    }

    /// Present work from a context to the window as soon as wait_semaphore gets triggered.
    pub fn present(&self,context: CommandContext,wait_semaphore: &Semaphore) {
        let image_index = context.index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.system.vk_queue,&info) };
    }

    /*
    Presenting pSwapchains[0] image without calling vkGetPhysicalDeviceSurfaceSupportKHR
    */
}
