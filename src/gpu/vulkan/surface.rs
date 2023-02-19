use {
    crate::*,
    std::ptr::null_mut,
};

pub(crate) struct SwapchainResources {
    pub vk_swapchain: sys::VkSwapchainKHR,
    pub vk_framebuffers: Vec<sys::VkFramebuffer>,
    pub vk_image_views: Vec<sys::VkImageView>,
}

pub(crate) struct WindowGpu {
    pub vk_surface: sys::VkSurfaceKHR,
    pub vk_render_pass: sys::VkRenderPass,
    pub swapchain_resources: SwapchainResources,
}

impl Window {

    /// Update swapchain resources after window resize.
    pub fn update_swapchain_resources(&mut self,r: Rect<isize,usize>) {
        self.drop_swapchain_resources();
        if let Some(swapchain_resources) = self.system.create_swapchain_resources(self.gpu.vk_surface,self.gpu.vk_render_pass,r) {
            self.gpu.swapchain_resources = swapchain_resources;
        }
    }
        
    /// Return number of framebuffers in the swapchain.
    pub fn get_framebuffer_count(&self) -> usize {
        self.gpu.swapchain_resources.vk_framebuffers.len()
    }

    /// Acquire index of the next available framebuffer in the window's swap chain. Also indicate to trigger signal_semaphore when this frame is ready to be drawn to.
    pub fn acquire_next(&self,signal_semaphore: &Semaphore) -> usize {
        let mut index = 0u32;
        unsafe { sys::vkAcquireNextImageKHR(
            self.system.gpu.vk_device,
            self.gpu.swapchain_resources.vk_swapchain,
            0xFFFFFFFFFFFFFFFF,
            signal_semaphore.vk_semaphore,
            null_mut(),
            &mut index,
        ) };
        index as usize
    }

    /// Present a newly created framebuffer in the swapchain as soon as wait_semaphore gets triggered.
    pub fn present(&self,index: usize,wait_semaphore: &Semaphore) {
        let image_index = index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.gpu.swapchain_resources.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.system.gpu.vk_queue,&info) };
    }

    /// Drop GPU-specific Window part.
    pub fn drop_gpu(&self) {
        self.drop_swapchain_resources();
        unsafe {
            sys::vkDestroySurfaceKHR(self.system.gpu.vk_instance,self.gpu.vk_surface,null_mut());
            sys::vkDestroyRenderPass(self.system.gpu.vk_device,self.gpu.vk_render_pass,null_mut());
        }
    }

    /// Drop current swapchain resources.
    fn drop_swapchain_resources(&self) {
        unsafe {
            for vk_framebuffer in &self.gpu.swapchain_resources.vk_framebuffers {
                sys::vkDestroyFramebuffer(self.system.gpu.vk_device,*vk_framebuffer,null_mut());
            }
            for vk_image_view in &self.gpu.swapchain_resources.vk_image_views {
                sys::vkDestroyImageView(self.system.gpu.vk_device,*vk_image_view,null_mut());
            }
            sys::vkDestroySwapchainKHR(self.system.gpu.vk_device,self.gpu.swapchain_resources.vk_swapchain,null_mut());
        }    
    }
}
