// E - GPU (Vulkan) - Queue
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

pub struct Queue {
    pub session: Rc<Session>,
    pub(crate) vk_queue: VkQueue,
}

impl Session {

    pub fn get_queue(self: &Rc<Self>,family: QueueFamilyID,index: usize) -> Option<Queue> {
        let mut vk_queue = MaybeUninit::uninit();
        unsafe { vkGetDeviceQueue(self.vk_device,family,index as u32,vk_queue.as_mut_ptr()) };
        let vk_queue = unsafe { vk_queue.assume_init() };
        Some(Queue {
            session: Rc::clone(&self),
            vk_queue: vk_queue,
        })
    }
}

impl Queue {

    pub fn submit(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        let wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let info = VkSubmitInfo {
            sType: VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            pWaitDstStageMask: &wait_stage,
            commandBufferCount: 1,
            pCommandBuffers: &command_buffer.vk_command_buffer,
            signalSemaphoreCount: 1,
            pSignalSemaphores: &signal_semaphore.vk_semaphore,
        };
        match unsafe { vkQueueSubmit(self.vk_queue,1,&info,null_mut()) } {
            VK_SUCCESS => true,
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to submit to queue (error {}).",code);
                false
            },
        }
    }

    pub fn present(&self,swapchain: &SwapChain,index: usize,semaphore: &Semaphore) {
        let image_index = index as u32;
        let info = VkPresentInfoKHR {
            sType: VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &swapchain.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { vkQueuePresentKHR(self.vk_queue,&info) };
    }
}
