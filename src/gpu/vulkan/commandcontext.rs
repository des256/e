use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandContext<'system,'window> {
    pub window: &'window Window<'system>,
    pub index: usize,
}

impl<'system> Window<'system> {

    pub fn acquire_next(&self,semaphore: &Semaphore) -> CommandContext {
        let mut index = 0u32;
        unsafe { sys::vkAcquireNextImageKHR(self.system.vk_device,self.vk_swapchain,0xFFFFFFFFFFFFFFFF,semaphore.vk_semaphore,null_mut(),&mut index) };
        CommandContext {
            window: &self,
            index: index as usize,
        }
    }

    pub fn present(&self,index: usize,semaphore: &Semaphore) {
        let image_index = index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.system.vk_queue,&info) };
    }
}

impl<'system,'window> CommandContext<'system,'window> {

    pub fn begin(&self) -> bool {
        let info = sys::VkCommandBufferBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { sys::vkBeginCommandBuffer(self.window.system.vk_command_buffer,&info) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to begin command buffer (error {})",code);
                false
            }
        }
    }

    pub fn end(&self) -> bool {
        match unsafe { sys::vkEndCommandBuffer(self.window.system.vk_command_buffer) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to end command buffer (error {})",code);
                false
            },
        }
    }

    pub fn begin_render_pass(&self,r: Rect<i32>) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: self.window.vk_render_pass,
            framebuffer: self.window.vk_framebuffers[self.index],
            renderArea: sys::VkRect2D {
                offset: sys::VkOffset2D {
                    x: r.o.x,
                    y: r.o.y,
                },
                extent: sys::VkExtent2D {
                    width: r.s.x as u32,
                    height: r.s.y as u32,
                },
            },
            clearValueCount: 1,
            pClearValues: &clear_color,
        };
        unsafe { sys::vkCmdBeginRenderPass(self.window.system.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.window.system.vk_command_buffer) };
    }

    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.window.system.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_graphics_pipeline,
        ) };
    }

    pub fn bind_vertex_buffer(&self,vertex_buffer: &VertexBuffer) {
        unsafe { sys::vkCmdBindVertexBuffers(
            self.window.system.vk_command_buffer,
            0,
            1,
            &vertex_buffer.vk_buffer,
            null_mut(),
        ) };
    }

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.window.system.vk_command_buffer,
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    pub fn set_viewport(&self,r: Hyper<f32>) {
        unsafe { sys::vkCmdSetViewport(
            self.window.system.vk_command_buffer,
            0,
            1,
            &sys::VkViewport {
                x: r.o.x,
                y: r.o.y,
                width: r.s.x,
                height: r.s.y,
                minDepth: r.o.z,
                maxDepth: r.o.z + r.s.z,
            },
        ) };
    }

    pub fn set_scissor(&self,r: Rect<i32>) {
        unsafe { sys::vkCmdSetScissor(
            self.window.system.vk_command_buffer,
            0,
            1,
            &sys::VkRect2D {
                offset: sys::VkOffset2D {
                    x: r.o.x,
                    y: r.o.y,
                },
                extent: sys::VkExtent2D {
                    width: r.s.x as u32,
                    height: r.s.y as u32,
                },
            },
        ) };
    }

    pub fn submit(&self,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
#[cfg(gpu="vulkan")]
        {
            let wait_stage = sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            let info = sys::VkSubmitInfo {
                sType: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
                pNext: null_mut(),
                waitSemaphoreCount: 1,
                pWaitSemaphores: &wait_semaphore.vk_semaphore,
                pWaitDstStageMask: &wait_stage,
                commandBufferCount: 1,
                pCommandBuffers: &self.window.system.vk_command_buffer,
                signalSemaphoreCount: 1,
                pSignalSemaphores: &signal_semaphore.vk_semaphore,
            };
            match unsafe { sys::vkQueueSubmit(self.window.system.vk_queue,1,&info,null_mut()) } {
                sys::VK_SUCCESS => true,
                code => {
                    println!("unable to submit to graphics queue (error {})",code);
                    false
                },
            }
        }

#[cfg(not(gpu="vulkan"))]
        false
    }
}
