use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandBuffer<'system,'window,'context> {
    pub context: &'context CommandContext<'system,'window>,
}

impl<'system,'window> CommandContext<'system,'window> {

    /// Begin a command buffer for this context.
    pub fn begin(&self) -> Option<CommandBuffer> {
        let info = sys::VkCommandBufferBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { sys::vkBeginCommandBuffer(self.window.system.vk_command_buffer,&info) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to begin command buffer (error {})",code);
                return None;
            }
        }
        Some(CommandBuffer {
            context: &self,
        })
    }
}

impl<'system,'window,'context> CommandBuffer<'system,'window,'context> {

    /// Begin render pass.
    pub fn begin_render_pass(&self,r: Rect<i32>) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: self.context.window.vk_render_pass,
            framebuffer: self.context.window.vk_framebuffers[self.context.index],
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
        unsafe { sys::vkCmdBeginRenderPass(self.context.window.system.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    /// End render pass.
    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.context.window.system.vk_command_buffer) };
    }

    /// Specify current graphics pipeline.
    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.context.window.system.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_graphics_pipeline,
        ) };
    }

    /// Specify current vertex buffer.
    pub fn bind_vertex_buffer(&self,vertex_buffer: &VertexBuffer) {
        unsafe { sys::vkCmdBindVertexBuffers(
            self.context.window.system.vk_command_buffer,
            0,
            1,
            &vertex_buffer.vk_buffer,
            null_mut(),
        ) };
    }

    /// Emit vertices.
    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.context.window.system.vk_command_buffer,
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    /// Specify current viewport transformation.
    pub fn set_viewport(&self,r: Hyper<f32>) {
        unsafe { sys::vkCmdSetViewport(
            self.context.window.system.vk_command_buffer,
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

    /// Specify current scissor rectangle.
    pub fn set_scissor(&self,r: Rect<i32>) {
        unsafe { sys::vkCmdSetScissor(
            self.context.window.system.vk_command_buffer,
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

    /// Finish the commands, and submit them such that they will start as soon as wait_semaphore is triggered. When all drawing is done, trigger signal_semaphore.
    pub fn end_submit(&self,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        match unsafe { sys::vkEndCommandBuffer(self.context.window.system.vk_command_buffer) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to end command buffer (error {})",code);
                return false;
            },
        }
        let wait_stage = sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let info = sys::VkSubmitInfo {
            sType: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            pWaitDstStageMask: &wait_stage,
            commandBufferCount: 1,
            pCommandBuffers: &self.context.window.system.vk_command_buffer,
            signalSemaphoreCount: 1,
            pSignalSemaphores: &signal_semaphore.vk_semaphore,
        };
        match unsafe { sys::vkQueueSubmit(self.context.window.system.vk_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to submit to graphics queue (error {})",code);
                false
            },
        }
    }
}
