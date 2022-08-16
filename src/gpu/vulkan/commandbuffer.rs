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
        match unsafe { sys::vkBeginCommandBuffer(self.window.vk_command_buffers[self.index],&info) } {
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

    /*
    Calling vkBeginCommandBuffer() on active VkCommandBuffer 0x55b677443510[]
    before it has completed. You must check command buffer fence before this
    call. The Vulkan spec states: commandBuffer must not be in the recording
    or pending state
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00049)
    */

    /*
    Call to vkBeginCommandBuffer() on VkCommandBuffer 0x55b677443510[] attempts
    to implicitly reset cmdBuffer created from VkCommandPool 0x10000000001[]
    that does NOT have the VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT bit
    set. The Vulkan spec states: If commandBuffer was allocated from a
    VkCommandPool which did not have the
    VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT flag set, commandBuffer
    must be in the initial state
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkBeginCommandBuffer-commandBuffer-00050)
    */
}

impl<'system,'window,'context> CommandBuffer<'system,'window,'context> {

    /// Begin render pass.
    pub fn begin_render_pass(&self,r: i32r) {
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
        unsafe { sys::vkCmdBeginRenderPass(self.context.window.vk_command_buffers[self.context.index],&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    /// End render pass.
    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.context.window.vk_command_buffers[self.context.index]) };
    }

    /// Specify current graphics pipeline.
    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.context.window.vk_command_buffers[self.context.index],
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_graphics_pipeline,
        ) };
    }

    /// Specify current vertex buffer.
    pub fn bind_vertex_buffer(&self,vertex_buffer: &VertexBuffer) {
        unsafe { sys::vkCmdBindVertexBuffers(
            self.context.window.vk_command_buffers[self.context.index],
            0,
            1,
            [ vertex_buffer.vk_buffer, ].as_ptr(),
            [ 0, ].as_ptr(),
        ) };
    }

    /// Specify current index buffer.
    pub fn bind_index_buffer(&self,index_buffer: &IndexBuffer) {
        unsafe { sys::vkCmdBindIndexBuffer(
            self.context.window.vk_command_buffers[self.context.index],
            index_buffer.vk_buffer,
            0,
            sys::VK_INDEX_TYPE_UINT32,
        ) };
    }

    /// Emit vertices.
    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.context.window.vk_command_buffers[self.context.index],
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    /// Emit indexed vertices.
    pub fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: isize,first_instance: usize) {
        unsafe { sys::vkCmdDrawIndexed(
            self.context.window.vk_command_buffers[self.context.index],
            index_count as u32,
            instance_count as u32,
            first_index as u32,
            vertex_offset as i32,
            first_instance as u32,
        ) };
    }

    /// Specify current viewport transformation.
    pub fn set_viewport(&self,h: f32h) {
        unsafe { sys::vkCmdSetViewport(
            self.context.window.vk_command_buffers[self.context.index],
            0,
            1,
            &sys::VkViewport {
                x: h.o.x,
                y: h.o.y,
                width: h.s.x,
                height: h.s.y,
                minDepth: h.o.z,
                maxDepth: h.o.z + h.s.z,
            },
        ) };
    }

    /// Specify current scissor rectangle.
    pub fn set_scissor(&self,r: i32r) {
        unsafe { sys::vkCmdSetScissor(
            self.context.window.vk_command_buffers[self.context.index],
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
        match unsafe { sys::vkEndCommandBuffer(self.context.window.vk_command_buffers[self.context.index]) } {
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
            pCommandBuffers: &self.context.window.vk_command_buffers[self.context.index],
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
