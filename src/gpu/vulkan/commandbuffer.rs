use {
    crate::*,
    std::{
        rc::Rc,
        ptr::null_mut,
        mem::MaybeUninit,
    }
};

#[derive(Debug)]
pub struct CommandBuffer {
    pub system: Rc<System>,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl System {

    /// Wait for wait_semaphore before submitting command_buffer to the queue, and signal signal_semaphore when rendering is done.
    pub fn submit_command_buffer(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> Result<(),String> {
        let wait_stage = sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let info = sys::VkSubmitInfo {
            sType: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            pWaitDstStageMask: &wait_stage,
            commandBufferCount: 1,
            pCommandBuffers: &command_buffer.vk_command_buffer,
            signalSemaphoreCount: 1,
            pSignalSemaphores: &signal_semaphore.vk_semaphore,
        };
        match unsafe { sys::vkQueueSubmit(self.gpu_system.vk_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => Ok(()),
            code => Err(format!("Unable to submit command buffer to graphics queue ({})",vk_code_to_string(code))),
        }
    }
}

impl CommandBuffer {

    /// Create command buffer.
    pub fn new(system: &Rc<System>) -> Result<CommandBuffer,String> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: system.gpu_system.vk_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkAllocateCommandBuffers(system.gpu_system.vk_device,&info,vk_command_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => Ok(CommandBuffer {
                system: Rc::clone(system),
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            }),
            code => Err(format!("Unable to create command buffer ({})",vk_code_to_string(code))),
        }
    }

    /// Begin the command buffer.
    pub fn begin(&self) -> Result<(),String> {
        let info = sys::VkCommandBufferBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { sys::vkBeginCommandBuffer(self.vk_command_buffer,&info) } {
            sys::VK_SUCCESS => Ok(()),
            code => Err(format!("unable to begin command buffer ({})",vk_code_to_string(code))),
        }
    }

    /// End the command buffer.
    pub fn end(&self) -> bool {
        match unsafe { sys::vkEndCommandBuffer(self.vk_command_buffer) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to end command buffer (error {})",code);
                return false;
            },
        }
        true
    }

    /// Begin render pass.
    pub fn begin_render_pass(&self,window: &Window,index: usize,r: Rect<i32>) {

        // Configure the render pass to write to a rectangle in a window's framebuffer.
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: window.gpu_window.vk_render_pass,
            framebuffer: window.gpu_window.swapchain.borrow().vk_framebuffers[index],
            renderArea: sys::VkRect2D {
                offset: sys::VkOffset2D {
                    x: r.o.x as i32,
                    y: r.o.y as i32,
                },
                extent: sys::VkExtent2D {
                    width: r.s.x as u32,
                    height: r.s.y as u32,
                },
            },
            clearValueCount: 1,
            pClearValues: &clear_color,
        };
        unsafe { sys::vkCmdBeginRenderPass(self.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    /// End render pass.
    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    /// Specify current graphics pipeline.
    pub fn bind_graphics_pipeline(&self,pipeline: &Rc<GraphicsPipeline>) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_pipeline,
        ) };
    }

    /// Specify current compute pipeline.
    pub fn bind_compute_pipeline(&self,pipeline: &Rc<ComputePipeline>) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.vk_pipeline,
        ) };
    }

    /// Specify current vertex buffer.
    pub fn bind_vertex_buffer(&self,vertex_buffer: &Rc<VertexBuffer>) {
        unsafe { sys::vkCmdBindVertexBuffers(
            self.vk_command_buffer,
            0,
            1,
            [ vertex_buffer.vk_buffer, ].as_ptr(),
            [ 0, ].as_ptr(),
        ) };
    }

    /// Specify current index buffer.
    pub fn bind_index_buffer(&self,index_buffer: &Rc<IndexBuffer>) {
        unsafe { sys::vkCmdBindIndexBuffer(
            self.vk_command_buffer,
            index_buffer.vk_buffer,
            0,
            sys::VK_INDEX_TYPE_UINT32,
        ) };
    }

    /// Emit vertices.
    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.vk_command_buffer,
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    /// Emit indexed vertices.
    pub fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: isize,first_instance: usize) {
        unsafe { sys::vkCmdDrawIndexed(
            self.vk_command_buffer,
            index_count as u32,
            instance_count as u32,
            first_index as u32,
            vertex_offset as i32,
            first_instance as u32,
        ) };
    }

    /// Specify current viewport transformation.
    pub fn set_viewport(&self,r: Rect<f32>,min_depth: f32,max_depth: f32) {
        unsafe { sys::vkCmdSetViewport(
            self.vk_command_buffer,
            0,
            1,
            &sys::VkViewport {
                x: r.o.x,
                y: r.o.y,
                width: r.s.x,
                height: r.s.y,
                minDepth: min_depth,
                maxDepth: max_depth,
            },
        ) };
    }

    /// Specify current scissor rectangle.
    pub fn set_scissor(&self,r: Rect<f32>) {
        unsafe { sys::vkCmdSetScissor(
            self.vk_command_buffer,
            0,
            1,
            &sys::VkRect2D {
                offset: sys::VkOffset2D {
                    x: r.o.x as i32,
                    y: r.o.y as i32,
                },
                extent: sys::VkExtent2D {
                    width: r.s.x as u32,
                    height: r.s.y as u32,
                },
            },
        ) };
    }
}

impl Drop for CommandBuffer {
    
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.system.gpu_system.vk_device,self.system.gpu_system.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
