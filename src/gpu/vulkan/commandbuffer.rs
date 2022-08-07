use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandBuffer {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_command_pool: sys::VkCommandPool,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl CommandBuffer {

    pub fn begin(&self) -> bool {
        let info = sys::VkCommandBufferBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { sys::vkBeginCommandBuffer(self.vk_command_buffer,&info) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to begin command buffer (error {})",code);
                false
            }
        }
    }

    pub fn end(&self) -> bool {
        match unsafe { sys::vkEndCommandBuffer(self.vk_command_buffer) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to end command buffer (error {})",code);
                false
            },
        }
    }

    pub fn begin_render_pass(&self,render_pass: &RenderPass,framebuffer: &Framebuffer,r: Rect<i32>) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: render_pass.vk_renderpass,
            framebuffer: framebuffer.vk_framebuffer,
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
        unsafe { sys::vkCmdBeginRenderPass(self.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_graphics_pipeline,
        ) };
    }

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.vk_command_buffer,
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    pub fn set_viewport(&self,r: Hyper<f32>) {
        unsafe { sys::vkCmdSetViewport(
            self.vk_command_buffer,
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
            self.vk_command_buffer,
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
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.vk_device,self.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
