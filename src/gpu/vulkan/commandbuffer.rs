use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandBuffer<'system> {
    pub system: &'system System,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl<'system> CommandBuffer<'system> {

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

    pub fn begin_render_pass(&self,window: &Window,framebuffer: &Framebuffer) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: window.vk_renderpass,
            framebuffer: framebuffer.vk_framebuffer,
            renderArea: sys::VkRect2D { offset: sys::VkOffset2D { x: 0,y: 0 },extent: window.vk_extent, },
            clearValueCount: 1,
            pClearValues: &clear_color,
        };
        unsafe { sys::vkCmdBeginRenderPass(self.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }

    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(self.vk_command_buffer,sys::VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.vk_graphics_pipeline) };
    }

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(self.vk_command_buffer,vertex_count as u32,instance_count as u32,first_vertex as u32,first_instance as u32) };
    }
}

impl<'system> Drop for CommandBuffer<'system> {
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.system.vk_device,self.system.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
