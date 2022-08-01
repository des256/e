use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct CommandBuffer<'system,'gpu,'screen> {
    pub screen: &'screen Screen<'system,'gpu>,
    pub(crate) vk_command_pool: sys::VkCommandPool,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl<'system,'gpu> Screen<'system,'gpu> {

    pub fn create_graphics_commandbuffer(&self) -> Option<CommandBuffer> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.vk_graphics_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        if unsafe { sys::vkAllocateCommandBuffers(self.vk_device,&info,vk_command_buffer.as_mut_ptr()) } == sys::VK_SUCCESS {
            Some(CommandBuffer {
                screen: &self,
                vk_command_pool: self.vk_graphics_command_pool,
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            })
        }
        else {
            None
        }
    }

    pub fn create_transfer_commandbuffer(&self) -> Option<CommandBuffer> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.vk_transfer_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        if unsafe { sys::vkAllocateCommandBuffers(self.vk_device,&info,vk_command_buffer.as_mut_ptr()) } == sys::VK_SUCCESS {
            Some(CommandBuffer {
                screen: &self,
                vk_command_pool: self.vk_transfer_command_pool,
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            })
        }
        else {
            None
        }
    }

    pub fn create_compute_commandbuffer(&self) -> Option<CommandBuffer> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.vk_compute_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        if unsafe { sys::vkAllocateCommandBuffers(self.vk_device,&info,vk_command_buffer.as_mut_ptr()) } == sys::VK_SUCCESS {
            Some(CommandBuffer {
                screen: &self,
                vk_command_pool: self.vk_compute_command_pool,
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            })
        }
        else {
            None
        }
    }
}

impl<'system,'gpu,'screen> CommandBuffer<'system,'gpu,'screen> {

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
                println!("Unable to begin command buffer (error {})",code);
                false
            }
        }
    }

    pub fn end(&self) -> bool {
        match unsafe { sys::vkEndCommandBuffer(self.vk_command_buffer) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("Unable to end command buffer (error {})",code);
                false
            },
        }
    }

    /*
    pub fn begin_render_pass(&self,render_pass: &RenderPass,framebuffer: &Framebuffer) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: render_pass.vk_render_pass,
            framebuffer: framebuffer.vk_framebuffer,
            renderArea: sys::VkRect2D { offset: sys::VkOffset2D { x: 0,y: 0 },extent: sys::VkExtent2D { width: framebuffer.size.x as u32,height: framebuffer.size.y as u32 } },
            clearValueCount: 1,
            pClearValues: &clear_color,
        };
        unsafe { sys::vkCmdBeginRenderPass(self.vk_command_buffer,&info,sys::VK_SUBPASS_CONTENTS_INLINE) }
    }
    */

    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    /*
    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(self.vk_command_buffer,sys::VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.vk_graphics_pipeline) };
    }
    */

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(self.vk_command_buffer,vertex_count as u32,instance_count as u32,first_vertex as u32,first_instance as u32) };
    }
}

impl<'system,'gpu,'screen> Drop for CommandBuffer<'system,'gpu,'screen> {
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.screen.vk_device,self.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
