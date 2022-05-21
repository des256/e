use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct CommandBuffer {
    pub session: Rc<Session>,
    pub(crate) vk_command_pool: VkCommandPool,
    pub(crate) vk_command_buffer: VkCommandBuffer,
}

impl Session {

    pub fn create_commandbuffer(self: &Rc<Self>,queue_family: QueueFamilyID) -> Option<Rc<CommandBuffer>> {

        let info = VkCommandBufferAllocateInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.vk_command_pools[queue_family as usize],
            level: VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        match unsafe { vkAllocateCommandBuffers(self.vk_device,&info,vk_command_buffer.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create Vulkan command buffer (error {})",code);
                return None;
            }
        }
        Some(Rc::new(CommandBuffer {
            session: Rc::clone(self),
            vk_command_pool: self.vk_command_pools[queue_family as usize],
            vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
        }))
    }
}

impl CommandBuffer {

    pub fn begin(&self) -> bool {
        let info = VkCommandBufferBeginInfo {
            sType: VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { vkBeginCommandBuffer(self.vk_command_buffer,&info) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to begin Vulkan command buffer (error {})",code);
                return false;
            },
        }
        true
    }

    pub fn end(&self) -> bool {
        match unsafe { vkEndCommandBuffer(self.vk_command_buffer) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to end Vulkan command buffer (error {})",code);
                return false;
            },
        }
        true
    }

    pub fn begin_render_pass(&self,render_pass: &RenderPass,framebuffer: &Framebuffer) {
        let clear_color = VkClearValue {
            color: VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = VkRenderPassBeginInfo {
            sType: VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: render_pass.vk_render_pass,
            framebuffer: framebuffer.vk_framebuffer,
            renderArea: VkRect2D { offset: VkOffset2D { x: 0,y: 0 },extent: VkExtent2D { width: framebuffer.size.x as u32,height: framebuffer.size.y as u32 } },
            clearValueCount: 1,
            pClearValues: &clear_color,
        };
        unsafe { vkCmdBeginRenderPass(self.vk_command_buffer,&info,VK_SUBPASS_CONTENTS_INLINE) }
    }

    pub fn end_render_pass(&self) {
        unsafe { vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        unsafe { vkCmdBindPipeline(self.vk_command_buffer,VK_PIPELINE_BIND_POINT_GRAPHICS,pipeline.vk_graphics_pipeline) };
    }

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { vkCmdDraw(self.vk_command_buffer,vertex_count as u32,instance_count as u32,first_vertex as u32,first_instance as u32) };
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe { vkFreeCommandBuffers(self.session.vk_device,self.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
