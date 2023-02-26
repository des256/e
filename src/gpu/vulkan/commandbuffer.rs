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
    /// Create command buffer.
    pub fn create_command_buffer(self: &Rc<Self>) -> Result<Rc<CommandBuffer>,String> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.gpu_system.vk_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkAllocateCommandBuffers(self.gpu_system.vk_device,&info,vk_command_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => Ok(Rc::new(CommandBuffer {
                system: Rc::clone(self),
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            })),
            code => Err(format!("Unable to create command buffer ({})",vk_code_to_string(code))),
        }
    }
}

impl CommandBuffer {

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
            code => Err(format!("unable to begin command buffer ({})",vk_code_to_string(code)))
        }
    }

    /// End render pass.
    pub fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    /// Begin render pass.
    pub fn begin_render_pass(&self,window: &Window,index: usize,r: Rect<i32>) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: window.gpu_window.vk_render_pass,
            framebuffer: window.gpu_window.swapchain_resources.vk_framebuffers[index],
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
}

impl Drop for CommandBuffer {
    
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.system.gpu_system.vk_device,self.system.gpu_system.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
