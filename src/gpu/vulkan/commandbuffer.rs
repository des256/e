use {
    crate::*,
    std::{
        rc::Rc,
        ptr::null_mut,
        cell::Cell,
    }
};

pub struct CommandBuffer {
    pub system: Rc<System>,
    pub vertex_buffer: Cell<Option<Rc<VertexBuffer>>>,
    pub index_buffer: Cell<Option<Rc<IndexBuffer>>>,
    pub graphics_pipeline: Cell<Option<Rc<GraphicsPipeline>>>,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl CommandBuffer {

    /// Begin the command buffer.
    pub fn begin(&self) -> bool {
        let info = sys::VkCommandBufferBeginInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            pNext: null_mut(),
            flags: 0,
            pInheritanceInfo: null_mut(),
        };
        match unsafe { sys::vkBeginCommandBuffer(self.vk_command_buffer,&info) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to begin command buffer (error {})",code);
                return false;
            }
        }
        true
    }

    /// Begin render pass.
    pub fn begin_render_pass(&self,window: &Window,index: usize,r: Rect<isize,usize>) {
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: window.gpu.vk_render_pass,
            framebuffer: window.gpu.swapchain_resources.vk_framebuffers[index],
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
    pub fn bind_pipeline(&self,pipeline: &Rc<GraphicsPipeline>) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_graphics_pipeline,
        ) };
        self.graphics_pipeline.set(Some(Rc::clone(pipeline)));
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
        self.vertex_buffer.set(Some(Rc::clone(vertex_buffer)));
    }

    /// Specify current index buffer.
    pub fn bind_index_buffer(&self,index_buffer: &Rc<IndexBuffer>) {
        unsafe { sys::vkCmdBindIndexBuffer(
            self.vk_command_buffer,
            index_buffer.vk_buffer,
            0,
            sys::VK_INDEX_TYPE_UINT32,
        ) };
        self.index_buffer.set(Some(Rc::clone(index_buffer)));
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
    pub fn set_viewport(&self,h: Hyper<f32,f32>) {
        unsafe { sys::vkCmdSetViewport(
            self.vk_command_buffer,
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
    pub fn set_scissor(&self,r: Rect<isize,usize>) {
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
        unsafe { sys::vkFreeCommandBuffers(self.system.gpu.vk_device,self.system.gpu.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
