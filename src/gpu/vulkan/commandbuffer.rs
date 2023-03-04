use {
    super::*,
    crate::gpu,
    std::{
        rc::Rc,
        ptr::null_mut,
    }
};

#[derive(Debug)]
pub struct CommandBuffer {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_command_buffer: sys::VkCommandBuffer,
}

impl gpu::CommandBuffer for CommandBuffer {

    type Surface = Surface;
    type GraphicsPipeline = GraphicsPipeline;
    type ComputePipeline = ComputePipeline;
    type VertexBuffer = VertexBuffer;
    type IndexBuffer = IndexBuffer;

    /// Begin the command buffer.
    fn begin(&self) -> Result<(),String> {
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
    fn end(&self) -> bool {
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
    fn begin_render_pass(&self,surface: &Self::Surface,index: usize,r: Rect<i32>) {

        // Configure the render pass to write to a rectangle in a surface's framebuffer.
        let clear_color = sys::VkClearValue {
            color: sys::VkClearColorValue {
                float32: [0.0,0.0,0.0,1.0]
            }
        };
        let info = sys::VkRenderPassBeginInfo {
        sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            pNext: null_mut(),
            renderPass: surface.vk_render_pass,
            framebuffer: surface.vk_framebuffers[index],
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
    fn end_render_pass(&self) {
        unsafe { sys::vkCmdEndRenderPass(self.vk_command_buffer) };
    }

    /// Specify current graphics pipeline.
    fn bind_graphics_pipeline(&self,pipeline: &Self::GraphicsPipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline.vk_pipeline,
        ) };
    }

    /// Specify current compute pipeline.
    fn bind_compute_pipeline(&self,pipeline: &Self::ComputePipeline) {
        unsafe { sys::vkCmdBindPipeline(
            self.vk_command_buffer,
            sys::VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.vk_pipeline,
        ) };
    }

    /// Specify current vertex buffer.
    fn bind_vertex_buffer(&self,vertex_buffer: &Self::VertexBuffer) {
        unsafe { sys::vkCmdBindVertexBuffers(
            self.vk_command_buffer,
            0,
            1,
            [ vertex_buffer.vk_buffer, ].as_ptr(),
            [ 0, ].as_ptr(),
        ) };
    }

    /// Specify current index buffer.
    fn bind_index_buffer(&self,index_buffer: &Self::IndexBuffer) {
        unsafe { sys::vkCmdBindIndexBuffer(
            self.vk_command_buffer,
            index_buffer.vk_buffer,
            0,
            sys::VK_INDEX_TYPE_UINT32,
        ) };
    }

    /// Emit vertices.
    fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        unsafe { sys::vkCmdDraw(
            self.vk_command_buffer,
            vertex_count as u32,
            instance_count as u32,
            first_vertex as u32,
            first_instance as u32,
        ) };
    }

    /// Emit indexed vertices.
    fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: isize,first_instance: usize) {
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
    fn set_viewport(&self,r: Rect<i32>,min_depth: f32,max_depth: f32) {
        unsafe { sys::vkCmdSetViewport(
            self.vk_command_buffer,
            0,
            1,
            &sys::VkViewport {
                x: r.o.x as f32,
                y: r.o.y as f32,
                width: r.s.x as f32,
                height: r.s.y as f32,
                minDepth: min_depth,
                maxDepth: max_depth,
            },
        ) };
    }

    /// Specify current scissor rectangle.
    fn set_scissor(&self,r: Rect<i32>) {
        unsafe { sys::vkCmdSetScissor(
            self.vk_command_buffer,
            0,
            1,
            &sys::VkRect2D {
                offset: sys::VkOffset2D { x: r.o.x, y: r.o.y, },
                extent: sys::VkExtent2D { width: r.s.x as u32,height: r.s.y as u32, },
            },
        ) };
    }
}

impl Drop for CommandBuffer {
    
    fn drop(&mut self) {
        unsafe { sys::vkFreeCommandBuffers(self.gpu.vk_device,self.gpu.vk_command_pool,1,&self.vk_command_buffer) };
    }
}
