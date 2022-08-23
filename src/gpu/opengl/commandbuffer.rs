use {
    crate::*,
    std::rc::Rc,
};

pub enum Command {
    BeginRenderPass(Rect<isize,usize>),
    EndRenderPass,
    BindPipeline,
    BindVertexBuffer,
    BindIndexBuffer,
    Draw(usize,usize,usize,usize),
    DrawIndexed(usize,usize,usize,isize,usize),
    SetViewport(Hyper<f32,f32>),
    SetScissor(Rect<isize,usize>),
}

pub struct CommandBuffer {
    pub system: Rc<System>,
    pub commands: Vec<Command>,
}

impl CommandBuffer {

    /// Begin a command buffer.
    pub fn begin(&self) -> bool {
        true
    }

    /// Begin render pass.
    pub fn begin_render_pass(&mut self,r: Rect<isize,usize>) {
        self.commands.push(Command::BeginRenderPass(r));
    }

    /// End render pass.
    pub fn end_render_pass(&mut self) {
        self.commands.push(Command::EndRenderPass);
    }

    /// Specify current graphics pipeline.
    pub fn bind_pipeline(&mut self,pipeline: &GraphicsPipeline) {
        self.commands.push(Command::BindPipeline);
    }

    /// Specify current vertex buffer.
    pub fn bind_vertex_buffer(&mut self,vertex_buffer: &VertexBuffer) {
        self.commands.push(Command::BindVertexBuffer);
    }

    /// Specify current index buffer.
    pub fn bind_index_buffer(&mut self,index_buffer: &IndexBuffer) {
        self.commands.push(Command::BindIndexBuffer);
    }

    /// Emit vertices.
    pub fn draw(&mut self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        self.commands.push(Command::Draw(vertex_count,instance_count,first_vertex,first_instance));
    }

    /// Emit indexed vertices.
    pub fn draw_indexed(&mut self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: isize,first_instance: usize) {
        self.commands.push(Command::DrawIndexed(index_count,instance_count,first_index,vertex_offset,first_instance));
    }

    /// Specify current viewport transformation.
    pub fn set_viewport(&mut self,h: Hyper<f32,f32>) {
        self.commands.push(Command::SetViewport(h));
    }

    /// Specify current scissor rectangle.
    pub fn set_scissor(&mut self,r: Rect<isize,usize>) {
        self.commands.push(Command::SetScissor(r));
    }

    /// Finish the command buffer.
    pub fn end(&self) -> bool {
        // actually perform the commands here
        true
    }
}
