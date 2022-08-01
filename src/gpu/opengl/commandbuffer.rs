use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub(crate) enum Command {
    BeginRenderPass,
    EndRenderPass,
    BindPipeline,
    Draw,
};

pub struct CommandBuffer<'system,'gpu,'screen> {
    pub screen: &'screen Screen<'system,'gpu>,
    pub(crate) buffer: Vec<Command>,
}

impl<'system,'gpu> Screen<'system,'gpu> {

    pub fn create_graphics_commandbuffer(&self) -> Option<CommandBuffer> {

        Some(CommandBuffer {
            screen: &self,
            buffer: Vec::new(),
        });
    }

    pub fn create_transfer_commandbuffer(&self) -> Option<CommandBuffer> {

        Some(CommandBuffer {
            screen: &self,
            buffer: Vec::new(),
        });
    }

    pub fn create_compute_commandbuffer(&self) -> Option<CommandBuffer> {

        Some(CommandBuffer {
            screen: &self,
            buffer: Vec::new(),
        });
    }
}

impl<'system,'gpu,'screen> CommandBuffer<'system,'gpu,'screen> {

    pub fn begin(&self) -> bool {
        self.buffer.clear();
        true
    }

    pub fn end(&self) -> bool {
        true
    }

    pub fn begin_render_pass(&self,render_pass: &RenderPass,framebuffer: &Framebuffer) {
        self.buffer.push(Command::BeginRenderPass);
    }

    pub fn end_render_pass(&self) {
        self.buffer.push(Command::EndRenderPass);
    }

    pub fn bind_pipeline(&self,pipeline: &GraphicsPipeline) {
        self.buffer.push(Command::BindPipeline);
    }

    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        self.buffer.push(Command::Draw);
    }
}
