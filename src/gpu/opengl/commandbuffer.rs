use {
    super::*,
    crate::checkgl,
    crate::gpu,
    std::{
        rc::Rc,
        cell::RefCell,
        ptr::null_mut,
    }
};

#[derive(Debug)]
pub enum Command {
    BeginRenderPass(u32,Rect<i32>),
    EndRenderPass,
    BindGraphicsPipeline(Rc<GraphicsPipeline>),
    BindComputePipeline(Rc<ComputePipeline>),
    BindVertexBuffer(Rc<VertexBuffer>),
    BindIndexBuffer(Rc<IndexBuffer>),
    Draw(usize,usize,usize,usize),
    DrawIndexed(usize,usize,usize,usize,usize),
    SetViewport(Rect<i32>,f32,f32),
    SetScissor(Rect<i32>),
}

#[derive(Debug)]
pub struct CommandBuffer {
    pub gpu: Rc<Gpu>,
    pub commands: RefCell<Vec<Command>>,
}

impl CommandBuffer {
    pub fn execute(&self) {
        let mut topology: sys::GLenum = sys::GL_POINTS;
        let commands = self.commands.borrow();
        for command in commands.iter() {
            match command {
                Command::BeginRenderPass(window_id,_) => {
#[cfg(system="linux")]
                    unsafe { sys::glXMakeCurrent(self.gpu.system.xdisplay,*window_id as u64,self.gpu.glx_context,) };
                },
                Command::EndRenderPass => { },
                Command::BindGraphicsPipeline(pipeline) => {
                    topology = pipeline.topology;
                    // TODO: primitive restart
                    // TODO: patch control points
                    unsafe {
                        checkgl!(sys::glUseProgram(pipeline.shader_program));
                        if pipeline.depth_clamp {
                            checkgl!(sys::glEnable(sys::GL_DEPTH_CLAMP));
                        }
                        else {
                            checkgl!(sys::glDisable(sys::GL_DEPTH_CLAMP));
                        }
                        if pipeline.rasterizer_discard {
                            checkgl!(sys::glEnable(sys::GL_RASTERIZER_DISCARD));
                        }
                        else {
                            checkgl!(sys::glDisable(sys::GL_RASTERIZER_DISCARD));
                        }
                        checkgl!(sys::glPolygonMode(sys::GL_FRONT_AND_BACK,pipeline.polygon_mode));
                        match pipeline.cull_mode {
                            Some((cull_face,front_face)) => {
                                checkgl!(sys::glEnable(sys::GL_CULL_FACE));
                                checkgl!(sys::glCullFace(cull_face));
                                checkgl!(sys::glFrontFace(front_face));
                            },
                            None => checkgl!(sys::glDisable(sys::GL_CULL_FACE)),
                        }
                        match pipeline.polygon_offset {
                            Some((s,c)) => {
                                checkgl!(sys::glEnable(sys::GL_POLYGON_OFFSET_POINT));
                                checkgl!(sys::glEnable(sys::GL_POLYGON_OFFSET_LINE));
                                checkgl!(sys::glEnable(sys::GL_POLYGON_OFFSET_FILL));
                                checkgl!(sys::glPolygonOffset(s,c));
                            },
                            None => {
                                checkgl!(sys::glDisable(sys::GL_POLYGON_OFFSET_POINT));
                                checkgl!(sys::glDisable(sys::GL_POLYGON_OFFSET_LINE));
                                checkgl!(sys::glDisable(sys::GL_POLYGON_OFFSET_FILL));    
                            },
                        }
                        checkgl!(sys::glLineWidth(pipeline.line_width));
                        // TODO: rasterization_samples
                        // TODO: sample_shading
                        if pipeline.sample_alpha_to_coverage {
                            checkgl!(sys::glEnable(sys::GL_SAMPLE_ALPHA_TO_COVERAGE));
                        }
                        else {
                            checkgl!(sys::glDisable(sys::GL_SAMPLE_ALPHA_TO_COVERAGE));
                        }
                        if pipeline.sample_alpha_to_one {
                            checkgl!(sys::glEnable(sys::GL_SAMPLE_ALPHA_TO_ONE));
                        }
                        else {
                            checkgl!(sys::glDisable(sys::GL_SAMPLE_ALPHA_TO_ONE));
                        }
                        match pipeline.depth_test {
                            Some(func) => {
                                checkgl!(sys::glEnable(sys::GL_DEPTH_TEST));
                                checkgl!(sys::glDepthFunc(func));
                            },
                            None => checkgl!(sys::glDisable(sys::GL_DEPTH_TEST)),
                        }
                        checkgl!(sys::glDepthMask(pipeline.depth_mask));
                        match pipeline.stencil_test {
                            Some((write_mask,func,fref,comp_mask,sfail,dpfail,dppass)) => {
                                checkgl!(sys::glEnable(sys::GL_STENCIL_TEST));
                                checkgl!(sys::glStencilMask(write_mask));
                                checkgl!(sys::glStencilFunc(func,fref,comp_mask));
                                checkgl!(sys::glStencilOp(sfail,dpfail,dppass));
                            },
                            None => checkgl!(sys::glDisable(sys::GL_STENCIL_TEST)),
                        }
                        match pipeline.logic_op {
                            Some(op) => {
                                checkgl!(sys::glEnable(sys::GL_COLOR_LOGIC_OP));
                                checkgl!(sys::glLogicOp(op));
                            },
                            None => checkgl!(sys::glDisable(sys::GL_COLOR_LOGIC_OP)),
                        }
                        match pipeline.blend {
                            Some((color_op,color_src,color_dst,alpha_op,alpha_src,alpha_dst)) => {
                                checkgl!(sys::glEnable(sys::GL_BLEND));
                                checkgl!(sys::glBlendEquationSeparate(color_op,alpha_op));
                                checkgl!(sys::glBlendFuncSeparate(color_src,color_dst,alpha_src,alpha_dst));
                            },
                            None => checkgl!(sys::glDisable(sys::GL_BLEND)),
                        }
                        checkgl!(sys::glColorMask(pipeline.color_mask.0,pipeline.color_mask.1,pipeline.color_mask.2,pipeline.color_mask.3));
                        checkgl!(sys::glBlendColor(pipeline.blend_constant.x,pipeline.blend_constant.y,pipeline.blend_constant.z,pipeline.blend_constant.w));
                    }
                },
                Command::BindComputePipeline(_pipeline) => {

                },
                Command::BindVertexBuffer(vertex_buffer) => {
                    unsafe { checkgl!(sys::glBindVertexArray(vertex_buffer.vao)) };
                },
                Command::BindIndexBuffer(index_buffer) => {
                    unsafe { checkgl!(sys::glBindBuffer(sys::GL_ELEMENT_ARRAY_BUFFER,index_buffer.ibo)) };
                },
                Command::Draw(vertices,_instances,first_vertex,_first_instance) => {
                    // TODO: first_instance not supported
                    //unsafe { sys::glDrawArraysInstanced(topology,*first_vertex as sys::GLint,*vertices as sys::GLsizei,*instances as sys::GLsizei) };
                    unsafe { checkgl!(sys::glDrawArrays(topology,*first_vertex as sys::GLint,*vertices as sys::GLsizei)) };
                },
                Command::DrawIndexed(indices,_instances,_first_index,_vertex_offset,_first_instance) => {
                    // TODO: first_instance not supported
                    // TODO: connect to bound instance buffer instead of direct access
                    //unsafe { sys::glDrawElementsInstancedBaseVertex(topology,*indices as sys::GLsizei,sys::GL_UNSIGNED_INT,null_mut(),*instances as sys::GLsizei,*vertex_offset as sys::GLint) };
                    unsafe { checkgl!(sys::glDrawElements(topology,*indices as sys::GLsizei,sys::GL_UNSIGNED_INT,null_mut())) };
                },
                Command::SetViewport(r,min_depth,max_depth) => {
                    unsafe {
                        checkgl!(sys::glViewport(r.o.x as sys::GLint,r.o.y as sys::GLint,r.s.x as sys::GLsizei,r.s.y as sys::GLsizei));
                        checkgl!(sys::glDepthRange(*min_depth as sys::GLdouble,*max_depth as sys::GLdouble));
                    }
                },
                Command::SetScissor(r) => {
                    unsafe { checkgl!(sys::glScissor(r.o.x as sys::GLint,r.o.y as sys::GLint,r.s.x as sys::GLsizei,r.s.y as sys::GLsizei)) };
                },
            }
        }
    }
}

impl gpu::CommandBuffer for CommandBuffer {

    type Surface = Surface;
    type GraphicsPipeline = GraphicsPipeline;
    type ComputePipeline = ComputePipeline;
    type VertexBuffer = VertexBuffer;
    type IndexBuffer = IndexBuffer;

    /// Begin the command buffer.
    fn begin(&self) -> Result<(),String> {
        let mut commands = self.commands.borrow_mut();
        commands.clear();
        Ok(())
    }

    /// End the command buffer.
    fn end(&self) -> bool {
        true
    }

    /// Begin render pass.
    fn begin_render_pass(&self,surface: &Surface,_index: usize,r: Rect<i32>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BeginRenderPass(surface.window.id(),r));
    }

    /// End render pass.
    fn end_render_pass(&self) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::EndRenderPass);
    }

    /// Specify current graphics pipeline.
    fn bind_graphics_pipeline(&self,pipeline: &Rc<GraphicsPipeline>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindGraphicsPipeline(Rc::clone(&pipeline)));
    }

    /// Specify current compute pipeline.
    fn bind_compute_pipeline(&self,pipeline: &Rc<ComputePipeline>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindComputePipeline(Rc::clone(&pipeline)));
    }

    /// Specify current vertex buffer.
    fn bind_vertex_buffer(&self,vertex_buffer: &Rc<VertexBuffer>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindVertexBuffer(Rc::clone(&vertex_buffer)));
    }

    /// Specify current index buffer.
    fn bind_index_buffer(&self,index_buffer: &Rc<IndexBuffer>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindIndexBuffer(Rc::clone(&index_buffer)));
    }

    /// Emit vertices.
    fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::Draw(vertex_count,instance_count,first_vertex,first_instance));
    }

    /// Emit indexed vertices.
    fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: usize,first_instance: usize) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::DrawIndexed(index_count,instance_count,first_index,vertex_offset,first_instance));
    }

    /// Specify current viewport transformation.
    fn set_viewport(&self,r: Rect<i32>,min_depth: f32,max_depth: f32) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::SetViewport(r,min_depth,max_depth));
    }

    /// Specify current scissor rectangle.
    fn set_scissor(&self,r: Rect<i32>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::SetScissor(r));
    }
}

impl Drop for CommandBuffer {
    
    fn drop(&mut self) {
    }
}
