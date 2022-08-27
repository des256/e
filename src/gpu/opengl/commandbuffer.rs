use {
    crate::*,
    std::{
        rc::Rc,
        cell::RefCell,
        ptr::null_mut,
    },
};

pub enum Command {
    BeginRenderPass(WindowId,Rect<isize,usize>),
    EndRenderPass,
    BindGraphicsPipeline(Rc<GraphicsPipeline>),
    BindVertexBuffer(Rc<VertexBuffer>),
    BindIndexBuffer(Rc<IndexBuffer>),
    Draw(usize,usize,usize,usize),
    DrawIndexed(usize,usize,usize,isize,usize),
    SetViewport(Hyper<f32,f32>),
    SetScissor(Rect<isize,usize>),
}

pub struct CommandBuffer {
    pub system: Rc<System>,
    pub commands: RefCell<Vec<Command>>,
}

impl CommandBuffer {

    /// Begin a command buffer.
    pub fn begin(&self) -> bool {
        let mut commands = self.commands.borrow_mut();
        commands.clear();
        true
    }

    /// Begin render pass on a rectangle on one of window's framebuffers.
    pub fn begin_render_pass(&self,window: &Window,_index: usize,r: Rect<isize,usize>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BeginRenderPass(window.id(),r));
    }

    /// End render pass.
    pub fn end_render_pass(&self) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::EndRenderPass);
    }

    /// Specify current graphics pipeline.
    pub fn bind_pipeline(&self,graphics_pipeline: &Rc<GraphicsPipeline>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindGraphicsPipeline(Rc::clone(graphics_pipeline)));
    }

    /// Specify current vertex buffer.
    pub fn bind_vertex_buffer(&self,vertex_buffer: &Rc<VertexBuffer>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindVertexBuffer(Rc::clone(vertex_buffer)));
    }

    /// Specify current index buffer.
    pub fn bind_index_buffer(&self,index_buffer: &Rc<IndexBuffer>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::BindIndexBuffer(Rc::clone(index_buffer)));
    }

    /// Draw primitives.
    pub fn draw(&self,vertex_count: usize,instance_count: usize,first_vertex: usize, first_instance: usize) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::Draw(vertex_count,instance_count,first_vertex,first_instance));
    }

    /// Draw indexed primitives.
    pub fn draw_indexed(&self,index_count: usize,instance_count: usize,first_index: usize,vertex_offset: isize,first_instance: usize) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::DrawIndexed(index_count,instance_count,first_index,vertex_offset,first_instance));
    }

    /// Specify current viewport transformation.
    pub fn set_viewport(&self,h: Hyper<f32,f32>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::SetViewport(h));
    }

    /// Specify current scissor rectangle.
    pub fn set_scissor(&self,r: Rect<isize,usize>) {
        let mut commands = self.commands.borrow_mut();
        commands.push(Command::SetScissor(r));
    }

    /// Finish the command buffer.
    pub fn end(&self) -> bool {
        // actually perform the commands here
        true
    }

    pub(crate) fn execute(&self) {
        let mut topology: sys::GLenum = sys::GL_POINTS;
        let commands = self.commands.borrow();
        for command in commands.iter() {
            match command {
                Command::BeginRenderPass(window_id,_) => {
                    unsafe { sys::glXMakeCurrent(
                        self.system.xdisplay,
                        *window_id,
                        self.system.gpu.glx_context,
                    ) };
                },
                Command::EndRenderPass => {
                },
                Command::BindGraphicsPipeline(graphics_pipeline) => {
                    topology = graphics_pipeline.topology;
                    // TODO: primitive restart
                    // TODO: patch control points
                    unsafe {
                        sys::glUseProgram(graphics_pipeline.shader_program);
                        if graphics_pipeline.depth_clamp {
                            sys::glEnable(sys::GL_DEPTH_CLAMP);
                        }
                        else {
                            sys::glDisable(sys::GL_DEPTH_CLAMP);
                        }
                        if graphics_pipeline.rasterizer_discard {
                            sys::glEnable(sys::GL_RASTERIZER_DISCARD);
                        }
                        else {
                            sys::glDisable(sys::GL_RASTERIZER_DISCARD);
                        }
                        sys::glPolygonMode(sys::GL_FRONT_AND_BACK,graphics_pipeline.polygon_mode);
                        match graphics_pipeline.cull_mode {
                            Some((cull_face,front_face)) => {
                                sys::glEnable(sys::GL_CULL_FACE);
                                sys::glCullFace(cull_face);
                                sys::glFrontFace(front_face);    
                            },
                            None => sys::glDisable(sys::GL_CULL_FACE),
                        }
                        match graphics_pipeline.polygon_offset {
                            Some((s,c)) => {
                                sys::glEnable(sys::GL_POLYGON_OFFSET_POINT);
                                sys::glEnable(sys::GL_POLYGON_OFFSET_LINE);
                                sys::glEnable(sys::GL_POLYGON_OFFSET_FILL);
                                sys::glPolygonOffset(s,c);    
                            },
                            None => {
                                sys::glDisable(sys::GL_POLYGON_OFFSET_POINT);
                                sys::glDisable(sys::GL_POLYGON_OFFSET_LINE);
                                sys::glDisable(sys::GL_POLYGON_OFFSET_FILL);    
                            },
                        }
                        sys::glLineWidth(graphics_pipeline.line_width);
                        // TODO: rasterization_samples
                        // TODO: sample_shading
                        if graphics_pipeline.sample_alpha_to_coverage {
                            sys::glEnable(sys::GL_SAMPLE_ALPHA_TO_COVERAGE);
                        }
                        else {
                            sys::glDisable(sys::GL_SAMPLE_ALPHA_TO_COVERAGE);
                        }
                        if graphics_pipeline.sample_alpha_to_one {
                            sys::glEnable(sys::GL_SAMPLE_ALPHA_TO_ONE);
                        }
                        else {
                            sys::glDisable(sys::GL_SAMPLE_ALPHA_TO_ONE);
                        }
                        match graphics_pipeline.depth_test {
                            Some(func) => {
                                sys::glEnable(sys::GL_DEPTH_TEST);
                                sys::glDepthFunc(func);    
                            },
                            None => sys::glDisable(sys::GL_DEPTH_TEST),
                        }
                        sys::glDepthMask(graphics_pipeline.depth_mask);
                        match graphics_pipeline.stencil_test {
                            Some((write_mask,func,fref,comp_mask,sfail,dpfail,dppass)) => {
                                sys::glEnable(sys::GL_STENCIL_TEST);
                                sys::glStencilMask(write_mask);
                                sys::glStencilFunc(func,fref,comp_mask);
                                sys::glStencilOp(sfail,dpfail,dppass);
                            },
                            None => sys::glDisable(sys::GL_STENCIL_TEST),
                        }
                        match graphics_pipeline.logic_op {
                            Some(op) => {
                                sys::glEnable(sys::GL_COLOR_LOGIC_OP);
                                sys::glLogicOp(op);
                            },
                            None => sys::glDisable(sys::GL_COLOR_LOGIC_OP),
                        }
                        match graphics_pipeline.blend {
                            Some((color_op,color_src,color_dst,alpha_op,alpha_src,alpha_dst)) => {
                                sys::glEnable(sys::GL_BLEND);
                                sys::glBlendEquationSeparate(color_op,alpha_op);
                                sys::glBlendFuncSeparate(color_src,color_dst,alpha_src,alpha_dst);
                            },
                            None => sys::glDisable(sys::GL_BLEND),
                        }
                        sys::glColorMask(graphics_pipeline.color_mask.0,graphics_pipeline.color_mask.1,graphics_pipeline.color_mask.2,graphics_pipeline.color_mask.3);
                        sys::glBlendColor(graphics_pipeline.blend_constant.r,graphics_pipeline.blend_constant.g,graphics_pipeline.blend_constant.b,graphics_pipeline.blend_constant.a);
                    }
                },
                Command::BindVertexBuffer(vertex_buffer) => {
                    unsafe { sys::glBindVertexArray(vertex_buffer.vao) };
                },
                Command::BindIndexBuffer(index_buffer) => {
                    unsafe { sys::glBindBuffer(sys::GL_ELEMENT_ARRAY_BUFFER,index_buffer.ibo) };
                },
                Command::Draw(vertices,_instances,first_vertex,_first_instance) => {
                    // TODO: first_instance not supported
                    //unsafe { sys::glDrawArraysInstanced(topology,*first_vertex as sys::GLint,*vertices as sys::GLsizei,*instances as sys::GLsizei) };
                    unsafe { sys::glDrawArrays(topology,*first_vertex as sys::GLint,*vertices as sys::GLsizei) };
                },
                Command::DrawIndexed(indices,_instances,_first_index,_vertex_offset,_first_instance) => {
                    // TODO: first_instance not supported
                    // TODO: connect to bound instance buffer instead of direct access
                    //unsafe { sys::glDrawElementsInstancedBaseVertex(topology,*indices as sys::GLsizei,sys::GL_UNSIGNED_INT,null_mut(),*instances as sys::GLsizei,*vertex_offset as sys::GLint) };
                    unsafe { sys::glDrawElements(topology,*indices as sys::GLsizei,sys::GL_UNSIGNED_INT,null_mut()) };
                },
                Command::SetViewport(viewport) => {
                    unsafe {
                        sys::glViewport(viewport.o.x as sys::GLint,viewport.o.y as sys::GLint,viewport.s.x as sys::GLsizei,viewport.s.y as sys::GLsizei);
                        sys::glDepthRange(viewport.o.z as sys::GLdouble,(viewport.o.z + viewport.s.z) as sys::GLdouble);
                    }
                },
                Command::SetScissor(r) => {
                    unsafe { sys::glScissor(r.o.x as sys::GLint,r.o.y as sys::GLint,r.s.x as sys::GLsizei,r.s.y as sys::GLsizei) };
                },
            }
        }
    }
}
