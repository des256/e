use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct VertexBuffer {
    pub gpu: Rc<Gpu>,
    pub vbo: sys::GLuint,
    pub vao: sys::GLuint,
}

impl gpu::VertexBuffer for VertexBuffer {
    
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe {
            checkgl!(sys::glDeleteBuffers(1,&self.vbo));
            checkgl!(sys::glDeleteVertexArrays(1,&self.vao));
        }
    }
}