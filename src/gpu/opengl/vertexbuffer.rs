use {
    crate::*,
    std::rc::Rc,
};

pub struct VertexBuffer {
    pub system: Rc<System>,
    pub(crate) vao: sys::GLuint,
    pub(crate) vbo: sys::GLuint,
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::glDeleteBuffers(1,&self.vbo);
            sys::glDeleteVertexArrays(1,&self.vao);
        }
    }
}