use {
    crate::*,
    std::rc::Rc,
};

pub struct IndexBuffer {
    pub system: Rc<System>,
    pub(crate) ibo: sys::GLuint,
}

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        unsafe { sys::glDeleteBuffers(1,&self.ibo) };
    }
}