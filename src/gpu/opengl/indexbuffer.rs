use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct IndexBuffer {
    pub gpu: Rc<Gpu>,
    pub ibo: sys::GLuint,
}

impl gpu::IndexBuffer for IndexBuffer {

}

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        unsafe { checkgl!(sys::glDeleteBuffers(1,&self.ibo)) };
    }
}