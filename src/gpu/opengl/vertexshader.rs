use {
    crate::*,
    std::rc::Rc,
};

pub struct VertexShader {
    pub system: Rc<System>,
    pub(crate) vs: sys::GLuint,
}

impl Drop for VertexShader {
    fn drop(&mut self) {
        unsafe { sys::glDeleteShader(self.vs) };
    }
}