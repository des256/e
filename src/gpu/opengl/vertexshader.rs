use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::rc::Rc,
};

/// Vertex shader.
#[derive(Debug)]
pub struct VertexShader {
    pub gpu: Rc<Gpu>,
    pub vs: sys::GLuint,
}

impl gpu::VertexShader for VertexShader {

}

impl Drop for VertexShader {

    fn drop(&mut self) {
        unsafe { checkgl!(sys::glDeleteShader(self.vs)) };
    }
}
