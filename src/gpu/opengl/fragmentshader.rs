use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct FragmentShader {
    pub gpu: Rc<Gpu>,
    pub fs: sys::GLuint,
}

impl gpu::FragmentShader for FragmentShader {

}

impl Drop for FragmentShader {

    fn drop(&mut self) {
        unsafe { checkgl!(sys::glDeleteShader(self.fs)) };
    }
}
