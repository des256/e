use {
    crate::*,
    std::rc::Rc,
};

pub struct FragmentShader {
    pub system: Rc<System>,
    pub(crate) fs: sys::GLuint,
}

impl Drop for FragmentShader {
    fn drop(&mut self) {
        unsafe { sys::glDeleteShader(self.fs) };
    }
}