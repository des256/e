use {
    super::*,
    crate::gpu,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct PipelineLayout {
    pub gpu: Rc<Gpu>,
}

impl gpu::PipelineLayout for PipelineLayout {
    
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
    }
}
