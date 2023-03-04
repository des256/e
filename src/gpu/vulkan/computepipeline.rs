use {
    super::*,
    crate::gpu,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct ComputePipeline {
    pub gpu: Rc<Gpu>,
    pub vk_pipeline: sys::VkPipeline,
}

impl gpu::ComputePipeline for ComputePipeline {

}

impl Drop for ComputePipeline {
    fn drop(&mut self) {

    }
}
