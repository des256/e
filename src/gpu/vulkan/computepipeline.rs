use {
    crate::*,
    std::rc::Rc,
};

#[derive(Debug)]
pub struct ComputePipeline {
    pub(crate) system: Rc<System>,
    pub(crate) vk_pipeline: sys::VkPipeline,
}

impl ComputePipeline {

    pub fn new(system: &Rc<System>) -> Result<ComputePipeline,String> {
        Err("TODO".to_string())
    }   
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {

    }
}
