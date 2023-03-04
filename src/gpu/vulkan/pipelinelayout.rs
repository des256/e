// Pipeline Layout
// descriptor set layouts: which type of information in the pipeline can be accessed by which shader stage
// push-constant ranges: which push-constants can be accessed by which shader stage

use {
    super::*,
    crate::gpu,
    std::{
        ptr::null_mut,
        rc::Rc,
    },
};

#[derive(Debug)]
pub struct PipelineLayout {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_pipeline_layout: sys::VkPipelineLayout,
}

impl gpu::PipelineLayout for PipelineLayout {
    
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.gpu.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
