// Pipeline Layout
// descriptor set layouts: which type of information in the pipeline can be accessed by which shader stage
// push-constant ranges: which push-constants can be accessed by which shader stage

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct PipelineLayout<'system> {
    pub system: &'system System,
    pub(crate) vk_pipeline_layout: sys::VkPipelineLayout,
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.system.gpu.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
