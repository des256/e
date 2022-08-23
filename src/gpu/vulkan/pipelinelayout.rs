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

impl<'system> System {

    /// Create a pipeline layout.
    pub fn create_pipeline_layout(&self) -> Option<PipelineLayout> {

        let info = sys::VkPipelineLayoutCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            setLayoutCount: 0,
            pSetLayouts: null_mut(),
            pushConstantRangeCount: 0,
            pPushConstantRanges: null_mut(),
        };
        let mut vk_pipeline_layout = MaybeUninit::uninit();
        match unsafe { sys::vkCreatePipelineLayout(self.vk_device,&info,null_mut(),vk_pipeline_layout.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create pipeline layout (error {})",code);
                return None;
            },
        }
        Some(PipelineLayout {
            system: &self,
            vk_pipeline_layout: unsafe { vk_pipeline_layout.assume_init() },
        })
    }
}

impl<'system> Drop for PipelineLayout<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.system.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
