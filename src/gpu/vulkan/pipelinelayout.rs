// Pipeline Layout
// descriptor set layouts: which type of information in the pipeline can be accessed by which shader stage
// push-constant ranges: which push-constants can be accessed by which shader stage

use {
    crate::*,
    std::{
        ptr::null_mut,
        rc::Rc,
        mem::MaybeUninit,
    },
};

#[derive(Debug)]
pub struct PipelineLayout {
    pub system: Rc<System>,
    pub(crate) vk_pipeline_layout: sys::VkPipelineLayout,
}

impl PipelineLayout {

    pub fn new(system: &Rc<System>) -> Result<PipelineLayout,String> {
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
        match unsafe { sys::vkCreatePipelineLayout(system.gpu_system.vk_device,&info,null_mut(),vk_pipeline_layout.as_mut_ptr()) } {
            sys::VK_SUCCESS => Ok(PipelineLayout {
                system: Rc::clone(system),
                vk_pipeline_layout: unsafe { vk_pipeline_layout.assume_init() },
            }),
            code => Err(format!("Unable to create pipeline layout ({})",vk_code_to_string(code))),
        }
    }
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipelineLayout(self.system.gpu_system.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
