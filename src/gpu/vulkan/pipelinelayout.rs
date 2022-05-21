// E - GPU (Vulkan) - PipelineLayout
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct PipelineLayout {
    pub session: Rc<Session>,
    pub(crate) vk_pipeline_layout: VkPipelineLayout,
}

impl Session {

    pub fn create_pipeline_layout(self: &Rc<Self>) -> Option<Rc<PipelineLayout>> {

        let info = VkPipelineLayoutCreateInfo {
            sType: VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            setLayoutCount: 0,
            pSetLayouts: null_mut(),
            pushConstantRangeCount: 0,
            pPushConstantRanges: null_mut(),
        };
        let mut vk_pipeline_layout = MaybeUninit::uninit();
        match unsafe { vkCreatePipelineLayout(self.vk_device,&info,null_mut(),vk_pipeline_layout.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create Vulkan pipeline layout (error {})",code);
                return None;
            },
        }

        Some(Rc::new(PipelineLayout {
            session: Rc::clone(self),
            vk_pipeline_layout: unsafe { vk_pipeline_layout.assume_init() },
        }))
    }
}

impl Drop for PipelineLayout {

    fn drop(&mut self) {
        unsafe { vkDestroyPipelineLayout(self.session.vk_device,self.vk_pipeline_layout,null_mut()) };
    }
}
