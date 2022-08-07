// RenderPass
// attachments: image descriptions and load/store indications of each attachment
// subpasses: subpass? descriptions
// dependencies: how subpasses depend on each other

use {
    crate::*,
    std::ptr::null_mut,
};

pub struct RenderPass {
    pub(crate) owned: bool,
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_renderpass: sys::VkRenderPass,
}

impl Drop for RenderPass {

    fn drop(&mut self) {
        if self.owned {
            unsafe { sys::vkDestroyRenderPass(self.vk_device,self.vk_renderpass,null_mut()) };
        }
    }
}
