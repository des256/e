use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct RenderPass<'system,'gpu,'screen> {
    pub screen: &'screen Screen<'system,'gpu>,
    pub(crate) vk_render_pass: sys::VkRenderPass,
}

impl<'system,'gpu> Screen<'system,'gpu> {

    pub fn create_render_pass(&self) -> Option<RenderPass> {

        let info = sys::VkRenderPassCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            attachmentCount: 1,
            pAttachments: &sys::VkAttachmentDescription {
                flags: 0,
                format: sys::VK_FORMAT_B8G8R8A8_SRGB,
                samples: sys::VK_SAMPLE_COUNT_1_BIT,
                loadOp: sys::VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp: sys::VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp: sys::VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp: sys::VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout: sys::VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout: sys::VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            },
            subpassCount: 1,
            pSubpasses: &sys::VkSubpassDescription {
                flags: 0,
                pipelineBindPoint: sys::VK_PIPELINE_BIND_POINT_GRAPHICS,
                inputAttachmentCount: 0,
                pInputAttachments: null_mut(),
                colorAttachmentCount: 1,
                pColorAttachments: &sys::VkAttachmentReference {
                    attachment: 0,
                    layout: sys::VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                },
                pResolveAttachments: null_mut(),
                pDepthStencilAttachment: null_mut(),
                preserveAttachmentCount: 0,
                pPreserveAttachments: null_mut(),
            },
            dependencyCount: 1,
            pDependencies: &sys::VkSubpassDependency {
                srcSubpass: sys::VK_SUBPASS_EXTERNAL as u32,
                dstSubpass: 0,
                srcStageMask: sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                dstStageMask: sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                srcAccessMask: 0,
                dstAccessMask: sys::VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dependencyFlags: 0,
            },
        };
        let mut vk_render_pass = MaybeUninit::uninit();
        match unsafe { sys::vkCreateRenderPass(self.vk_device,&info,null_mut(),vk_render_pass.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("Unable to create Vulkan render pass (error {})",code);
                return None;
            }
        }

        Some(RenderPass {
            screen: &self,
            vk_render_pass: unsafe { vk_render_pass.assume_init() },
        })
    }
}

impl<'system,'gpu,'screen> Drop for RenderPass<'system,'gpu,'screen> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyRenderPass(self.screen.vk_device,self.vk_render_pass,null_mut()) };
    }
}
