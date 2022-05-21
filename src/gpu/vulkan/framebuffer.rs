use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct Framebuffer {
    pub imageview: Rc<ImageView>,
    pub size: Vec2<usize>,
    pub render_pass: Rc<RenderPass>,
    pub(crate) vk_framebuffer: VkFramebuffer,
}

impl ImageView {

    pub fn create_framebuffer(self: &Rc<Self>,size: Vec2<usize>,render_pass: &Rc<RenderPass>) -> Option<Rc<Framebuffer>> {

        let info = VkFramebufferCreateInfo {
            sType: VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            renderPass: render_pass.vk_render_pass,
            attachmentCount: 1,
            pAttachments: &self.vk_imageview,
            width: size.x as u32,
            height: size.y as u32,
            layers: 1,
        };
        let mut vk_framebuffer = MaybeUninit::uninit();
        match unsafe { vkCreateFramebuffer(self.image.session.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
        #[cfg(feature="debug_output")]
                println!("Unable to create Vulkan frame buffer (error {})",code);
                return None;
            }
        }
        Some(Rc::new(Framebuffer {
            imageview: Rc::clone(self),
            size: size,
            render_pass: Rc::clone(render_pass),
            vk_framebuffer: unsafe { vk_framebuffer.assume_init() },
        }))
    }
}

impl Drop for Framebuffer {

    fn drop(&mut self) {
        unsafe { vkDestroyFramebuffer(self.imageview.image.session.vk_device,self.vk_framebuffer,null_mut()) };
    }
}
