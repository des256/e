// E - GPU (Vulkan) - ImageView
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

pub struct ImageView {
    pub image: Rc<Image>,
    pub(crate) vk_imageview: VkImageView,
}

impl Image {

    pub fn get_view(self: &Rc<Self>) -> Option<Rc<ImageView>> {

        let info = VkImageViewCreateInfo {
            sType: VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            image: self.vk_image,
            viewType: VK_IMAGE_VIEW_TYPE_2D,
            format: VK_FORMAT_B8G8R8A8_SRGB,
            components: VkComponentMapping {
                r: VK_COMPONENT_SWIZZLE_IDENTITY,
                g: VK_COMPONENT_SWIZZLE_IDENTITY,
                b: VK_COMPONENT_SWIZZLE_IDENTITY,
                a: VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            subresourceRange: VkImageSubresourceRange {
                aspectMask: VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },            
        };
        let mut vk_imageview = MaybeUninit::uninit();
        match unsafe { vkCreateImageView(self.session.vk_device,&info,null_mut(),vk_imageview.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create Vulkan image view (error {})",code);
                return None;
            },
        }
        
        Some(Rc::new(ImageView {
            image: Rc::clone(self),
            vk_imageview: unsafe { vk_imageview.assume_init() },
        }))

    }
}

impl Drop for ImageView {

    fn drop(&mut self) {
        unsafe { vkDestroyImageView(self.image.session.vk_device,self.vk_imageview,null_mut()) };
    }
}
