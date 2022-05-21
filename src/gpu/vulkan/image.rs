// E - GPU (Vulkan) - Image
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        ptr::null_mut,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct Image {
    pub session: Rc<Session>,
    pub(crate) owned: bool,
    pub(crate) vk_image: VkImage,
}

// TODO: implement creation from session, like for texture mapping and such

impl SwapChain {

    pub fn get_images(&self) -> Vec<Rc<Image>> {

        let mut count = 0u32;
        match unsafe { vkGetSwapchainImagesKHR(self.session.vk_device,self.vk_swapchain,&mut count,null_mut()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to get swap chain image count (error {})",code);
                return Vec::new();
            }
        }
        let mut vk_images = vec![null_mut() as VkImage; count as usize];
        match unsafe { vkGetSwapchainImagesKHR(self.session.vk_device,self.vk_swapchain,&mut count,vk_images.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to get swap chain images (error {})",code);
                unsafe { vkDestroySwapchainKHR(self.session.vk_device,self.vk_swapchain,null_mut()) };
                return Vec::new();
            },
        }
        let mut images = Vec::<Rc<Image>>::new();
        for vk_image in &vk_images {
            images.push(Rc::new(Image {
                session: Rc::clone(&self.session),
                owned: false,
                vk_image: *vk_image,
            }));
        }
        images
    }
}

impl Drop for Image {

    fn drop(&mut self) {
        if self.owned {
            unsafe { vkDestroyImage(self.session.vk_device,self.vk_image,null_mut()) };
        }
    }
}
