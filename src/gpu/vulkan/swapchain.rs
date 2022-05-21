// E - GPU (Vulkan) - SwapChain
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

pub struct SwapChain {
    pub session: Rc<Session>,
    pub window: Rc<Window>,
    pub extent: Vec2<usize>,
    pub(crate) vk_swapchain: VkSwapchainKHR,
}

impl Session {

    pub fn create_swapchain(self: &Rc<Self>,window: &Rc<Window>) -> Option<SwapChain> {

        let mut capabilities = MaybeUninit::uninit();
        unsafe { vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.gpu.vk_physical_device,window.vk_surface,capabilities.as_mut_ptr()) };
        let capabilities = unsafe { capabilities.assume_init() };
        let vk_extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
            capabilities.currentExtent
        }
        else {
            let mut vk_extent = VkExtent2D { width: window.r.get().s.x as u32,height: window.r.get().s.y as u32 };
            if vk_extent.width < capabilities.minImageExtent.width {
                vk_extent.width = capabilities.minImageExtent.width;
            }
            if vk_extent.height < capabilities.minImageExtent.height {
                vk_extent.height = capabilities.minImageExtent.height;
            }
            if vk_extent.width > capabilities.maxImageExtent.width {
                vk_extent.width = capabilities.maxImageExtent.width;
            }
            if vk_extent.height > capabilities.maxImageExtent.height {
                vk_extent.height = capabilities.maxImageExtent.height;
            }
            vk_extent
        };
        println!("extent = {},{}",vk_extent.width,vk_extent.height);

        let mut image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount != 0) && (image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }
        println!("image_count = {}",image_count);

        let mut count = 0u32;
        unsafe { vkGetPhysicalDeviceSurfaceFormatsKHR(self.gpu.vk_physical_device,window.vk_surface,&mut count,null_mut()) };
        if count == 0 {
#[cfg(feature="debug_output")]
            println!("No formats supported.");
            return None;
        }
        let mut formats = vec![VkSurfaceFormatKHR {
            format: 0,
            colorSpace: 0,
        }; count as usize];
        unsafe { vkGetPhysicalDeviceSurfaceFormatsKHR(self.gpu.vk_physical_device,window.vk_surface,&mut count,formats.as_mut_ptr()) };
        let mut format_supported = false;
        for i in 0..formats.len() {
            if (formats[i].format == VK_FORMAT_B8G8R8A8_SRGB) && 
               (formats[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format_supported = true;
            }
        }
        if !format_supported {
#[cfg(feature="debug_output")]
            println!("Format ARGB8 not supported.");
            return None;
        }
        println!("format = {}",VK_FORMAT_B8G8R8A8_SRGB);

        let mut count = 0u32;
        unsafe { vkGetPhysicalDeviceSurfacePresentModesKHR(self.gpu.vk_physical_device,window.vk_surface,&mut count,null_mut()) };
        if count == 0 {
#[cfg(feature="debug_output")]
            println!("No present modes supported.");
            unsafe { vkDestroySurfaceKHR(self.gpu.system.vk_instance,window.vk_surface,null_mut()) };
            return None;
        }
        let mut modes = vec![0 as VkPresentModeKHR; count as usize];
        unsafe { vkGetPhysicalDeviceSurfacePresentModesKHR(self.gpu.vk_physical_device,window.vk_surface,&mut count,modes.as_mut_ptr()) };
        let mut present_mode = VK_PRESENT_MODE_FIFO_KHR;
        for mode in &modes {
            if *mode == VK_PRESENT_MODE_MAILBOX_KHR {
                present_mode = *mode;
            }
        }
        println!("present_mode = {}",present_mode);

        let info = VkSwapchainCreateInfoKHR {
            sType: VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: null_mut(),
            flags: 0,
            surface: window.vk_surface,
            minImageCount: image_count,
            imageFormat: VK_FORMAT_B8G8R8A8_SRGB,
            imageColorSpace: VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            imageExtent: vk_extent,
            imageArrayLayers: 1,
            imageUsage: VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode: VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
            preTransform: capabilities.currentTransform,
            compositeAlpha: VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode: present_mode,
            clipped: VK_TRUE,
            oldSwapchain: null_mut(),
        };
        let mut vk_swapchain = MaybeUninit::uninit();
        match unsafe { vkCreateSwapchainKHR(self.vk_device,&info,null_mut(),vk_swapchain.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create swap chain (error {})",code);
                return None;
            },
        }
        let vk_swapchain = unsafe { vk_swapchain.assume_init() };

        Some(SwapChain {
            session: Rc::clone(self),
            window: Rc::clone(window),
            extent: vec2![vk_extent.width as usize,vk_extent.height as usize],
            vk_swapchain: vk_swapchain,
        })
    }
}

impl SwapChain {

    pub fn next(&self,semaphore: &Semaphore) -> usize {
        let mut image_index = 0u32;
        unsafe { vkAcquireNextImageKHR(self.session.vk_device,self.vk_swapchain,0xFFFFFFFFFFFFFFFF,semaphore.vk_semaphore,null_mut(),&mut image_index) };
        image_index as usize
    }
}

impl Drop for SwapChain {

    fn drop(&mut self) {
        unsafe { vkDestroySwapchainKHR(self.session.vk_device,self.vk_swapchain,null_mut()) };
    }
}
