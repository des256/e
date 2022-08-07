use crate::*;

#[cfg(gpu="vulkan")]
use std::{
    ptr::null_mut,
    cell::RefCell,
    mem::MaybeUninit,
};

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

#[cfg(gpu="vulkan")]
pub(crate) struct WindowResources {
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_swapchain: sys::VkSwapchainKHR,
    pub(crate) vk_renderpass: sys::VkRenderPass,
    pub(crate) vk_imageviews: Vec<sys::VkImageView>,
    pub(crate) vk_framebuffers: Vec<sys::VkFramebuffer>,
}

#[cfg(gpu="vulkan")]
impl WindowResources {

    pub fn get_framebuffers(&self) -> Vec<Framebuffer> {

        let mut framebuffers = Vec::<Framebuffer>::new();
        for vk_framebuffer in &self.vk_framebuffers {
            framebuffers.push(Framebuffer {
                owned: false,
                vk_device: self.vk_device,
                vk_framebuffer: *vk_framebuffer,
            });
        }
        framebuffers
    }

    pub fn get_render_pass(&self) -> RenderPass {

        RenderPass {
            owned: false,
            vk_device: self.vk_device,
            vk_renderpass: self.vk_renderpass,
        }            
    }
}

#[cfg(gpu="vulkan")]
impl Drop for WindowResources {
    fn drop(&mut self) {
        for vk_framebuffer in &self.vk_framebuffers {
            unsafe { sys::vkDestroyFramebuffer(self.vk_device,*vk_framebuffer,null_mut()) };
        }
        for vk_imageview in &self.vk_imageviews {
            unsafe { sys::vkDestroyImageView(self.vk_device,*vk_imageview,null_mut()) };
        }
        unsafe { sys::vkDestroySwapchainKHR(self.vk_device,self.vk_swapchain,null_mut()) };
    }
}

#[cfg(gpu="vulkan")]
pub(crate) fn create_window_resources(vk_physical_device: sys::VkPhysicalDevice,vk_device: sys::VkDevice,vk_surface: sys::VkSurfaceKHR,vk_renderpass: sys::VkRenderPass,r: Rect<i32>) -> Option<WindowResources> {

    // get surface capabilities to calculate the extent and image count
    dprintln!("obtaining surface capabilities...");
    let mut capabilities = MaybeUninit::uninit();
    unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_physical_device,vk_surface,capabilities.as_mut_ptr()) };
    let capabilities = unsafe { capabilities.assume_init() };
    let vk_extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
        dprintln!("fixed extent = {} x {}",capabilities.currentExtent.width,capabilities.currentExtent.height);
        capabilities.currentExtent
    }
    else {
        let mut vk_extent = sys::VkExtent2D { width: r.s.x as u32,height: r.s.y as u32 };
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
        dprintln!("specified extent = {} x {}",vk_extent.width,vk_extent.height);
        vk_extent
    };
    let mut image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount != 0) && (image_count > capabilities.maxImageCount) {
        image_count = capabilities.maxImageCount;
    }
    dprintln!("image count = {}",image_count);

    // make sure VK_FORMAT_B8G8R8A8_SRGB is supported (BGRA8UN)
    let mut count = 0u32;
    unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physical_device,vk_surface,&mut count,null_mut()) };
    if count == 0 {
        println!("no formats supported");
        return None;
    }
    let mut formats = vec![sys::VkSurfaceFormatKHR {
        format: 0,
        colorSpace: 0,
    }; count as usize];
    unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(vk_physical_device,vk_surface,&mut count,formats.as_mut_ptr()) };
    let mut format_supported = false;
    for i in 0..formats.len() {
        if (formats[i].format == sys::VK_FORMAT_B8G8R8A8_SRGB) && 
            (formats[i].colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            format_supported = true;
        }
    }
    if !format_supported {
        println!("window does not support ARGB8UN");
        return None;
    }
    dprintln!("format = {}",sys::VK_FORMAT_B8G8R8A8_SRGB);

    // select VK_PRESENT_MODE_MAILBOX_KHR, or otherwise VK_PRESENT_MODE_FIFO_KHR
    let mut count = 0u32;
    unsafe { sys::vkGetPhysicalDeviceSurfacePresentModesKHR(vk_physical_device,vk_surface,&mut count,null_mut()) };
    if count == 0 {
        println!("unable to select present mode");
        return None;
    }
            
    // create swap chain for this window
    dprintln!("creating swap chain...");
    let info = sys::VkSwapchainCreateInfoKHR {
        sType: sys::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        pNext: null_mut(),
        flags: 0,
        surface: vk_surface,
        minImageCount: image_count,
        imageFormat: sys::VK_FORMAT_B8G8R8A8_SRGB,
        imageColorSpace: sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
        imageExtent: vk_extent,
        imageArrayLayers: 1,
        imageUsage: sys::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        imageSharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
        queueFamilyIndexCount: 0,
        pQueueFamilyIndices: null_mut(),
        preTransform: capabilities.currentTransform,
        compositeAlpha: sys::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        presentMode: sys::VK_PRESENT_MODE_IMMEDIATE_KHR,
        clipped: sys::VK_TRUE,
        oldSwapchain: null_mut(),
    };
    let mut vk_swapchain = MaybeUninit::uninit();
    match unsafe { sys::vkCreateSwapchainKHR(vk_device,&info,null_mut(),vk_swapchain.as_mut_ptr()) } {
        sys::VK_SUCCESS => { },
        code => {
            println!("unable to create swap chain (error {})",code);
            return None;
        },
    }
    let vk_swapchain = unsafe { vk_swapchain.assume_init() };

    // get swapchain images
    dprintln!("getting swap chain images...");
    let mut count = 0u32;
    match unsafe { sys::vkGetSwapchainImagesKHR(vk_device,vk_swapchain,&mut count,null_mut()) } {
        sys::VK_SUCCESS => { },
        code => {
            println!("unable to get swap chain image count (error {})",code);
            // TODO: unwind
            return None;
        }
    }
    let mut vk_images = vec![null_mut() as sys::VkImage; count as usize];
    match unsafe { sys::vkGetSwapchainImagesKHR(vk_device,vk_swapchain,&mut count,vk_images.as_mut_ptr()) } {
        sys::VK_SUCCESS => { },
        code => {
            println!("unable to get swap chain images (error {})",code);
            // TODO: unwind
            return None;
        },
    }

    // create image views for the swapchain images
    dprintln!("creating image views onto swap chain images...");
    let mut vk_imageviews = Vec::<sys::VkImageView>::new();
    for vk_image in &vk_images {

        let info = sys::VkImageViewCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            image: *vk_image,
            viewType: sys::VK_IMAGE_VIEW_TYPE_2D,
            format: sys::VK_FORMAT_B8G8R8A8_SRGB,
            components: sys::VkComponentMapping {
                r: sys::VK_COMPONENT_SWIZZLE_IDENTITY,
                g: sys::VK_COMPONENT_SWIZZLE_IDENTITY,
                b: sys::VK_COMPONENT_SWIZZLE_IDENTITY,
                a: sys::VK_COMPONENT_SWIZZLE_IDENTITY,
            },
            subresourceRange: sys::VkImageSubresourceRange {
                aspectMask: sys::VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel: 0,
                levelCount: 1,
                baseArrayLayer: 0,
                layerCount: 1,
            },
        };
        let mut vk_imageview = MaybeUninit::uninit();
        match unsafe { sys::vkCreateImageView(vk_device,&info,null_mut(),vk_imageview.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create image view (error {})",code);
                // TODO: unwind
                return None;
            }
        }
        vk_imageviews.push(unsafe { vk_imageview.assume_init() });
    }

    // create framebuffers for the image views
    dprintln!("creating frame buffers for the image views...");
    let mut vk_framebuffers = Vec::<sys::VkFramebuffer>::new();
    for vk_imageview in &vk_imageviews {

        let info = sys::VkFramebufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            renderPass: vk_renderpass,
            attachmentCount: 1,
            pAttachments: &*vk_imageview,
            width: vk_extent.width,
            height: vk_extent.height,
            layers: 1,
        };
        let mut vk_framebuffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateFramebuffer(vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create framebuffer (error {})",code);
                // TODO: unwind
                return None;
            }
        }
        vk_framebuffers.push(unsafe { vk_framebuffer.assume_init() });
    }

    dprintln!("success.");

    Some(WindowResources {
        vk_device: vk_device,
        vk_swapchain: vk_swapchain,
        vk_renderpass: vk_renderpass,
        vk_imageviews: vk_imageviews,
        vk_framebuffers: vk_framebuffers,
    })
}

pub struct Window {
    #[doc(hidden)]
    pub(crate) xcb_connection: *mut sys::xcb_connection_t,
#[doc(hidden)]
    pub(crate) xcb_window: sys::xcb_window_t,
#[cfg(gpu="vulkan")]
    pub(crate) vk_instance: sys::VkInstance,
#[cfg(gpu="vulkan")]
    pub(crate) vk_physical_device: sys::VkPhysicalDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_device: sys::VkDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_surface: sys::VkSurfaceKHR,
#[cfg(gpu="vulkan")]
    pub(crate) resources: RefCell<WindowResources>,
}

impl Window {

    /// Get WindowID for this window.
    pub fn id(&self) -> WindowId {
        self.xcb_window as WindowId
    }

    pub fn get_framebuffers(&self) -> Vec<Framebuffer> {
        self.resources.borrow().get_framebuffers()
    }

    pub fn get_render_pass(&self) -> RenderPass {
        self.resources.borrow().get_render_pass()
    }

    pub fn acquire_next(&self,semaphore: &Semaphore) -> usize {
#[cfg(gpu="vulkan")]        
        {
            let mut index = 0u32;
            unsafe { sys::vkAcquireNextImageKHR(self.vk_device,self.resources.borrow().vk_swapchain,0xFFFFFFFFFFFFFFFF,semaphore.vk_semaphore,null_mut(),&mut index) };
            index as usize
        }

#[cfg(not(gpu="vulkan"))]
        0
    }

    pub fn present(&self,index: usize,semaphore: &Semaphore) {
        let image_index = index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.resources.borrow().vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.vk_queue,&info) };
    }

    pub fn rebuild_resources(&mut self,r: Rect<i32>) {

#[cfg(gpu="vulkan")]
        {
            let vk_renderpass = self.resources.borrow().vk_renderpass;
            if let Some(resources) = create_window_resources(self.vk_physical_device,self.vk_device,self.vk_surface,vk_renderpass,r) {
                self.resources.replace(resources);
            }
        }
    }
}

/*
impl Window {self.resources.borrow().vk_renderpass
    pub(crate) fn handle_event(&self,event: Event) {
        if let Event::Configure(r) = &event {
            // When resizing, X seems to return a rectangle with the initial
            // origin as specified during window creation. But when moving, X
            // seems to correctly update the origin coordinates.
            // Not sure what to make of this, but in order to get the actual
            // rectangle, this seems to work:
            let old_r = self.r.get();
            if r.s != old_r.s {
                self.r.set(Rect { o: old_r.o,s: r.s, });
            }
            else {
                self.r.set(*r);
            }
        }
        if let Some(handler) = &*(self.handler).borrow() {
            (handler)(event);
        }
    }

    pub fn set_handler<T: Fn(Event) + 'static>(&self,handler: T) {
        *self.handler.borrow_mut() = Some(Box::new(handler));
    }

    pub fn clear_handler(&self) {
        *self.handler.borrow_mut() = None;
    }

    pub fn show(&self) {
        unsafe {
            xcb_map_window(self.screen.system.xcb_connection,self.xcb_window as u32);
            xcb_flush(self.screen.system.xcb_connection);
        }
    }

    pub fn hide(&self) {
        unsafe {
            xcb_unmap_window(self.screen.system.xcb_connection,self.xcb_window as u32);
            xcb_flush(self.screen.system.xcb_connection);
        }
    }

    pub fn set_rect(&self,r: &Rect<i32>) {
        let values = xcb_configure_window_value_list_t {
            x: r.o.x as i32,
            y: r.o.y as i32,
            width: r.s.x as u32,
            height: r.s.y as u32,
            border_width: 0,
            sibling: 0,
            stack_mode: 0,
        };
        unsafe { xcb_configure_window(
            self.screen.system.xcb_connection,
            self.xcb_window as u32,
            XCB_CONFIG_WINDOW_X as u16 |
                XCB_CONFIG_WINDOW_Y as u16 |
                XCB_CONFIG_WINDOW_WIDTH as u16 |
                XCB_CONFIG_WINDOW_HEIGHT as u16,
            &values as *const xcb_configure_window_value_list_t as *const std::os::raw::c_void
        ) };
    }
}
*/

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
#[cfg(gpu="vulkan")]
            sys::vkDestroySurfaceKHR(self.vk_instance,self.vk_surface,null_mut());

            sys::xcb_unmap_window(self.xcb_connection,self.xcb_window as u32);
            sys::xcb_destroy_window(self.xcb_connection,self.xcb_window as u32);
        }
    }
}
