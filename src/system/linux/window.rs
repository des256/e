use crate::*;

#[cfg(gpu="vulkan")]
use std::{
    ptr::null_mut,
    mem::{
        MaybeUninit,
        transmute,
    },
};

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

pub struct Window<'system> {
    pub system: &'system System,
#[doc(hidden)]
    pub(crate) xcb_window: sys::xcb_window_t,
#[cfg(gpu="vulkan")]
    pub(crate) vk_surface: sys::VkSurfaceKHR,
#[cfg(gpu="vulkan")]
    pub(crate) vk_render_pass: sys::VkRenderPass,
#[cfg(gpu="vulkan")]
    pub(crate) vk_swapchain: sys::VkSwapchainKHR,
#[cfg(gpu="vulkan")]
    pub(crate) vk_image_views: Vec<sys::VkImageView>,
#[cfg(gpu="vulkan")]
    pub(crate) vk_framebuffers: Vec<sys::VkFramebuffer>,
}

impl System {

#[cfg(gpu="vulkan")]
    fn create_swapchain_resources(
        &self,
        vk_surface: sys::VkSurfaceKHR,
        vk_render_pass: sys::VkRenderPass,
        r: Rect<i32>,
    ) -> Option<(
        sys::VkSwapchainKHR,
        Vec<sys::VkImageView>,
        Vec<sys::VkFramebuffer>,
    )> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::uninit();
        unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.vk_physical_device,
            vk_surface,
            capabilities.as_mut_ptr(),
        ) };
        let capabilities = unsafe { capabilities.assume_init() };

        let extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
            dprintln!("fixed extent = {} x {}",capabilities.currentExtent.width,capabilities.currentExtent.height);
            vec2!(capabilities.currentExtent.width,capabilities.currentExtent.height)
        }
        else {
            let mut extent: Vec2<u32> = vec2!(r.s.x as u32,r.s.y as u32);
            if extent.x < capabilities.minImageExtent.width {
                extent.x = capabilities.minImageExtent.width;
            }
            if extent.y < capabilities.minImageExtent.height {
                extent.y = capabilities.minImageExtent.height;
            }
            if extent.x > capabilities.maxImageExtent.width {
                extent.x = capabilities.maxImageExtent.width;
            }
            if extent.y > capabilities.maxImageExtent.height {
                extent.y = capabilities.maxImageExtent.height;
            }
            dprintln!("specified extent = {}",extent);
            extent
        };
        let mut image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount != 0) && (image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }
        dprintln!("image count = {}",image_count);

        // make sure VK_FORMAT_B8G8R8A8_SRGB is supported (BGRA8UN)
        let mut count = 0u32;
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.vk_physical_device,
            vk_surface,
            &mut count,
            null_mut(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get surface formats (error {})",code);
                return None;
            }
        }
        let mut formats = vec![MaybeUninit::<sys::VkSurfaceFormatKHR>::uninit(); count as usize];
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.vk_physical_device,
            vk_surface,
            &mut count,
            formats.as_mut_ptr() as *mut sys::VkSurfaceFormatKHR,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get surface formats (error {})",code);
                return None;
            }
        }
        let formats = unsafe { transmute::<_,Vec<sys::VkSurfaceFormatKHR>>(formats) };

        let mut format_supported = false;
        for i in 0..formats.len() {
            if (formats[i].format == sys::VK_FORMAT_B8G8R8A8_SRGB) && 
                (formats[i].colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format_supported = true;
            }
        }
        if !format_supported {
            println!("window does not support BGRA8UN");
            return None;
        }
        dprintln!("format = {}",sys::VK_FORMAT_B8G8R8A8_SRGB);

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
            imageExtent: sys::VkExtent2D { width: extent.x,height: extent.y, },
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
        match unsafe { sys::vkCreateSwapchainKHR(
            self.vk_device,
            &info,
            null_mut(),
            vk_swapchain.as_mut_ptr(),
        ) } {
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
        match unsafe { sys::vkGetSwapchainImagesKHR(self.vk_device,vk_swapchain,&mut count,null_mut()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain image count (error {})",code);
                unsafe { sys::vkDestroySwapchainKHR(self.vk_device,vk_swapchain,null_mut()) };
                return None;
            }
        }
        let mut vk_images = vec![MaybeUninit::<sys::VkImage>::uninit(); count as usize];
        match unsafe { sys::vkGetSwapchainImagesKHR(
            self.vk_device,
            vk_swapchain,
            &count as *const u32 as *mut u32,
            vk_images.as_mut_ptr() as *mut sys::VkImage,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain images (error {})",code);
                unsafe { sys::vkDestroySwapchainKHR(self.vk_device,vk_swapchain,null_mut()) };
                return None;
            },
        }
        let vk_images = unsafe { transmute::<_,Vec<sys::VkImage>>(vk_images) };

        // create image views for the swapchain images
        dprintln!("creating image views onto swap chain images...");
        let mut vk_image_views = Vec::<sys::VkImageView>::new();
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
            let mut vk_image_view = MaybeUninit::uninit();
            match unsafe { sys::vkCreateImageView(self.vk_device,&info,null_mut(),vk_image_view.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create image view (error {})",code);
                    unsafe { sys::vkDestroySwapchainKHR(self.vk_device,vk_swapchain,null_mut()) };
                    return None;
                }
            }
            vk_image_views.push(unsafe { vk_image_view.assume_init() });
        }

        // create framebuffers for the image views
        dprintln!("creating frame buffers for the image views...");
        let mut vk_framebuffers = Vec::<sys::VkFramebuffer>::new();
        for vk_image_view in &vk_image_views {
            let info = sys::VkFramebufferCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                renderPass: vk_render_pass,
                attachmentCount: 1,
                pAttachments: vk_image_view,
                width: extent.x,
                height: extent.y,
                layers: 1,
            };
            let mut vk_framebuffer = MaybeUninit::uninit();
            match unsafe { sys::vkCreateFramebuffer(self.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create framebuffer (error {})",code);
                    unsafe { sys::vkDestroySwapchainKHR(self.vk_device,vk_swapchain,null_mut()) };
                    return None;
                }
            }
            vk_framebuffers.push(unsafe { vk_framebuffer.assume_init() });
        }

        dprintln!("success.");

        Some((vk_swapchain,vk_image_views,vk_framebuffers))
    }
}
    
impl<'system> System {
    
    // create basic window, decorations are handled in the public create_frame and create_popup
    fn create_window(&'system self,r: Rect<i32>,_absolute: bool) -> Option<Window> {

        // create window
        let xcb_window = unsafe { sys::xcb_generate_id(self.xcb_connection) };
        let values = [
            sys::XCB_EVENT_MASK_EXPOSURE
            | sys::XCB_EVENT_MASK_KEY_PRESS
            | sys::XCB_EVENT_MASK_KEY_RELEASE
            | sys::XCB_EVENT_MASK_BUTTON_PRESS
            | sys::XCB_EVENT_MASK_BUTTON_RELEASE
            | sys::XCB_EVENT_MASK_POINTER_MOTION
            | sys::XCB_EVENT_MASK_STRUCTURE_NOTIFY,
            unsafe { *self.xcb_screen }.default_colormap,
        ];
        unsafe {
            sys::xcb_create_window(
                self.xcb_connection,
                (*self.xcb_screen).root_depth as u8,
                xcb_window as u32,
                //if let Some(id) = parent { id as u32 } else { system.rootwindow as u32 },
                (*self.xcb_screen).root as u32,
                r.o.x as i16,
                r.o.y as i16,
                r.s.x as u16,
                r.s.y as u16,
                0,
                sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                (*self.xcb_screen).root_visual as u32,
                sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
                &values as *const u32 as *const std::os::raw::c_void
            );
            sys::xcb_map_window(self.xcb_connection,xcb_window as u32);
            sys::xcb_flush(self.xcb_connection);
        }

#[cfg(gpu="vulkan")]
        let (vk_surface,vk_render_pass,vk_swapchain,vk_image_views,vk_framebuffers) = {

            // create surface for this window
            let info = sys::VkXcbSurfaceCreateInfoKHR {
                sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: null_mut(),
                flags: 0,
                connection: self.xcb_connection,
                window: xcb_window,
            };
            let mut vk_surface = MaybeUninit::uninit();
            match unsafe { sys::vkCreateXcbSurfaceKHR(self.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("Unable to create Vulkan XCB surface (error {})",code);
                    return None;
                },
            }
            let vk_surface = unsafe { vk_surface.assume_init() };
        
            // create render pass

            // attachments: image descriptions and load/store indications of each attachment
            // subpasses: subpass? descriptions
            // dependencies: how subpasses depend on each othe

            // window needs one, and probably also off-screen rendering contexts

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
                    println!("unable to create render pass (error {})",code);
                    unsafe { sys::vkDestroySurfaceKHR(self.vk_instance,vk_surface,null_mut()) };
                    return None;
                }
            }
            let vk_render_pass = unsafe { vk_render_pass.assume_init() };
    
            // create swapchain resources
            if let Some((vk_swapchain,vk_image_views,vk_framebuffers)) = self.create_swapchain_resources(vk_surface,vk_render_pass,r) {
                (vk_surface,vk_render_pass,vk_swapchain,vk_image_views,vk_framebuffers)
            }
            else {
                unsafe {
                    sys::vkDestroyRenderPass(self.vk_device,vk_render_pass,null_mut());
                    sys::vkDestroySurfaceKHR(self.vk_instance,vk_surface,null_mut());
                }
                return None;
            }
        };

        Some(Window {
            system: &self,
            xcb_window,
#[cfg(gpu="vulkan")]
            vk_surface,
#[cfg(gpu="vulkan")]
            vk_render_pass,
#[cfg(gpu="vulkan")]
            vk_swapchain,
#[cfg(gpu="vulkan")]
            vk_image_views,
#[cfg(gpu="vulkan")]
            vk_framebuffers,            
        })
    }
    
    /// Create application frame window.
    pub fn create_frame_window(&'system self,r: Rect<i32>,title: &str) -> Option<Window> {
        let window = self.create_window(r,false)?;
        let protocol_set = [self.wm_delete_window];
        let protocol_set_void = protocol_set.as_ptr() as *const std::os::raw::c_void;
        unsafe { sys::xcb_change_property(
            self.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.wm_protocols,
            sys::XCB_ATOM_ATOM,
            32,
            1,
            protocol_set_void
        ) };
        unsafe { sys::xcb_change_property(
            self.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            sys::XCB_ATOM_WM_NAME,
            sys::XCB_ATOM_STRING,
            8,
            title.len() as u32,
            title.as_bytes().as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { sys::xcb_flush(self.xcb_connection) };
        Some(window)
    }
    
    /// Create standalone popup window.
    pub fn create_popup_window(&'system self,r: Rect<i32>) -> Option<Window> {
        let window = self.create_window(r,true)?;
        let net_state = [self.wm_net_state_above];
        unsafe { sys::xcb_change_property(
            self.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.wm_net_state,
            sys::XCB_ATOM_ATOM,
            32,
            1,
            net_state.as_ptr() as *const std::os::raw::c_void
        ) };
        let hints = [2u32,0,0,0,0];
        unsafe { sys::xcb_change_property(
            self.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.wm_motif_hints,
            sys::XCB_ATOM_ATOM,
            32,
            5,
            hints.as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { sys::xcb_flush(self.xcb_connection) };
        Some(window)
    }
}

impl<'system> Window<'system> {

#[cfg(gpu="vulkan")]
    fn destroy_swapchain_resources(&self) {
        unsafe {
            for vk_framebuffer in &self.vk_framebuffers {
                sys::vkDestroyFramebuffer(self.system.vk_device,*vk_framebuffer,null_mut());
            }
            for vk_image_view in &self.vk_image_views {
                sys::vkDestroyImageView(self.system.vk_device,*vk_image_view,null_mut());
            }
            sys::vkDestroySwapchainKHR(self.system.vk_device,self.vk_swapchain,null_mut());
        }
    }

    /// Get WindowID for this window.
    pub fn id(&self) -> WindowId {
        self.xcb_window as WindowId
    }

    pub fn update_swapchain_resources(&mut self,r: Rect<i32>) {
#[cfg(gpu="vulkan")]
        self.destroy_swapchain_resources();
        if let Some((vk_swapchain,vk_image_views,vk_framebuffers)) = self.system.create_swapchain_resources(self.vk_surface,self.vk_render_pass,r) {
            self.vk_swapchain = vk_swapchain;
            self.vk_image_views = vk_image_views;
            self.vk_framebuffers = vk_framebuffers;
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

impl<'system> Drop for Window<'system> {
    fn drop(&mut self) {
#[cfg(gpu="vulkan")]
        {
            self.destroy_swapchain_resources();
            unsafe {
                sys::vkDestroySurfaceKHR(self.system.vk_instance,self.vk_surface,null_mut());
                sys::vkDestroyRenderPass(self.system.vk_device,self.vk_render_pass,null_mut());
            }
        }
        unsafe {
            sys::xcb_unmap_window(self.system.xcb_connection,self.xcb_window as u32);
            sys::xcb_destroy_window(self.system.xcb_connection,self.xcb_window as u32);
        }
    }
}
