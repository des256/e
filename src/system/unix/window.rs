use {
    crate::*,
};

#[cfg(feature="gpu_vulkan")]
use {
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    }
};

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

pub struct Window<'system,'gpu,'screen> {
    pub screen: &'screen Screen<'system,'gpu>,
    pub r: Rect<i32>,
#[doc(hidden)]
    pub(crate) xcb_window: sys::xcb_window_t,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_surface: sys::VkSurfaceKHR,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_extent: sys::VkExtent2D,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_swapchain: sys::VkSwapchainKHR,
}

impl<'system,'gpu,'screen> Screen<'system,'gpu> {

    fn create_window(&self,r: Rect<i32>,_absolute: bool) -> Option<Window> {

        // create window
        let xcb_window = unsafe { sys::xcb_generate_id(self.gpu.system.xcb_connection) };
        let values = [sys::XCB_EVENT_MASK_EXPOSURE
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
                self.gpu.system.xcb_connection,
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
            sys::xcb_map_window(self.gpu.system.xcb_connection,xcb_window as u32);
            sys::xcb_flush(self.gpu.system.xcb_connection);
        }

#[cfg(feature="gpu_vulkan")]
        let (vk_surface,vk_extent,vk_swapchain) = {

            // create surface
            let info = sys::VkXcbSurfaceCreateInfoKHR {
                sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: null_mut(),
                flags: 0,
                connection: self.gpu.system.xcb_connection as *mut sys::xcb_connection_t,
                window: xcb_window,
            };
            let mut vk_surface = MaybeUninit::uninit();
            match unsafe { sys::vkCreateXcbSurfaceKHR(self.gpu.system.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("Unable to create Vulkan XCB surface (error {})",code);
                    unsafe {
                        sys::xcb_unmap_window(self.gpu.system.xcb_connection,xcb_window as u32);
                        sys::xcb_destroy_window(self.gpu.system.xcb_connection,xcb_window as u32);    
                    }
                    return None;
                },
            }
            let vk_surface = unsafe { vk_surface.assume_init() };

            // create swapchain for this window
            let mut capabilities = MaybeUninit::uninit();
            unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.gpu.vk_physical_device,vk_surface,capabilities.as_mut_ptr()) };
            let capabilities = unsafe { capabilities.assume_init() };
            let vk_extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
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
                vk_extent
            };
            println!("extent = {},{}",vk_extent.width,vk_extent.height);
    
            let mut image_count = capabilities.minImageCount + 1;
            if (capabilities.maxImageCount != 0) && (image_count > capabilities.maxImageCount) {
                image_count = capabilities.maxImageCount;
            }
            println!("image_count = {}",image_count);
    
            let mut count = 0u32;
            unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(self.gpu.vk_physical_device,vk_surface,&mut count,null_mut()) };
            if count == 0 {
                println!("No formats supported.");
                unsafe {
                    sys::vkDestroySurfaceKHR(self.gpu.system.vk_instance,vk_surface,null_mut());
                    sys::xcb_unmap_window(self.gpu.system.xcb_connection,xcb_window as u32);
                    sys::xcb_destroy_window(self.gpu.system.xcb_connection,xcb_window as u32);
                }
                return None;
            }
            let mut formats = vec![sys::VkSurfaceFormatKHR {
                format: 0,
                colorSpace: 0,
            }; count as usize];
            unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(self.gpu.vk_physical_device,vk_surface,&mut count,formats.as_mut_ptr()) };
            let mut format_supported = false;
            for i in 0..formats.len() {
                if (formats[i].format == sys::VK_FORMAT_B8G8R8A8_SRGB) && 
                   (formats[i].colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    format_supported = true;
                }
            }
            if !format_supported {
                println!("Format ARGB8UN not supported.");
                unsafe {
                    sys::vkDestroySurfaceKHR(self.gpu.system.vk_instance,vk_surface,null_mut());
                    sys::xcb_unmap_window(self.gpu.system.xcb_connection,xcb_window as u32);
                    sys::xcb_destroy_window(self.gpu.system.xcb_connection,xcb_window as u32);
                }
                return None;
            }
            println!("format = {}",sys::VK_FORMAT_B8G8R8A8_SRGB);
    
            let mut count = 0u32;
            unsafe { sys::vkGetPhysicalDeviceSurfacePresentModesKHR(self.gpu.vk_physical_device,vk_surface,&mut count,null_mut()) };
            if count == 0 {
                println!("No present modes supported.");
                unsafe { 
                    sys::vkDestroySurfaceKHR(self.gpu.system.vk_instance,vk_surface,null_mut());
                    sys::xcb_unmap_window(self.gpu.system.xcb_connection,xcb_window as u32);
                    sys::xcb_destroy_window(self.gpu.system.xcb_connection,xcb_window as u32);
                }
                return None;
            }
            let mut modes = vec![0 as sys::VkPresentModeKHR; count as usize];
            unsafe { sys::vkGetPhysicalDeviceSurfacePresentModesKHR(self.gpu.vk_physical_device,vk_surface,&mut count,modes.as_mut_ptr()) };
            let mut present_mode = sys::VK_PRESENT_MODE_FIFO_KHR;
            for mode in &modes {
                if *mode == sys::VK_PRESENT_MODE_MAILBOX_KHR {
                    present_mode = *mode;
                }
            }
            println!("present_mode = {}",present_mode);
    
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
                presentMode: present_mode,
                clipped: sys::VK_TRUE,
                oldSwapchain: null_mut(),
            };
            let mut vk_swapchain = MaybeUninit::uninit();
            match unsafe { sys::vkCreateSwapchainKHR(self.vk_device,&info,null_mut(),vk_swapchain.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("Unable to create swap chain (error {})",code);
                    unsafe {
                        sys::vkDestroySurfaceKHR(self.gpu.system.vk_instance,vk_surface,null_mut());
                        sys::xcb_unmap_window(self.gpu.system.xcb_connection,xcb_window as u32);
                        sys::xcb_destroy_window(self.gpu.system.xcb_connection,xcb_window as u32);
                    }
                    return None;
                },
            }
            let vk_swapchain = unsafe { vk_swapchain.assume_init() };
    
            (vk_surface,vk_extent,vk_swapchain)
        };

#[cfg(feature="gpu_opengl")]
        unsafe { sys::glXMakeCurrent(self.gpu.system.xdisplay,xcb_window as sys::GLXDrawable,self.gpu.system.glx_context); }

        Some(Window {
            screen: &self,
            r: r,
            xcb_window: xcb_window,
#[cfg(feature="gpu_vulkan")]
            vk_surface: vk_surface,
#[cfg(feature="gpu_vulkan")]
            vk_extent: vk_extent,
#[cfg(feature="gpu_vulkan")]
            vk_swapchain: vk_swapchain,
        })
    }

    pub fn create_frame(&self,r: Rect<i32>,title: &str) -> Option<Window> {
        let window = self.create_window(r,false)?;
        let protocol_set = [self.gpu.system.wm_delete_window];
        let protocol_set_void = protocol_set.as_ptr() as *const std::os::raw::c_void;
        unsafe { sys::xcb_change_property(
            self.gpu.system.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.gpu.system.wm_protocols,
            sys::XCB_ATOM_ATOM,
            32,
            1,
            protocol_set_void
        ) };
        unsafe { sys::xcb_change_property(
            self.gpu.system.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            sys::XCB_ATOM_WM_NAME,
            sys::XCB_ATOM_STRING,
            8,
            title.len() as u32,
            title.as_bytes().as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { sys::xcb_flush(self.gpu.system.xcb_connection) };
        Some(window)
    }

    pub fn create_popup(&self,r: Rect<i32>) -> Option<Window> {
        let window = self.create_window(r,true)?;
        let net_state = [self.gpu.system.wm_net_state_above];
        unsafe { sys::xcb_change_property(
            self.gpu.system.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.gpu.system.wm_net_state,
            sys::XCB_ATOM_ATOM,
            32,
            1,
            net_state.as_ptr() as *const std::os::raw::c_void
        ) };
        let hints = [2u32,0,0,0,0];
        unsafe { sys::xcb_change_property(
            self.gpu.system.xcb_connection,
            sys::XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.gpu.system.wm_motif_hints,
            sys::XCB_ATOM_ATOM,
            32,
            5,
            hints.as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { sys::xcb_flush(self.gpu.system.xcb_connection) };
        Some(window)
    }
}

impl<'system,'gpu,'screen> Window<'system,'gpu,'screen> {

    /*
    pub fn present(&self,index: usize,semaphore: &Semaphore) {
        let image_index = index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.screen.vk_present_queue,&info) };
    }
    */

    /*
    pub fn swap(&self,semaphore: &Semaphore) -> usize {
        let mut image_index = 0u32;
        unsafe { sys::vkAcquireNextImageKHR(self.screen.vk_device,self.vk_swapchain,0xFFFFFFFFFFFFFFFF,semaphore.vk_semaphore,null_mut(),&mut image_index) };
        image_index as usize
    }
    */
}
/*
impl Window {
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

impl<'system,'gpu,'screen> Drop for Window<'system,'gpu,'screen> {
    fn drop(&mut self) {
        unsafe {
#[cfg(feature="gpu_vulkan")]
            sys::vkDestroySwapchainKHR(self.screen.vk_device,self.vk_swapchain,null_mut());
#[cfg(feature="gpu_vulkan")]
            sys::vkDestroySurfaceKHR(self.screen.gpu.system.vk_instance,self.vk_surface,null_mut());

            sys::xcb_unmap_window(self.screen.gpu.system.xcb_connection,self.xcb_window as u32);
            sys::xcb_destroy_window(self.screen.gpu.system.xcb_connection,self.xcb_window as u32);
        }
    }
}
