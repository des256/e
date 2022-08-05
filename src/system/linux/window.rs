use crate::*;

#[cfg(gpu="vulkan")]
use std::ptr::null_mut;

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

pub struct Window<'system> {
    pub system: &'system System,
    pub r: Rect<i32>,
#[doc(hidden)]
    pub(crate) xcb_window: sys::xcb_window_t,
#[cfg(gpu="vulkan")]
    pub(crate) vk_surface: sys::VkSurfaceKHR,
#[cfg(gpu="vulkan")]
    pub(crate) vk_extent: sys::VkExtent2D,
#[cfg(gpu="vulkan")]
    pub(crate) vk_swapchain: sys::VkSwapchainKHR,
#[cfg(gpu="vulkan")]
    pub(crate) vk_renderpass: sys::VkRenderPass,
#[cfg(gpu="vulkan")]
    pub(crate) vk_imageviews: Vec<sys::VkImageView>,
#[cfg(gpu="vulkan")]
    pub(crate) vk_framebuffers: Vec<sys::VkFramebuffer>,
}

impl<'system> Window<'system> {

    /// Get WindowID for this window.
    pub fn id(&self) -> WindowId {
        self.xcb_window as WindowId
    }

    pub fn get_framebuffers(&self) -> Vec<Framebuffer> {

#[cfg(gpu="vulkan")]
        {
            let mut framebuffers = Vec::<Framebuffer>::new();
            for vk_framebuffer in &self.vk_framebuffers {
                framebuffers.push(Framebuffer {
                    system: self.system,
                    owned: false,
                    vk_framebuffer: *vk_framebuffer,
                });
            }
            framebuffers
        }

#[cfg(not(gpu="vulkan"))]
        Vec::new()
    }

    pub fn acquire_next(&self,semaphore: &Semaphore) -> usize {
#[cfg(gpu="vulkan")]        
        {
            let mut image_index = 0u32;
            unsafe { sys::vkAcquireNextImageKHR(self.system.vk_device,self.vk_swapchain,0xFFFFFFFFFFFFFFFF,semaphore.vk_semaphore,null_mut(),&mut image_index) };
            image_index as usize
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
            pSwapchains: &self.vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        unsafe { sys::vkQueuePresentKHR(self.system.vk_queue,&info) };
    }

#[cfg(gpu="vulkan")]
    fn destroy_swapchain(&self) {
        for vk_framebuffer in &self.vk_framebuffers {
            unsafe { sys::vkDestroyFramebuffer(self.system.vk_device,*vk_framebuffer,null_mut()) };
        }
        for vk_imageview in &self.vk_imageviews {
            unsafe { sys::vkDestroyImageView(self.system.vk_device,*vk_imageview,null_mut()) };
        }
        unsafe { sys::vkDestroySwapchainKHR(self.system.vk_device,self.vk_swapchain,null_mut()) };
    }

    pub fn update_configure(&mut self,r: Rect<i32>) {
#[cfg(gpu="vulkan")]
        {
            if r.s != self.r.s {
                self.destroy_swapchain();
                if let Some((vk_extent,vk_swapchain,vk_imageviews,vk_framebuffers)) = self.system.create_swapchain(self.vk_surface,self.vk_renderpass,r) {
                    self.r = r;
                    self.vk_extent = vk_extent;
                    self.vk_swapchain = vk_swapchain;
                    self.vk_imageviews = vk_imageviews;
                    self.vk_framebuffers = vk_framebuffers;
                }
                else {
                    println!("unable to recreate swapchain after window resize");
                }
            }
        }
    }
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

impl<'system> Drop for Window<'system> {
    fn drop(&mut self) {
        unsafe {
#[cfg(gpu="vulkan")]
            {
                sys::vkDestroySwapchainKHR(self.system.vk_device,self.vk_swapchain,null_mut());
                sys::vkDestroySurfaceKHR(self.system.vk_instance,self.vk_surface,null_mut());
            }
            sys::xcb_unmap_window(self.system.xcb_connection,self.xcb_window as u32);
            sys::xcb_destroy_window(self.system.xcb_connection,self.xcb_window as u32);
        }
    }
}
