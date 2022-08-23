use {
    crate::*,
    std::rc::Rc,
};

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

pub struct Window {
    pub system: Rc<System>,
    pub(crate) gpu: WindowGpu,
#[doc(hidden)]
    pub(crate) xcb_window: sys::xcb_window_t,
}

impl Window {

    /// Get WindowID for this window.
    pub fn id(&self) -> WindowId {
        self.xcb_window as WindowId
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
        self.drop_gpu();
        unsafe {
            sys::xcb_unmap_window(self.system.xcb_connection,self.xcb_window as u32);
            sys::xcb_destroy_window(self.system.xcb_connection,self.xcb_window as u32);
        }
    }

    /*
    Cannot call vkDestroyRenderPass on VkRenderPass 0x30000000003[] that is
    currently in use by a command buffer. The Vulkan spec states: All
    submitted commands that refer to renderPass must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroyRenderPass-renderPass-00873)
    */
}
