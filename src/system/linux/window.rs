use {
    crate::*,
    std::{
        rc::Rc,
        cell::Cell,
        cell::RefCell,
        ptr::null_mut,
        mem::MaybeUninit,
    },
    sys_sys::*,
};

pub const KEY_UP: u8 = 111;
pub const KEY_DOWN: u8 = 116;
pub const KEY_LEFT: u8 = 113;
pub const KEY_RIGHT: u8 = 114;

pub struct Window {
    pub screen: Rc<Screen>,
    pub r: Cell<Rect<i32>>,
    pub handler: RefCell<Option<Box<dyn Fn(Event)>>>,
#[doc(hidden)]
    pub(crate) xcb_window: xcb_window_t,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_surface: VkSurfaceKHR,
}

impl Screen {

    fn create_window(self: &Rc<Self>,r: Rect<i32>,_absolute: bool) -> Option<Rc<Window>> {

        let xcb_window = unsafe { xcb_generate_id(self.system.xcb_connection) };
        let values = [XCB_EVENT_MASK_EXPOSURE
            | XCB_EVENT_MASK_KEY_PRESS
            | XCB_EVENT_MASK_KEY_RELEASE
            | XCB_EVENT_MASK_BUTTON_PRESS
            | XCB_EVENT_MASK_BUTTON_RELEASE
            | XCB_EVENT_MASK_POINTER_MOTION
            | XCB_EVENT_MASK_STRUCTURE_NOTIFY,
            unsafe { *self.xcb_screen }.default_colormap,
        ];
        unsafe {
            xcb_create_window(
                self.system.xcb_connection,
                (*self.xcb_screen).root_depth as u8,
                xcb_window as u32,
                //if let Some(id) = parent { id as u32 } else { system.rootwindow as u32 },
                (*self.xcb_screen).root as u32,
                r.o.x as i16,
                r.o.y as i16,
                r.s.x as u16,
                r.s.y as u16,
                0,
                XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                (*self.xcb_screen).root_visual as u32,
                XCB_CW_EVENT_MASK | XCB_CW_COLORMAP,
                &values as *const u32 as *const std::os::raw::c_void
            );
            xcb_map_window(self.system.xcb_connection,xcb_window as u32);
            xcb_flush(self.system.xcb_connection);
        }

#[cfg(feature="gpu_vulkan")]
        let vk_surface = {
            let info = VkXcbSurfaceCreateInfoKHR {
                sType: VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: null_mut(),
                flags: 0,
                connection: self.system.xcb_connection as *mut xcb_connection_t,
                window: xcb_window,
            };
            let mut vk_surface = MaybeUninit::uninit();
            match unsafe { vkCreateXcbSurfaceKHR(self.system.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
                VK_SUCCESS => { },
                code => {
#[cfg(feature="debug_output")]
                    println!("Unable to create Vulkan XCB surface (error {})",code);
                    unsafe {
                        xcb_unmap_window(self.system.xcb_connection,xcb_window as u32);
                        xcb_destroy_window(self.system.xcb_connection,xcb_window as u32);    
                    }
                    return None;
                },
            }
            let vk_surface = unsafe { vk_surface.assume_init() };

            let mut support: VkBool32 = 0;
            unsafe { vkGetPhysicalDeviceSurfaceSupportKHR(self.gpu.vk_physical_device,self.present_queue_id,vk_surface,&mut support) };
            if support == 0 {
#[cfg(feature="debug_output")]
                println!("Window not compatible with GPU.");
                unsafe {
                    vkDestroySurfaceKHR(self.system.vk_instance,vk_surface,null_mut());
                    xcb_unmap_window(self.system.xcb_connection,xcb_window as u32);
                    xcb_destroy_window(self.system.xcb_connection,xcb_window as u32);    
                }
                return None;
            }

            vk_surface
        };

        let window = Rc::new(Window {
            screen: Rc::clone(self),
            r: Cell::new(r),
            handler: RefCell::new(None),
            xcb_window: xcb_window,
#[cfg(feature="gpu_vulkan")]
            vk_surface: vk_surface,
        });
        self.system.xcb_window_pointers.borrow_mut().insert(xcb_window,Rc::as_ptr(&window));

        Some(window)
    }

    pub fn create_frame(self: &Rc<Self>,r: Rect<i32>,title: &str) -> Option<Rc<Window>> {
        let window = self.create_window(r,false)?;
        let protocol_set = [self.system.xcb_atoms.wm_delete_window];
        let protocol_set_void = protocol_set.as_ptr() as *const std::os::raw::c_void;
        unsafe { xcb_change_property(
            self.system.xcb_connection,
            XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.system.xcb_atoms.wm_protocols,
            XCB_ATOM_ATOM,
            32,
            1,
            protocol_set_void
        ) };
        unsafe { xcb_change_property(
            self.system.xcb_connection,
            XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            XCB_ATOM_WM_NAME,
            XCB_ATOM_STRING,
            8,
            title.len() as u32,
            title.as_bytes().as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { xcb_flush(self.system.xcb_connection) };
        Some(window)
    }

    pub fn create_popup(self: &Rc<Self>,r: Rect<i32>) -> Option<Rc<Window>> {
        let window = self.create_window(r,true)?;
        let net_state = [self.system.xcb_atoms.wm_net_state_above];
        unsafe { xcb_change_property(
            self.system.xcb_connection,
            XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.system.xcb_atoms.wm_net_state,
            XCB_ATOM_ATOM,
            32,
            1,
            net_state.as_ptr() as *const std::os::raw::c_void
        ) };
        let hints = [2u32,0,0,0,0];
        unsafe { xcb_change_property(
            self.system.xcb_connection,
            XCB_PROP_MODE_REPLACE as u8,
            window.xcb_window as u32,
            self.system.xcb_atoms.wm_motif_hints,
            XCB_ATOM_ATOM,
            32,
            5,
            hints.as_ptr() as *const std::os::raw::c_void
        ) };
        unsafe { xcb_flush(self.system.xcb_connection) };
        Some(window)
    }
}

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

impl Drop for Window {
    fn drop(&mut self) {
        unsafe {
            self.screen.system.xcb_window_pointers.borrow_mut().remove(&self.xcb_window);
#[cfg(feature="gpu_vulkan")]
            vkDestroySurfaceKHR(self.screen.system.vk_instance,self.vk_surface,null_mut());
            xcb_unmap_window(self.screen.system.xcb_connection,self.xcb_window as u32);
            xcb_destroy_window(self.screen.system.xcb_connection,self.xcb_window as u32);
        }
    }
}
