use {
    crate::*,
    std::{
        rc::Rc,
        os::raw::c_int,
        ptr::null_mut,
    },
};

/// The system structure (linux).
pub struct System {
    pub(crate) gpu: SystemGpu,
    pub(crate) xdisplay: *mut sys::Display,
    pub(crate) xcb_connection: *mut sys::xcb_connection_t,
    pub(crate) xcb_root_window: sys::xcb_window_t,
    pub(crate) xcb_colormap: sys::xcb_colormap_t,
    pub(crate) epfd: c_int,
    pub(crate) wm_protocols: u32,
    pub(crate) wm_delete_window: u32,
    pub(crate) wm_motif_hints: u32,
#[allow(dead_code)]
    pub(crate) wm_transient_for: u32,
#[allow(dead_code)]
    pub(crate) wm_net_type: u32,
#[allow(dead_code)]
    pub(crate) wm_net_type_utility: u32,
#[allow(dead_code)]
    pub(crate) wm_net_type_dropdown_menu: u32,
    pub(crate) wm_net_state: u32,
    pub(crate) wm_net_state_above: u32,
}

fn intern_atom_cookie(xcb_connection: *mut sys::xcb_connection_t,name: &str) -> sys::xcb_intern_atom_cookie_t {
    let i8_name = unsafe { std::mem::transmute::<_,&[i8]>(name.as_bytes()) };
    unsafe { sys::xcb_intern_atom(xcb_connection,0,name.len() as u16,i8_name.as_ptr()) }
}

fn resolve_atom_cookie(xcb_connection: *mut sys::xcb_connection_t,cookie: sys::xcb_intern_atom_cookie_t) -> u32 {
    unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,cookie,null_mut())).atom }
}

/// Open the system interface.
pub fn open_system() -> Option<Rc<System>> {

    // open X connection and get first screen
    let xdisplay = unsafe { sys::XOpenDisplay(null_mut()) };
    if xdisplay == null_mut() {
        println!("unable to connect to X server");
        return None;
    }
    let xcb_connection = unsafe { sys::XGetXCBConnection(xdisplay) };
    if xcb_connection == null_mut() {
        println!("unable to connect to X server");
        unsafe { sys::XCloseDisplay(xdisplay) };
        return None;
    }
    unsafe { sys::XSetEventQueueOwner(xdisplay,sys::XCBOwnsEventQueue) };
    let xcb_setup = unsafe { sys::xcb_get_setup(xcb_connection) };
    if xcb_setup == null_mut() {
        println!("unable to obtain X server setup");
        return None;
    }

    // start by assuming the root depth and visual
    let xcb_screen = unsafe { sys::xcb_setup_roots_iterator(xcb_setup) }.data;
    let xcb_root_window = unsafe { *xcb_screen }.root;
    let xcb_colormap = unsafe { *xcb_screen }.default_colormap;

    // create epoll descriptor to be able to wait for UI events on a system level
    let fd = unsafe { sys::xcb_get_file_descriptor(xcb_connection) };
    let epfd = unsafe { sys::epoll_create1(0) };
    let mut epe = [sys::epoll_event { events: sys::EPOLLIN as u32,data: sys::epoll_data_t { u64_: 0, }, }];
    unsafe { sys::epoll_ctl(epfd,sys::EPOLL_CTL_ADD as c_int,fd,epe.as_mut_ptr()) };

    // get the atoms
    let protocols_cookie = intern_atom_cookie(xcb_connection,"WM_PROTOCOLS");
    let delete_window_cookie = intern_atom_cookie(xcb_connection,"WM_DELETE_WINDOW");
    let motif_hints_cookie = intern_atom_cookie(xcb_connection,"_MOTIF_WM_HINTS");
    let transient_for_cookie = intern_atom_cookie(xcb_connection,"WM_TRANSIENT_FOR");
    let net_type_cookie = intern_atom_cookie(xcb_connection,"_NET_WM_TYPE");
    let net_type_utility_cookie = intern_atom_cookie(xcb_connection,"_NET_WM_TYPE_UTILITY");
    let net_type_dropdown_menu_cookie = intern_atom_cookie(xcb_connection,"_NET_WM_TYPE_DROPDOWN_MENU");
    let net_state_cookie = intern_atom_cookie(xcb_connection,"_NET_WM_STATE");
    let net_state_above_cookie = intern_atom_cookie(xcb_connection,"_NET_WM_STATE_ABOVE");

    let wm_protocols = resolve_atom_cookie(xcb_connection,protocols_cookie);
    let wm_delete_window = resolve_atom_cookie(xcb_connection,delete_window_cookie);
    let wm_motif_hints = resolve_atom_cookie(xcb_connection,motif_hints_cookie);
    let wm_transient_for = resolve_atom_cookie(xcb_connection,transient_for_cookie);
    let wm_net_type = resolve_atom_cookie(xcb_connection,net_type_cookie);
    let wm_net_type_utility = resolve_atom_cookie(xcb_connection,net_type_utility_cookie);
    let wm_net_type_dropdown_menu = resolve_atom_cookie(xcb_connection,net_type_dropdown_menu_cookie);
    let wm_net_state = resolve_atom_cookie(xcb_connection,net_state_cookie);
    let wm_net_state_above = resolve_atom_cookie(xcb_connection,net_state_above_cookie);

#[cfg(gpu="vulkan")]
    if let Some(gpu) = open_system_gpu(xcb_screen) {
        Some(Rc::new(System {
            gpu,
            xdisplay,
            xcb_connection,
            xcb_root_window,
            xcb_colormap,
            epfd,
            wm_protocols,
            wm_delete_window,
            wm_motif_hints,
            wm_transient_for,
            wm_net_type,
            wm_net_type_utility,
            wm_net_type_dropdown_menu,
            wm_net_state,
            wm_net_state_above,
        }))
    }
    else {
        None
    }

#[cfg(gpu="opengl")]
    if let Some(gpu) = open_system_gpu(xdisplay,xcb_connection,xcb_colormap,xcb_root_window) {
        Some(Rc::new(System {
            gpu,
            xdisplay,
            xcb_connection,
            xcb_root_window,
            xcb_colormap,
            epfd,
            wm_protocols,
            wm_delete_window,
            wm_motif_hints,
            wm_transient_for,
            wm_net_type,
            wm_net_type_utility,
            wm_net_type_dropdown_menu,
            wm_net_state,
            wm_net_state_above,
        }))
    }
    else {
        None
    }
}

impl System {

#[doc(hidden)]
    fn translate_event(&self,xcb_event: *mut sys::xcb_generic_event_t) -> Option<(WindowId,Event)> {
        match (unsafe { *xcb_event }.response_type & 0x7F) as u32 {
            sys::XCB_EXPOSE => {
                let expose = xcb_event as *const sys::xcb_expose_event_t;
                //let expose = unsafe { std::mem::transmute::<_,xcb_expose_event_t>(xcb_event) };
                //let r = rect!(expose.x as isize,expose.y as isize,expose.width() as isize,expose.height() as isize);
                let xcb_window = unsafe { *expose }.window;
                return Some((xcb_window as WindowId,Event::Expose));
            },
            sys::XCB_KEY_PRESS => {
                let key_press = xcb_event as *const sys::xcb_key_press_event_t;
                let xcb_window = unsafe { *key_press }.event;
                return Some((xcb_window as WindowId,Event::KeyPress(unsafe { *key_press }.detail as u8)));
            },
            sys::XCB_KEY_RELEASE => {
                let key_release = xcb_event as *const sys::xcb_key_release_event_t;
                let xcb_window = unsafe { *key_release }.event;
                return Some((xcb_window as WindowId,Event::KeyRelease(unsafe { *key_release }.detail as u8)));
            },
            sys::XCB_BUTTON_PRESS => {
                let button_press = xcb_event as *const sys::xcb_button_press_event_t;
                let p = unsafe { Vec2 {
                    x: (*button_press).event_x as isize,
                    y: (*button_press).event_y as isize,
                } };
                let xcb_window = unsafe { *button_press }.event;
                match unsafe { *button_press }.detail {
                    1 => { return Some((xcb_window as WindowId,Event::MousePress(p,MouseButton::Left))); },
                    2 => { return Some((xcb_window as WindowId,Event::MousePress(p,MouseButton::Middle))); },
                    3 => { return Some((xcb_window as WindowId,Event::MousePress(p,MouseButton::Right))); },
                    4 => { return Some((xcb_window as WindowId,Event::MouseWheel(MouseWheel::Up))); },
                    5 => { return Some((xcb_window as WindowId,Event::MouseWheel(MouseWheel::Down))); },
                    6 => { return Some((xcb_window as WindowId,Event::MouseWheel(MouseWheel::Left))); },
                    7 => { return Some((xcb_window as WindowId,Event::MouseWheel(MouseWheel::Right))); },
                    _ => { },
                }        
            },
            sys::XCB_BUTTON_RELEASE => {
                let button_release = xcb_event as *const sys::xcb_button_release_event_t;
                let p = unsafe { Vec2 {
                    x: (*button_release).event_x as isize,
                    y: (*button_release).event_y as isize,
                } };
                let xcb_window = unsafe { *button_release }.event;
                match unsafe { *button_release }.detail {
                    1 => { return Some((xcb_window as WindowId,Event::MouseRelease(p,MouseButton::Left))); },
                    2 => { return Some((xcb_window as WindowId,Event::MouseRelease(p,MouseButton::Middle))); },
                    3 => { return Some((xcb_window as WindowId,Event::MouseRelease(p,MouseButton::Right))); },
                    _ => { },
                }        
            },
            sys::XCB_MOTION_NOTIFY => {
                let motion_notify = xcb_event as *const sys::xcb_motion_notify_event_t;
                let p = Vec2 {
                    x: unsafe { *motion_notify }.event_x as isize,
                    y: unsafe { *motion_notify }.event_y as isize,
                };
                let xcb_window = unsafe { *motion_notify }.event;
                return Some((xcb_window as WindowId,Event::MouseMove(p)));
            },
            sys::XCB_CONFIGURE_NOTIFY => {
                let configure_notify = xcb_event as *const sys::xcb_configure_notify_event_t;
                let r = Rect {
                    o: Vec2 {
                        x: unsafe { *configure_notify }.x as isize,
                        y: unsafe { *configure_notify }.y as isize,
                    },
                    s: Vec2 {
                        x: unsafe { *configure_notify }.width as usize,
                        y: unsafe { *configure_notify }.height as usize,
                    },
                };
                let xcb_window = unsafe { *configure_notify }.window;
                return Some((xcb_window as WindowId,Event::Configure(r)));
            },
            sys::XCB_CLIENT_MESSAGE => {
                let client_message = xcb_event as *const sys::xcb_client_message_event_t;
                let atom = unsafe { (*client_message).data.data32[0] };
                if atom == self.wm_delete_window {
                    let xcb_window = unsafe { *client_message }.window;
                    return Some((xcb_window as WindowId,Event::Close));
                }
            },
            _ => {
            },
        }
        None
    }

    /// Get all OS window events that have gathered.
    pub fn flush(&self) -> Vec<(WindowId,Event)> {
        let mut events = Vec::<(WindowId,Event)>::new();
        loop {
            let event = unsafe { sys::xcb_poll_for_event(self.xcb_connection) };
            if event != null_mut() {
                if let Some((window_id,event)) = self.translate_event(event) {
                    events.push((window_id,event));
                }
            }
            else {
                break;
            }
        }
        events
    }

    /// Sleep until new OS window events appear.
    pub fn wait(&self) {
        let mut epe = [ sys::epoll_event { events: sys::EPOLLIN as u32,data: sys::epoll_data_t { u64_: 0, } } ];
        unsafe { sys::epoll_wait(self.epfd,epe.as_mut_ptr(),1,-1) };
    }

    /*// take ownership of the mouse
    pub fn capture_mouse(&self,_id: u64) {
        /*println!("XGrabPointer");
        grab_pointer(
            &self.connection,
            false,
            id as u32,
            (EVENT_MASK_BUTTON_PRESS | EVENT_MASK_BUTTON_RELEASE| EVENT_MASK_POINTER_MOTION) as u16,
            GRAB_MODE_ASYNC as u8,
            GRAB_MODE_ASYNC as u8,
            WINDOW_NONE,
            CURSOR_NONE,
            TIME_CURRENT_TIME
        );*/
    }
    
    // release ownership of the mouse
    pub fn release_mouse(&self) {
        //println!("XUngrabPointer");
        //ungrab_pointer(&self.connection,TIME_CURRENT_TIME);
    }

    // switch mouse cursor
    pub fn set_mousecursor(&self,_id: u64,_n: usize) {
        //let values = [(CW_CURSOR,self.cursors[n])];
        //change_window_attributes(&self.connection,id as u32,&values);
    }*/

    // create basic window, decorations are handled in the public create_frame and create_popup
    fn create_window(self: &Rc<System>,r: Rect<isize,usize>,_absolute: bool) -> Option<Window> {

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
            self.xcb_colormap,
        ];
        unsafe {
            sys::xcb_create_window(
                self.xcb_connection,
                self.gpu.xcb_depth,
                xcb_window as u32,
                //if let Some(id) = parent { id as u32 } else { system.rootwindow as u32 },
                self.xcb_root_window,
                r.o.x as i16,
                r.o.y as i16,
                r.s.x as u16,
                r.s.y as u16,
                0,
                sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                self.gpu.xcb_visual_id,
                sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
                &values as *const u32 as *const std::os::raw::c_void
            );
            sys::xcb_map_window(self.xcb_connection,xcb_window as u32);
            sys::xcb_flush(self.xcb_connection);
        }

#[cfg(gpu="vulkan")]
        {
            // create surface for this window
            let info = sys::VkXcbSurfaceCreateInfoKHR {
                sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: null_mut(),
                flags: 0,
                connection: self.xcb_connection,
                window: xcb_window,
            };
            let mut vk_surface = std::mem::MaybeUninit::uninit();
            match unsafe { sys::vkCreateXcbSurfaceKHR(self.gpu.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("Unable to create Vulkan XCB surface (error {})",code);
                    return None;
                },
            }
            let vk_surface = unsafe { vk_surface.assume_init() };

            if let Some(gpu) = self.create_window_gpu(vk_surface,r) {
                Some(Window {
                    system: Rc::clone(self),
                    xcb_window,
                    gpu,
                })                
            }
            else {
                None
            }
        }
        
#[cfg(gpu="opengl")]
        if let Some(gpu) = self.create_window_gpu() {
            Some(Window {
                system: Rc::clone(self),
                gpu,
                xcb_window,
            })
        }
        else {
            None
        }
    }
    
    /// Create application frame window.
    pub fn create_frame_window(self: &Rc<System>,r: Rect<isize,usize>,title: &str) -> Option<Window> {
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
    pub fn create_popup_window(self: &Rc<System>,r: Rect<isize,usize>) -> Option<Window> {
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

impl Drop for System {

    /// Drop the system interface.
    fn drop(&mut self) {
        self.drop_gpu();
        unsafe { sys::XCloseDisplay(self.xdisplay) };
    }
}
