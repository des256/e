use {
    crate::*,
    std::{
        rc::Rc,
        os::raw::{
            c_int,
            c_void,
        },
        ptr::null_mut,
    },
};

/// The system structure (linux).
#[derive(Debug)]
pub struct System {
    pub(crate) gpu_system: Rc<GpuSystem>,
    pub(crate) xdisplay: *mut sys::Display,
    pub(crate) xcb_connection: *mut sys::xcb_connection_t,
    pub(crate) xcb_screen: *mut sys::xcb_screen_t,
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

impl System {

    /// Open the system interface.
    pub fn open() -> Result<Rc<System>,String> {

        // open X connection and get first screen
        let xdisplay = unsafe { sys::XOpenDisplay(null_mut()) };
        if xdisplay == null_mut() {
            return Err("unable to connect to X server".to_string());
        }
        let xcb_connection = unsafe { sys::XGetXCBConnection(xdisplay) };
        if xcb_connection == null_mut() {
            unsafe { sys::XCloseDisplay(xdisplay) };
            return Err("unable to connect to X server".to_string());
        }
        unsafe { sys::XSetEventQueueOwner(xdisplay,sys::XCBOwnsEventQueue) };
        let xcb_setup = unsafe { sys::xcb_get_setup(xcb_connection) };
        if xcb_setup == null_mut() {
            unsafe { sys::XCloseDisplay(xdisplay) };
            return Err("unable to obtain X server setup".to_string());
        }

        // start by assuming the root depth and visual
        let xcb_screen = unsafe { sys::xcb_setup_roots_iterator(xcb_setup) }.data;

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

        // initialize GPU
        let gpu_system = GpuSystem::open()?;

        Ok(Rc::new(System {
            gpu_system,
            xdisplay,
            xcb_connection,
            xcb_screen,
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

#[doc(hidden)]
    fn translate_xevent(&self,xcb_event: *mut sys::xcb_generic_event_t) -> Option<(u32,Event)> {
        match (unsafe { *xcb_event }.response_type & 0x7F) as u32 {
            sys::XCB_EXPOSE => {
                let expose = xcb_event as *const sys::xcb_expose_event_t;
                //let expose = unsafe { std::mem::transmute::<_,xcb_expose_event_t>(xcb_event) };
                let r = Rect {
                    o: Vec2 {
                        x: unsafe { *expose }.x as f32,
                        y: unsafe { *expose }.y as f32,
                    },
                    s: Vec2 {
                        x: unsafe { *expose }.width as f32,
                        y: unsafe { *expose }.height as f32,
                    },
                };
                let xcb_window = unsafe { *expose }.window;
                return Some((xcb_window,Event::Expose(r)));
            },
            sys::XCB_KEY_PRESS => {
                let key_press = xcb_event as *const sys::xcb_key_press_event_t;
                let xcb_window = unsafe { *key_press }.event;
                return Some((xcb_window,Event::Key(KeyEvent::Press { code: unsafe { *key_press }.detail as u32, })));
            },
            sys::XCB_KEY_RELEASE => {
                let key_release = xcb_event as *const sys::xcb_key_release_event_t;
                let xcb_window = unsafe { *key_release }.event;
                return Some((xcb_window,Event::Key(KeyEvent::Release { code: unsafe { *key_release }.detail as u32, })));
            },
            sys::XCB_BUTTON_PRESS => {
                let button_press = xcb_event as *const sys::xcb_button_press_event_t;
                let p = unsafe { Vec2 {
                    x: (*button_press).event_x as f32,
                    y: (*button_press).event_y as f32,
                } };
                let xcb_window = unsafe { *button_press }.event;
                match unsafe { *button_press }.detail {
                    1 => { return Some((xcb_window,Event::Pointer(PointerEvent::Down { position: p,button: Button::Left, }))); },
                    2 => { return Some((xcb_window,Event::Pointer(PointerEvent::Down { position: p,button: Button::Middle, }))); },
                    3 => { return Some((xcb_window,Event::Pointer(PointerEvent::Down { position: p,button: Button::Right, }))); },
                    4 => { return Some((xcb_window,Event::Pointer(PointerEvent::Scroll { position: p,buttons: Vec::new(), delta: Vec2 { x: 0.0,y: -1.0, }, }))); },
                    5 => { return Some((xcb_window,Event::Pointer(PointerEvent::Scroll { position: p,buttons: Vec::new(), delta: Vec2 { x: 0.0,y: 1.0, }, }))); },
                    6 => { return Some((xcb_window,Event::Pointer(PointerEvent::Scroll { position: p,buttons: Vec::new(), delta: Vec2 { x: -1.0,y: 0.0, }, }))); },
                    7 => { return Some((xcb_window,Event::Pointer(PointerEvent::Scroll { position: p,buttons: Vec::new(), delta: Vec2 { x: 1.0,y: 0.0, }, }))); },
                    _ => { },
                }        
            },
            sys::XCB_BUTTON_RELEASE => {
                let button_release = xcb_event as *const sys::xcb_button_release_event_t;
                let p = unsafe { Vec2 {
                    x: (*button_release).event_x as f32,
                    y: (*button_release).event_y as f32,
                } };
                let xcb_window = unsafe { *button_release }.event;
                match unsafe { *button_release }.detail {
                    1 => { return Some((xcb_window,Event::Pointer(PointerEvent::Up { position: p,button: Button::Left, }))); },
                    2 => { return Some((xcb_window,Event::Pointer(PointerEvent::Up { position: p,button: Button::Middle, }))); },
                    3 => { return Some((xcb_window,Event::Pointer(PointerEvent::Up { position: p,button: Button::Right, }))); },
                    _ => { },
                }        
            },
            sys::XCB_MOTION_NOTIFY => {
                let motion_notify = xcb_event as *const sys::xcb_motion_notify_event_t;
                let p = Vec2 {
                    x: unsafe { *motion_notify }.event_x as f32,
                    y: unsafe { *motion_notify }.event_y as f32,
                };
                let xcb_window = unsafe { *motion_notify }.event;
                return Some((xcb_window,Event::Pointer(PointerEvent::Move { position: p,buttons: Vec::new(),hover: false, })));
            },
            sys::XCB_CONFIGURE_NOTIFY => {
                let configure_notify = xcb_event as *const sys::xcb_configure_notify_event_t;
                let r = Rect {
                    o: Vec2 {
                        x: unsafe { *configure_notify }.x as f32,
                        y: unsafe { *configure_notify }.y as f32,
                    },
                    s: Vec2 {
                        x: unsafe { *configure_notify }.width as f32,
                        y: unsafe { *configure_notify }.height as f32,
                    },
                };
                let xcb_window = unsafe { *configure_notify }.window;
                return Some((xcb_window,Event::Configure(r)));
            },
            sys::XCB_CLIENT_MESSAGE => {
                let client_message = xcb_event as *const sys::xcb_client_message_event_t;
                let atom = unsafe { (*client_message).data.data32[0] };
                if atom == self.wm_delete_window {
                    let xcb_window = unsafe { *client_message }.window;
                    return Some((xcb_window,Event::Close));
                }
            },
            _ => {
            },
        }
        None
    }

    /// Get all OS window events that have gathered.

    // TODO: this should be combined with a regular async handler/loop
    pub fn flush(&self) -> Vec<(u32,Event)> {
        let mut events = Vec::<(u32,Event)>::new();
        loop {
            let event = unsafe { sys::xcb_poll_for_event(self.xcb_connection) };
            if event != null_mut() {
                if let Some((window_id,event)) = self.translate_xevent(event) {
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

    // TODO: this should be combined with a regular async handler/loop
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
    fn create_window(self: &Rc<System>,r: Rect<f32>,_absolute: bool) -> Result<Window,String> {

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
            sys::XCB_COPY_FROM_PARENT,
        ];
        unsafe {
            sys::xcb_create_window(
                self.xcb_connection,
                (*self.xcb_screen).root_depth,
                xcb_window as u32,
                (*self.xcb_screen).root,
                r.o.x as i16,
                r.o.y as i16,
                r.s.x as u16,
                r.s.y as u16,
                0,
                sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                (*self.xcb_screen).root_visual,
                sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
                &values as *const u32 as *const c_void
            );
            sys::xcb_map_window(self.xcb_connection,xcb_window as u32);
            sys::xcb_flush(self.xcb_connection);
        }

        // create GPU-specific portion of the window
        let gpu_window = GpuWindow::create(&self.gpu_system,r,self.xcb_connection,xcb_window)?;

        Ok(Window {
            system: Rc::clone(&self),
            gpu_window,
            xcb_window,
        })
    }
    
    /// Create application frame window (with frame and title bar).
    pub fn create_frame_window(self: &Rc<System>,r: Rect<f32>,title: &str) -> Result<Window,String> {
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
        Ok(window)
    }
    
    /// Create standalone popup window (no frame or title bar).
    pub fn create_popup_window(self: &Rc<System>,r: Rect<f32>) -> Result<Window,String> {
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
        Ok(window)
    }
}

impl Drop for System {

    /// Drop the system interface.
    fn drop(&mut self) {
        unsafe { sys::XCloseDisplay(self.xdisplay) };
    }
}
