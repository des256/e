// E - Linux - System
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        os::{
            raw::{
                c_int,
            },
        },
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
        cell::RefCell,
        collections::HashMap,
    },
    sys_sys::*,
    libc::{
        epoll_create1,
        epoll_ctl,
        EPOLL_CTL_ADD,
        epoll_event,
        EPOLLIN,
        epoll_wait,
    },
};

pub(crate) struct XCBAtoms {
    pub wm_protocols: u32,
    pub wm_delete_window: u32,
    pub wm_motif_hints: u32,
    pub _wm_transient_for: u32,
    pub _wm_net_type: u32,
    pub _wm_net_type_utility: u32,
    pub _wm_net_type_dropdown_menu: u32,
    pub wm_net_state: u32,
    pub wm_net_state_above: u32,
}

/// Main system context.
pub struct System {
    pub xcb_connection: *mut xcb_connection_t,
    epfd: c_int,
    pub(crate) xcb_window_pointers: RefCell<HashMap<xcb_window_t,*const Window>>,
    pub(crate) xcb_atoms: XCBAtoms,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_instance: VkInstance,
}

fn xcb_intern_atom_(connection: *mut xcb_connection_t,name: &str) -> xcb_intern_atom_cookie_t {
    let i8_name = unsafe { std::mem::transmute::<_,&[i8]>(name.as_bytes()) };
    unsafe { xcb_intern_atom(connection,false as u8,name.len() as u16,i8_name.as_ptr()) }
}

fn xcb_obtain_atoms(xcb_connection: *mut xcb_connection_t) -> XCBAtoms {

    let protocols_cookie = xcb_intern_atom_(xcb_connection,"WM_PROTOCOLS");
    let delete_window_cookie = xcb_intern_atom_(xcb_connection,"WM_DELETE_WINDOW");
    let motif_hints_cookie = xcb_intern_atom_(xcb_connection,"_MOTIF_WM_HINTS");
    let transient_for_cookie = xcb_intern_atom_(xcb_connection,"WM_TRANSIENT_FOR");
    let net_type_cookie = xcb_intern_atom_(xcb_connection,"_NET_WM_TYPE");
    let net_type_utility_cookie = xcb_intern_atom_(xcb_connection,"_NET_WM_TYPE_UTILITY");
    let net_type_dropdown_menu_cookie = xcb_intern_atom_(xcb_connection,"_NET_WM_TYPE_DROPDOWN_MENU");
    let net_state_cookie = xcb_intern_atom_(xcb_connection,"_NET_WM_STATE");
    let net_state_above_cookie = xcb_intern_atom_(xcb_connection,"_NET_WM_STATE_ABOVE");

    let wm_protocols = unsafe { (*xcb_intern_atom_reply(xcb_connection,protocols_cookie,null_mut())).atom };
    let wm_delete_window = unsafe { (*xcb_intern_atom_reply(xcb_connection,delete_window_cookie,null_mut())).atom };
    let wm_motif_hints = unsafe { (*xcb_intern_atom_reply(xcb_connection,motif_hints_cookie,null_mut())).atom };
    let wm_transient_for = unsafe { (*xcb_intern_atom_reply(xcb_connection,transient_for_cookie,null_mut())).atom };
    let wm_net_type = unsafe { (*xcb_intern_atom_reply(xcb_connection,net_type_cookie,null_mut())).atom };
    let wm_net_type_utility = unsafe { (*xcb_intern_atom_reply(xcb_connection,net_type_utility_cookie,null_mut())).atom };
    let wm_net_type_dropdown_menu = unsafe { (*xcb_intern_atom_reply(xcb_connection,net_type_dropdown_menu_cookie,null_mut())).atom };
    let wm_net_state = unsafe { (*xcb_intern_atom_reply(xcb_connection,net_state_cookie,null_mut())).atom };
    let wm_net_state_above = unsafe { (*xcb_intern_atom_reply(xcb_connection,net_state_above_cookie,null_mut())).atom };

    XCBAtoms {
        wm_protocols: wm_protocols,
        wm_delete_window: wm_delete_window,
        wm_motif_hints: wm_motif_hints,
        _wm_transient_for: wm_transient_for,
        _wm_net_type: wm_net_type,
        _wm_net_type_utility: wm_net_type_utility,
        _wm_net_type_dropdown_menu: wm_net_type_dropdown_menu,
        wm_net_state: wm_net_state,
        wm_net_state_above: wm_net_state_above,
    }
}

impl System {

    pub fn new() -> Option<Rc<System>> {

        let xcb_connection = unsafe { xcb_connect(null_mut(),null_mut()) };
        if xcb_connection == null_mut() {
#[cfg(feature="debug_output")]
            println!("Unable to connect to X server.");
            return None;
        }
        let fd = unsafe { xcb_get_file_descriptor(xcb_connection) };

        let epfd = unsafe { epoll_create1(0) };
        let mut epe = [epoll_event { events: EPOLLIN as u32,u64: 0, }];
        unsafe { epoll_ctl(epfd,EPOLL_CTL_ADD,fd,epe.as_mut_ptr()) };

#[cfg(feature="gpu_vulkan")]
        let vk_instance = {
            let _application = VkApplicationInfo {
                sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pNext: null_mut(),
                pApplicationName: b"E::System\0".as_ptr() as *const i8,
                applicationVersion: (1 << 22) as u32,
                pEngineName: b"E::GPU\0".as_ptr() as *const i8,
                engineVersion: (1 << 22) as u32,
                apiVersion: ((1 << 22) | (2 << 11)) as u32,
            };
            let extension_names = [
                VK_KHR_SURFACE_EXTENSION_NAME.as_ptr(),
                VK_KHR_XCB_SURFACE_EXTENSION_NAME.as_ptr(),
            ];
            let info = VkInstanceCreateInfo {
                sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                //pApplicationInfo: &application,
                pApplicationInfo: &VkApplicationInfo {
                    sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
                    pNext: null_mut(),
                    pApplicationName: b"E::System\0".as_ptr() as *const i8,
                    applicationVersion: (1 << 22) as u32,
                    pEngineName: b"E::GPU\0".as_ptr() as *const i8,
                    engineVersion: (1 << 22) as u32,
                    apiVersion: ((1 << 22) | (2 << 11)) as u32,
                },
                enabledExtensionCount: extension_names.len() as u32,
                ppEnabledExtensionNames: extension_names.as_ptr() as *const *const i8,
                enabledLayerCount: 0,
                flags: 0,
                pNext: null_mut(),
                ppEnabledLayerNames: null_mut(),
            };
            let mut vk_instance = MaybeUninit::uninit();
            match unsafe { vkCreateInstance(&info,null_mut(),vk_instance.as_mut_ptr()) } {
                VK_SUCCESS => {
                    unsafe { vk_instance.assume_init() }
                },
                code => {
#[cfg(feature="debug_output")]
                    println!("Unable to create Vulkan instance (error {}).",code);
                    return None;
                },
            }
        };

        Some(Rc::new(System {
            xcb_connection: xcb_connection,
            epfd: epfd,
            xcb_window_pointers: RefCell::new(HashMap::new()),
            xcb_atoms: xcb_obtain_atoms(xcb_connection),
#[cfg(feature="gpu_vulkan")]
            vk_instance: vk_instance,
        }))
    }

    fn translate_event(&self,xcb_event: *mut xcb_generic_event_t) -> Option<(xcb_window_t,Event)> {
        match (unsafe { *xcb_event }.response_type & 0x7F) as u32 {
            XCB_EXPOSE => {
                let expose = xcb_event as *const xcb_expose_event_t;
                //let expose = unsafe { std::mem::transmute::<_,xcb_expose_event_t>(xcb_event) };
                //let r = rect!(expose.x as isize,expose.y as isize,expose.width() as isize,expose.height() as isize);
                let xcb_window = unsafe { *expose }.window;
                return Some((xcb_window,Event::Render));
            },
            XCB_KEY_PRESS => {
                let key_press = xcb_event as *const xcb_key_press_event_t;
                let xcb_window = unsafe { *key_press }.event;
                return Some((xcb_window,Event::KeyPress(unsafe { *key_press }.detail as u8)));
            },
            XCB_KEY_RELEASE => {
                let key_release = xcb_event as *const xcb_key_release_event_t;
                let xcb_window = unsafe { *key_release }.event;
                return Some((xcb_window,Event::KeyRelease(unsafe { *key_release }.detail as u8)));
            },
            XCB_BUTTON_PRESS => {
                let button_press = xcb_event as *const xcb_button_press_event_t;
                let p = Vec2 { x: unsafe { *button_press }.event_x as i32,y: unsafe { *button_press }.event_y as i32, };
                let xcb_window = unsafe { *button_press }.event;
                match unsafe { *button_press }.detail {
                    1 => { return Some((xcb_window,Event::MousePress(p,MouseButton::Left))); },
                    2 => { return Some((xcb_window,Event::MousePress(p,MouseButton::Middle))); },
                    3 => { return Some((xcb_window,Event::MousePress(p,MouseButton::Right))); },
                    4 => { return Some((xcb_window,Event::MouseWheel(MouseWheel::Up))); },
                    5 => { return Some((xcb_window,Event::MouseWheel(MouseWheel::Down))); },
                    6 => { return Some((xcb_window,Event::MouseWheel(MouseWheel::Left))); },
                    7 => { return Some((xcb_window,Event::MouseWheel(MouseWheel::Right))); },
                    _ => { },
                }        
            },
            XCB_BUTTON_RELEASE => {
                let button_release = xcb_event as *const xcb_button_release_event_t;
                let p = Vec2 {
                    x: unsafe { *button_release }.event_x as i32,
                    y: unsafe { *button_release }.event_y as i32,
                };
                let xcb_window = unsafe { *button_release }.event;
                match unsafe { *button_release }.detail {
                    1 => { return Some((xcb_window,Event::MouseRelease(p,MouseButton::Left))); },
                    2 => { return Some((xcb_window,Event::MouseRelease(p,MouseButton::Middle))); },
                    3 => { return Some((xcb_window,Event::MouseRelease(p,MouseButton::Right))); },
                    _ => { },
                }        
            },
            XCB_MOTION_NOTIFY => {
                let motion_notify = xcb_event as *const xcb_motion_notify_event_t;
                let p = Vec2 {
                    x: unsafe { *motion_notify }.event_x as i32,
                    y: unsafe { *motion_notify }.event_y as i32,
                };
                let xcb_window = unsafe { *motion_notify }.event;
                return Some((xcb_window,Event::MouseMove(p)));
            },
            XCB_CONFIGURE_NOTIFY => {
                let configure_notify = xcb_event as *const xcb_configure_notify_event_t;
                let r = Rect::new(
                    unsafe { *configure_notify }.x as i32,
                    unsafe { *configure_notify }.y as i32,
                    unsafe { *configure_notify }.width as i32,
                    unsafe { *configure_notify }.height as i32
                );
                let xcb_window = unsafe { *configure_notify }.window;
                return Some((xcb_window,Event::Configure(r)));
            },
            XCB_CLIENT_MESSAGE => {
                let client_message = xcb_event as *const xcb_client_message_event_t;
                let atom = unsafe { (*client_message).data.data32[0] };
                if atom == self.xcb_atoms.wm_delete_window {
                    let xcb_window = unsafe { *client_message }.window;
                    return Some((xcb_window,Event::Close));
                }
            },
            _ => {
            },
        }
        None
    }

    pub fn flush(&self) {
        loop {
            let event = unsafe { xcb_poll_for_event(self.xcb_connection) };
            if event != null_mut() {
                if let Some((xcb_window,event)) = self.translate_event(event) {
                    let window_pointers = self.xcb_window_pointers.borrow();
                    if window_pointers.contains_key(&xcb_window) {
                        unsafe { (*window_pointers[&xcb_window]).handle_event(event); }
                    }
                }
            }
            else {
                break;
            }
        }
    }

    pub fn wait(&self) {
        let mut epe = [epoll_event { events: EPOLLIN as u32,u64: 0, }];
        unsafe { epoll_wait(self.epfd,epe.as_mut_ptr(),1,-1) };
    }

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
    
    pub fn release_mouse(&self) {
        //println!("XUngrabPointer");
        //ungrab_pointer(&self.connection,TIME_CURRENT_TIME);
    }

    pub fn set_mousecursor(&self,_id: u64,_n: usize) {
        //let values = [(CW_CURSOR,self.cursors[n])];
        //change_window_attributes(&self.connection,id as u32,&values);
    }
}

impl Drop for System {
    fn drop(&mut self) {
#[cfg(feature="gpu_vulkan")]
        unsafe { vkDestroyInstance(self.vk_instance,null_mut()) };
        // TODO: close connection
    }
}
