// E - Linux - System
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        os::raw::c_int,
        ptr::null_mut,
    },
};

#[cfg(target_os="linux")]
use {
    libc::{
        epoll_create1,
        epoll_ctl,
        EPOLL_CTL_ADD,
        epoll_event,
        EPOLLIN,
        epoll_wait,
    },
};

#[cfg(feature="gpu_vulkan")]
use std::mem::MaybeUninit;

#[cfg(feature="gpu_opengl")]
use {
    std::{
        os::raw::c_void,
        ffi::CString,
    },
};

#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
#[derive(Clone)]
pub(crate) struct QueueFamilyDescriptor {
    pub index: usize,
    pub graphics: bool,
    pub transfer: bool,
    pub compute: bool,
    pub sparse_binding: bool,
    pub count: usize,
}

#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
#[derive(Clone)]
pub(crate) struct GpuDescriptor {
    pub name: String,
    pub queue_family_descriptors: Vec<QueueFamilyDescriptor>,
    pub vk_physical_device: sys::VkPhysicalDevice,
}

/// The system structure (linux).
pub struct System {
    pub(crate) xdisplay: *mut sys::Display,
    pub(crate) xcb_connection: *mut sys::xcb_connection_t,
    pub(crate) epfd: c_int,
    pub(crate) wm_protocols: u32,
    pub(crate) wm_delete_window: u32,
    pub(crate) wm_motif_hints: u32,
    pub(crate) _wm_transient_for: u32,
    pub(crate) _wm_net_type: u32,
    pub(crate) _wm_net_type_utility: u32,
    pub(crate) _wm_net_type_dropdown_menu: u32,
    pub(crate) wm_net_state: u32,
    pub(crate) wm_net_state_above: u32,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_instance: sys::VkInstance,
#[cfg(feature="gpu_vulkan")]
    pub(crate) gpu_descriptors: Vec<GpuDescriptor>,
}

#[doc(hidden)]
#[cfg(feature="opengl")]
type GlXCreateContextAttribsARBProc = unsafe extern "C" fn(
    dpy: *mut sys::Display,
    fbc: sys::GLXFBConfig,
    share_context: sys::GLXContext,
    direct: i32,
    attribs: *const c_int
) -> sys::GLXContext;

#[doc(hidden)]
#[cfg(feature="opengl")]
fn load_gl_function(name: &str) -> *mut c_void {
    let newname = CString::new(name).unwrap();
    let pointer: *mut c_void = unsafe { std::mem::transmute(sys::glXGetProcAddress(newname.as_ptr() as *const u8)) };
    if pointer.is_null() { panic!("Canvas: unable to access {}", name); }
    pointer
}

#[doc(hidden)]
fn xcb_intern_atom(connection: *mut sys::xcb_connection_t,name: &str) -> sys::xcb_intern_atom_cookie_t {
    let i8_name = unsafe { std::mem::transmute::<_,&[i8]>(name.as_bytes()) };
    unsafe { sys::xcb_intern_atom(connection,false as u8,name.len() as u16,i8_name.as_ptr()) }
}

/// Open the system interface.
pub fn open_system() -> Option<System> {

    // open X connection
    let xdisplay = unsafe { sys::XOpenDisplay(null_mut()) };
    if xdisplay == null_mut() {
        print!("Unable to connect to X server.");
        return None;
    }
    let xcb_connection = unsafe { sys::XGetXCBConnection(xdisplay) };
    if xcb_connection == null_mut() {
        print!("Unable to connect to X server.");
        return None;
    }

    // we want to use XCB only (but might need X11)
    unsafe { sys::XSetEventQueueOwner(xdisplay,sys::XCBOwnsEventQueue) };

    // create epoll descriptor to be able to wait for UI events
    let fd = unsafe { sys::xcb_get_file_descriptor(xcb_connection) };
    let epfd = unsafe { epoll_create1(0) };
    let mut epe = [epoll_event { events: EPOLLIN as u32,u64: 0, }];
    unsafe { epoll_ctl(epfd,EPOLL_CTL_ADD,fd,epe.as_mut_ptr()) };

    // get the atoms
    let protocols_cookie = xcb_intern_atom(xcb_connection,"WM_PROTOCOLS");
    let delete_window_cookie = xcb_intern_atom(xcb_connection,"WM_DELETE_WINDOW");
    let motif_hints_cookie = xcb_intern_atom(xcb_connection,"_MOTIF_WM_HINTS");
    let transient_for_cookie = xcb_intern_atom(xcb_connection,"WM_TRANSIENT_FOR");
    let net_type_cookie = xcb_intern_atom(xcb_connection,"_NET_WM_TYPE");
    let net_type_utility_cookie = xcb_intern_atom(xcb_connection,"_NET_WM_TYPE_UTILITY");
    let net_type_dropdown_menu_cookie = xcb_intern_atom(xcb_connection,"_NET_WM_TYPE_DROPDOWN_MENU");
    let net_state_cookie = xcb_intern_atom(xcb_connection,"_NET_WM_STATE");
    let net_state_above_cookie = xcb_intern_atom(xcb_connection,"_NET_WM_STATE_ABOVE");

    let wm_protocols = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,protocols_cookie,null_mut())).atom };
    let wm_delete_window = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,delete_window_cookie,null_mut())).atom };
    let wm_motif_hints = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,motif_hints_cookie,null_mut())).atom };
    let wm_transient_for = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,transient_for_cookie,null_mut())).atom };
    let wm_net_type = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,net_type_cookie,null_mut())).atom };
    let wm_net_type_utility = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,net_type_utility_cookie,null_mut())).atom };
    let wm_net_type_dropdown_menu = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,net_type_dropdown_menu_cookie,null_mut())).atom };
    let wm_net_state = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,net_state_cookie,null_mut())).atom };
    let wm_net_state_above = unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,net_state_above_cookie,null_mut())).atom };

    // open GPU connection
#[cfg(feature="gpu_vulkan")]
    let (vk_instance,gpu_descriptors) = {

        // create instance
        let _application = sys::VkApplicationInfo {
            sType: sys::VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: null_mut(),
            pApplicationName: b"E::System\0".as_ptr() as *const i8,
            applicationVersion: (1 << 22) as u32,
            pEngineName: b"E::GPU\0".as_ptr() as *const i8,
            engineVersion: (1 << 22) as u32,
            apiVersion: ((1 << 22) | (2 << 11)) as u32,
        };
        let extension_names = [
            sys::VK_KHR_SURFACE_EXTENSION_NAME.as_ptr(),
            sys::VK_KHR_XCB_SURFACE_EXTENSION_NAME.as_ptr(),
        ];
        let info = sys::VkInstanceCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            //pApplicationInfo: &application,
            pApplicationInfo: &sys::VkApplicationInfo {
                sType: sys::VK_STRUCTURE_TYPE_APPLICATION_INFO,
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
        let mut vk_instance: sys::VkInstance = null_mut();
        let code = unsafe { sys::vkCreateInstance(&info,null_mut(),&mut vk_instance as *mut sys::VkInstance) };
        if code != sys::VK_SUCCESS {
            println!("Unable to create instance (error {}).",code);
            return None;
        }

        // create physical device list
        let mut count = 0u32;
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&mut count,null_mut()) };
        if count == 0 {
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            println!("Unable to find physical devices.");
            return None;
        }
        let mut vk_physical_devices = vec![null_mut() as sys::VkPhysicalDevice; count as usize];
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&mut count,vk_physical_devices.as_mut_ptr()) };
        let mut gpu_descriptors: Vec<GpuDescriptor> = Vec::new();
        for vk_physical_device in &vk_physical_devices {

            // get GPU properties
            let mut properties = MaybeUninit::<sys::VkPhysicalDeviceProperties>::uninit();
            unsafe { sys::vkGetPhysicalDeviceProperties(*vk_physical_device,properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            let slice: &[u8] = unsafe { &*(&properties.deviceName as *const [i8] as *const [u8]) };
            let name = std::str::from_utf8(slice).unwrap();

            // get supported queue families
            let mut queue_family_descriptors: Vec<QueueFamilyDescriptor> = Vec::new();
            let mut count = 0u32;
            unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(*vk_physical_device,&mut count,null_mut()) };
            if count > 0 {
                let mut vk_queue_families = vec![sys::VkQueueFamilyProperties {
                    queueFlags: 0,
                    queueCount: 0,
                    timestampValidBits: 0,
                    minImageTransferGranularity: sys::VkExtent3D {
                        width: 0,
                        height: 0,
                        depth: 0,
                    },
                }; count as usize];
                unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(*vk_physical_device,&mut count,vk_queue_families.as_mut_ptr()) };
                let mut index = 0usize;
                for vk_queue_family in &vk_queue_families {
                    queue_family_descriptors.push(QueueFamilyDescriptor {
                        index: index,
                        graphics: vk_queue_family.queueFlags & sys::VK_QUEUE_GRAPHICS_BIT != 0,
                        transfer: vk_queue_family.queueFlags & sys::VK_QUEUE_TRANSFER_BIT != 0,
                        compute: vk_queue_family.queueFlags & sys::VK_QUEUE_COMPUTE_BIT != 0,
                        sparse_binding: vk_queue_family.queueFlags & sys::VK_QUEUE_SPARSE_BINDING_BIT != 0,
                        count: vk_queue_family.queueCount as usize,
                    });
                    index += 1;
                }
            }

            // and add GPU to the list
            gpu_descriptors.push(GpuDescriptor {
                name: name.to_string(),
                vk_physical_device: *vk_physical_device,
                queue_family_descriptors: queue_family_descriptors,
            });
        }

        (vk_instance,gpu_descriptors)
    };

    Some(System {
        xdisplay: xdisplay,
        xcb_connection: xcb_connection,
        epfd: epfd,
        wm_protocols: wm_protocols,
        wm_delete_window: wm_delete_window,
        wm_motif_hints: wm_motif_hints,
        _wm_transient_for: wm_transient_for,
        _wm_net_type: wm_net_type,
        _wm_net_type_utility: wm_net_type_utility,
        _wm_net_type_dropdown_menu: wm_net_type_dropdown_menu,
        wm_net_state: wm_net_state,
        wm_net_state_above: wm_net_state_above,
#[cfg(feature="gpu_vulkan")]
        vk_instance: vk_instance,
#[cfg(feature="gpu_vulkan")]
        gpu_descriptors: gpu_descriptors,
    })
}

impl<'system> System {

    /// Enumerate all available GPUs on this system.
    pub fn enumerate_gpus(&self) -> Vec<String> {
        let mut gpus: Vec<String> = Vec::new();
#[cfg(feature="gpu_vulkan")]
        {
            for gpu_descriptor in &self.gpu_descriptors {
                gpus.push(gpu_descriptor.name.clone());
            }
        }
#[cfg(feature="gpu_opengl")]
        {
            gpus.push("Default GPU");
        }
        gpus
    }

#[doc(hidden)]
    fn translate_event(&self,xcb_event: *mut sys::xcb_generic_event_t) -> Option<(WindowId,Event)> {
        match (unsafe { *xcb_event }.response_type & 0x7F) as u32 {
            sys::XCB_EXPOSE => {
                let expose = xcb_event as *const sys::xcb_expose_event_t;
                //let expose = unsafe { std::mem::transmute::<_,xcb_expose_event_t>(xcb_event) };
                //let r = rect!(expose.x as isize,expose.y as isize,expose.width() as isize,expose.height() as isize);
                let xcb_window = unsafe { *expose }.window;
                return Some((xcb_window as WindowId,Event::Render));
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
                let p = Vec2 { x: unsafe { *button_press }.event_x as i32,y: unsafe { *button_press }.event_y as i32, };
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
                let p = Vec2 {
                    x: unsafe { *button_release }.event_x as i32,
                    y: unsafe { *button_release }.event_y as i32,
                };
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
                    x: unsafe { *motion_notify }.event_x as i32,
                    y: unsafe { *motion_notify }.event_y as i32,
                };
                let xcb_window = unsafe { *motion_notify }.event;
                return Some((xcb_window as WindowId,Event::MouseMove(p)));
            },
            sys::XCB_CONFIGURE_NOTIFY => {
                let configure_notify = xcb_event as *const sys::xcb_configure_notify_event_t;
                let r = Rect::new(
                    unsafe { *configure_notify }.x as i32,
                    unsafe { *configure_notify }.y as i32,
                    unsafe { *configure_notify }.width as i32,
                    unsafe { *configure_notify }.height as i32
                );
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
        let mut epe = [epoll_event { events: EPOLLIN as u32,u64: 0, }];
        unsafe { epoll_wait(self.epfd,epe.as_mut_ptr(),1,-1) };
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
}

impl Drop for System {

    /// Drop the system interface.
    fn drop(&mut self) {
#[cfg(feature="gpu_vulkan")]
        unsafe { sys::vkDestroyInstance(self.vk_instance,null_mut()) };
        
        unsafe { sys::XCloseDisplay(self.xdisplay) };
    }
}
