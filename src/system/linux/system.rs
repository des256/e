use {
    crate::*,
    std::{
        os::raw::c_int,
        ptr::null_mut,
        mem::transmute,
    },
};

#[cfg(gpu="vulkan")]
use std::mem::MaybeUninit;

/// The system structure (linux).
pub struct System {
    pub(crate) xdisplay: *mut sys::Display,
    pub(crate) xcb_connection: *mut sys::xcb_connection_t,
    pub(crate) xcb_screen: *mut sys::xcb_screen_t,
    pub(crate) epfd: c_int,
    pub(crate) wm_protocols: u32,
    pub(crate) wm_delete_window: u32,
    pub(crate) wm_motif_hints: u32,
    pub(crate) wm_transient_for: u32,
    pub(crate) wm_net_type: u32,
    pub(crate) wm_net_type_utility: u32,
    pub(crate) wm_net_type_dropdown_menu: u32,
    pub(crate) wm_net_state: u32,
    pub(crate) wm_net_state_above: u32,
#[cfg(gpu="vulkan")]
    pub(crate) vk_instance: sys::VkInstance,
#[cfg(gpu="vulkan")]
    pub(crate) vk_physical_device: sys::VkPhysicalDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_device: sys::VkDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_command_pool: sys::VkCommandPool,
#[cfg(gpu="vulkan")]
    pub(crate) shared_index: usize,
}

fn intern_atom_cookie(xcb_connection: *mut sys::xcb_connection_t,name: &str) -> sys::xcb_intern_atom_cookie_t {
    let i8_name = unsafe { std::mem::transmute::<_,&[i8]>(name.as_bytes()) };
    unsafe { sys::xcb_intern_atom(xcb_connection,0,name.len() as u16,i8_name.as_ptr()) }
}

fn resolve_atom_cookie(xcb_connection: *mut sys::xcb_connection_t,cookie: sys::xcb_intern_atom_cookie_t) -> u32 {
    unsafe { (*sys::xcb_intern_atom_reply(xcb_connection,cookie,null_mut())).atom }
}

/// Open the system interface.
pub fn open_system() -> Option<System> {

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
    let xcb_screen = unsafe { sys::xcb_setup_roots_iterator(xcb_setup) }.data;

    // create epoll descriptor to be able to wait for UI events on a system level
    let fd = unsafe { sys::xcb_get_file_descriptor(xcb_connection) };
    let epfd = unsafe { sys::epoll_create1(0) };
    let mut epe = [sys::epoll_event { events: sys::EPOLLIN as u32,data: sys::epoll_data_t { u64_: 0, }, }];
    unsafe { sys::epoll_ctl(epfd,sys::EPOLL_CTL_ADD as i32,fd,epe.as_mut_ptr()) };

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

    // GPU-specific stuff
#[cfg(gpu="vulkan")]
    let (vk_instance,vk_physical_device,vk_device,vk_queue,vk_command_pool,shared_index) = {

        // create instance
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
                pApplicationName: b"e::System\0".as_ptr() as *const i8,
                applicationVersion: (1 << 22) as u32,
                pEngineName: b"e::Gpu\0".as_ptr() as *const i8,
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
        let mut vk_instance = MaybeUninit::<sys::VkInstance>::uninit();
        match unsafe { sys::vkCreateInstance(&info,null_mut(),vk_instance.as_mut_ptr()) } {
            sys::VK_SUCCESS => { }
            code => {
                println!("unable to create VkInstance ({})",code);
                return None;
            },
        }
        let vk_instance = unsafe { vk_instance.assume_init() };

        // enumerate physical devices
        let mut count = MaybeUninit::<u32>::uninit();
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,count.as_mut_ptr(),null_mut()) };
        let count = unsafe { count.assume_init() };
        if count == 0 {
            println!("unable to enumerate physical devices");
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return None;
        }
        let mut vk_physical_devices = vec![null_mut() as sys::VkPhysicalDevice; count as usize];
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&count as *const u32 as *mut u32,vk_physical_devices.as_mut_ptr()) };

        // get first physical device
        let vk_physical_device = vk_physical_devices[0];

        // DEBUG: show the name in debug build
#[cfg(build="debug")]
        {
            let mut properties = MaybeUninit::<sys::VkPhysicalDeviceProperties>::uninit();
            unsafe { sys::vkGetPhysicalDeviceProperties(vk_physical_device,properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            let slice: &[u8] = unsafe { &*(&properties.deviceName as *const [i8] as *const [u8]) };
            dprintln!("physical device: {}",std::str::from_utf8(slice).unwrap());
        }
            
        // get supported queue families
        let mut count = MaybeUninit::<u32>::uninit();
        unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device,count.as_mut_ptr(),null_mut()) };
        let count = unsafe { count.assume_init() };
        if count == 0 {
            println!("no queue families supported on this GPU");
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return None;
        }
        let mut vk_queue_families = vec![MaybeUninit::<sys::VkQueueFamilyProperties>::uninit(); count as usize];
        unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(
            vk_physical_device,
            &count as *const u32 as *mut u32,
            vk_queue_families.as_mut_ptr() as *mut sys::VkQueueFamilyProperties,
        ) };
        let vk_queue_families = unsafe { transmute::<_,Vec<sys::VkQueueFamilyProperties>>(vk_queue_families) };

        // DEBUG: display the number of queues and capabilities
#[cfg(build="debug")]
        for i in 0..vk_queue_families.len() {
            let mut capabilities = String::new();
            if vk_queue_families[i].queueFlags & sys::VK_QUEUE_GRAPHICS_BIT != 0 {
                capabilities.push_str("graphics ");
            }
            if vk_queue_families[i].queueFlags & sys::VK_QUEUE_TRANSFER_BIT != 0 {
                capabilities.push_str("transfer ");
            }
            if vk_queue_families[i].queueFlags & sys::VK_QUEUE_COMPUTE_BIT != 0 {
                capabilities.push_str("compute ");
            }
            if vk_queue_families[i].queueFlags & sys::VK_QUEUE_SPARSE_BINDING_BIT != 0 {
                capabilities.push_str("sparse ");
            }
            dprintln!("    {}: {} queues, capable of: {}",i,vk_queue_families[i].queueCount,capabilities);
        }

        // assume the first queue family is the one we want for all queues
        let vk_queue_family = vk_queue_families[0];
        let mask = sys::VK_QUEUE_GRAPHICS_BIT | sys::VK_QUEUE_TRANSFER_BIT | sys::VK_QUEUE_COMPUTE_BIT;
        if (vk_queue_family.queueFlags & mask) != mask {
            println!("queue family 0 of the GPU does not support graphics, transfer and compute operations");
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return None;
        }

        // assume that presentation is done on the same family as graphics and create logical device with one queue of queue family 0
        let mut queue_create_infos = Vec::<sys::VkDeviceQueueCreateInfo>::new();
        let priority = 1f32;
        queue_create_infos.push(sys::VkDeviceQueueCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueFamilyIndex: 0,
            queueCount: 1,
            pQueuePriorities: &priority as *const f32,
        });
        let extension_names = [
            sys::VK_KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr(),
        ];
        let info = sys::VkDeviceCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueCreateInfoCount: queue_create_infos.len() as u32,
            pQueueCreateInfos: queue_create_infos.as_mut_ptr(),
            enabledLayerCount: 0,
            ppEnabledLayerNames: null_mut(),
            enabledExtensionCount: extension_names.len() as u32,
            ppEnabledExtensionNames: extension_names.as_ptr() as *const *const i8,
            pEnabledFeatures: &sys::VkPhysicalDeviceFeatures {
                robustBufferAccess: 0,
                fullDrawIndexUint32: 0,
                imageCubeArray: 0,
                independentBlend: 0,
                geometryShader: 0,
                tessellationShader: 0,
                sampleRateShading: 0,
                dualSrcBlend: 0,
                logicOp: 0,
                multiDrawIndirect: 0,
                drawIndirectFirstInstance: 0,
                depthClamp: 0,
                depthBiasClamp: 0,
                fillModeNonSolid: 0,
                depthBounds: 0,
                wideLines: 0,
                largePoints: 0,
                alphaToOne: 0,
                multiViewport: 0,
                samplerAnisotropy: 0,
                textureCompressionETC2: 0,
                textureCompressionASTC_LDR: 0,
                textureCompressionBC: 0,
                occlusionQueryPrecise: 0,
                pipelineStatisticsQuery: 0,
                vertexPipelineStoresAndAtomics: 0,
                fragmentStoresAndAtomics: 0,
                shaderTessellationAndGeometryPointSize: 0,
                shaderImageGatherExtended: 0,
                shaderStorageImageExtendedFormats: 0,
                shaderStorageImageMultisample: 0,
                shaderStorageImageReadWithoutFormat: 0,
                shaderStorageImageWriteWithoutFormat: 0,
                shaderUniformBufferArrayDynamicIndexing: 0,
                shaderSampledImageArrayDynamicIndexing: 0,
                shaderStorageBufferArrayDynamicIndexing: 0,
                shaderStorageImageArrayDynamicIndexing: 0,
                shaderClipDistance: 0,
                shaderCullDistance: 0,
                shaderFloat64: 0,
                shaderInt64: 0,
                shaderInt16: 0,
                shaderResourceResidency: 0,
                shaderResourceMinLod: 0,
                sparseBinding: 0,
                sparseResidencyBuffer: 0,
                sparseResidencyImage2D: 0,
                sparseResidencyImage3D: 0,
                sparseResidency2Samples: 0,
                sparseResidency4Samples: 0,
                sparseResidency8Samples: 0,
                sparseResidency16Samples: 0,
                sparseResidencyAliased: 0,
                variableMultisampleRate: 0,
                inheritedQueries: 0,
            },
        };
        let mut vk_device = MaybeUninit::uninit();
        if unsafe { sys::vkCreateDevice(vk_physical_device,&info,null_mut(),vk_device.as_mut_ptr()) } != sys::VK_SUCCESS {
            println!("unable to create VkDevice");
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return None;
        }
        let vk_device = unsafe { vk_device.assume_init() };

        // obtain the queue from queue family 0
        let mut vk_queue = MaybeUninit::uninit();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,0,vk_queue.as_mut_ptr()) };
        let vk_queue = unsafe { vk_queue.assume_init() };

        // create command pool for this queue
        let info = sys::VkCommandPoolCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: null_mut(),
            flags: sys::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: 0,
        };
        let mut vk_command_pool = MaybeUninit::uninit();
        if unsafe { sys::vkCreateCommandPool(vk_device,&info,null_mut(),vk_command_pool.as_mut_ptr()) } != sys::VK_SUCCESS {
            println!("unable to create command pool");
            unsafe { 
                //sys::vkDestroyDevice(vk_device,null_mut());
                //sys::vkDestroyInstance(vk_instance,null_mut());
            }
            return None;
        }
        let vk_command_pool = unsafe { vk_command_pool.assume_init() };

        // get memory properties
        let mut vk_memory_properties = MaybeUninit::<sys::VkPhysicalDeviceMemoryProperties>::uninit();
        unsafe { sys::vkGetPhysicalDeviceMemoryProperties(vk_physical_device,vk_memory_properties.as_mut_ptr()) };
        let vk_memory_properties = unsafe { vk_memory_properties.assume_init() };

        // DEBUG: show the entire memory description
#[cfg(build="debug")]
        {
            dprintln!("device memory properties:");
            dprintln!("    memory types:");
            for i in 0..vk_memory_properties.memoryTypeCount {
                let vk_memory_type = &vk_memory_properties.memoryTypes[i as usize];
                let mut flags = String::new();
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0 {
                    flags += "device_local ";
                }
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 {
                    flags += "host_visible ";
                }
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0 {
                    flags += "host_coherent ";
                }
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0 {
                    flags += "host_cached ";
                }
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) != 0 {
                    flags += "lazily_allocated ";
                }
                if (vk_memory_type.propertyFlags & sys::VK_MEMORY_PROPERTY_PROTECTED_BIT) != 0 {
                    flags += "protected ";
                }            
                dprintln!("        {}: on heap {}, {}",i,vk_memory_type.heapIndex,flags);
            }
            dprintln!("    memory heaps:");
            for i in 0..vk_memory_properties.memoryHeapCount {
                let vk_memory_heap = &vk_memory_properties.memoryHeaps[i as usize];
                dprintln!("        {}: size {} MiB, {:X}",i,vk_memory_heap.size / (1024 * 1024),vk_memory_heap.flags);
            }
        }

        // find shared memory heap and type (later also find device-only index)
        let mut shared_index: usize = 0;
        for i in 0..vk_memory_properties.memoryTypeCount {
            let flags = vk_memory_properties.memoryTypes[i as usize].propertyFlags;
            if ((flags & sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0) && ((flags & sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0) && ((flags & sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0) {
                shared_index = i as usize;
                break;
            }
        }

        (vk_instance,vk_physical_device,vk_device,vk_queue,vk_command_pool,shared_index)
    };

    Some(System {
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
#[cfg(gpu="vulkan")]
        vk_instance,
#[cfg(gpu="vulkan")]
        vk_physical_device,
#[cfg(gpu="vulkan")]
        vk_device,
#[cfg(gpu="vulkan")]
        vk_queue,
#[cfg(gpu="vulkan")]
        vk_command_pool,
#[cfg(gpu="vulkan")]
        shared_index,
    })
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
                let r = i32r::new(
                    i32xy::new(
                        unsafe { *configure_notify }.x as i32,
                        unsafe { *configure_notify }.y as i32,
                    ),
                    u32xy::new(
                        unsafe { *configure_notify }.width as u32,
                        unsafe { *configure_notify }.height as u32,
                    ),
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
}

impl Drop for System {

    /// Drop the system interface.
    fn drop(&mut self) {
        
        unsafe {
#[cfg(gpu="vulkan")]
            {
                sys::vkDestroyCommandPool(self.vk_device,self.vk_command_pool,null_mut());
                //sys::vkDestroyDevice(self.vk_device,null_mut());
                //sys::vkDestroyInstance(self.vk_instance,null_mut());
            }
            sys::XCloseDisplay(self.xdisplay);
        }
    }

    /*
    Attempt to free VkCommandBuffer 0x55b677443510[] which is in use. The
    Vulkan spec states: All elements of pCommandBuffers must not be in the
    pending state
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkFreeCommandBuffers-pCommandBuffers-00047)
    */
}
