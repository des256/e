// E - Linux - System
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        os::raw::c_int,
        ptr::null_mut,
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
    pub(crate) _wm_transient_for: u32,
    pub(crate) _wm_net_type: u32,
    pub(crate) _wm_net_type_utility: u32,
    pub(crate) _wm_net_type_dropdown_menu: u32,
    pub(crate) wm_net_state: u32,
    pub(crate) wm_net_state_above: u32,
#[cfg(gpu="vulkan")]
    pub(crate) vk_instance: sys::VkInstance,
#[cfg(gpu="vulkan")]
    pub(crate) vk_physical_device: sys::VkPhysicalDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_device: sys::VkDevice,
#[cfg(gpu="vulkan")]
    pub(crate) vk_present_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_graphics_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_transfer_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_compute_queue: sys::VkQueue,
#[cfg(gpu="vulkan")]
    pub(crate) vk_command_pool: sys::VkCommandPool,
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
        println!("unable to connect to X server");
        return None;
    }
    let xcb_connection = unsafe { sys::XGetXCBConnection(xdisplay) };
    if xcb_connection == null_mut() {
        println!("unable to connect to X server");
        unsafe { sys::XCloseDisplay(xdisplay) };
        return None;
    }

    // we want to use XCB (but might need X11)
    unsafe { sys::XSetEventQueueOwner(xdisplay,sys::XCBOwnsEventQueue) };

    // create epoll descriptor to be able to wait for UI events on a system level
    let fd = unsafe { sys::xcb_get_file_descriptor(xcb_connection) };
    let epfd = unsafe { sys::epoll_create1(0) };
    let mut epe = [sys::epoll_event { events: sys::EPOLLIN as u32,data: sys::epoll_data_t { u64_: 0, }, }];
    unsafe { sys::epoll_ctl(epfd,sys::EPOLL_CTL_ADD as i32,fd,epe.as_mut_ptr()) };

    // assume the first screen is the one we want
    let xcb_setup = unsafe { sys::xcb_get_setup(xcb_connection) };
    if xcb_setup == null_mut() {
        println!("unable to obtain X server setup");
        // TODO: unwind
        unsafe { sys::XCloseDisplay(xdisplay) };
        return None;
    }
    let xcb_screen = unsafe { sys::xcb_setup_roots_iterator(xcb_setup) }.data;

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

    // GPU-specific stuff
#[cfg(gpu="vulkan")]
    {
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
        let mut vk_instance: sys::VkInstance = null_mut();
        let code = unsafe { sys::vkCreateInstance(&info,null_mut(),&mut vk_instance as *mut sys::VkInstance) };
        if code != sys::VK_SUCCESS {
            println!("unable to create VkInstance ({})",code);
            unsafe { sys::XCloseDisplay(xdisplay) };
            return None;
        }

        // enumerate physical devices
        let mut count = 0u32;
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&mut count,0 as *mut sys::VkPhysicalDevice) };
        if count == 0 {
            println!("unable to enumerate physical devices");
            unsafe {
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }
        let mut vk_physical_devices = vec![null_mut() as sys::VkPhysicalDevice; count as usize];
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&mut count,vk_physical_devices.as_mut_ptr()) };

        // assume the first device is the one we want
        let vk_physical_device = vk_physical_devices[0];

        // DEBUG: show the name in debug build
#[cfg(build="debug")]
        {
            let mut properties = MaybeUninit::<sys::VkPhysicalDeviceProperties>::uninit();
            unsafe { sys::vkGetPhysicalDeviceProperties(vk_physical_device,properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            let slice: &[u8] = unsafe { &*(&properties.deviceName as *const [i8] as *const [u8]) };
            let name = std::str::from_utf8(slice).unwrap();
            dprintln!("physical device: {}",name);
        }
        
        // get supported queue families
        let mut count = 0u32;
        unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device,&mut count,null_mut()) };
        if count == 0 {
            println!("no queue families supported on this GPU");
            unsafe {
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }

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
        unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device,&mut count,vk_queue_families.as_mut_ptr()) };

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
            unsafe {
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }
        if vk_queue_family.queueCount < 4 {
            println!("unable to allocate at least 4 queues on this GPU");
            unsafe {
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }

        // assume that presentation is done on the same family as graphics

        // create logical device with 4 queues of queue family 0
        let mut queue_create_infos = Vec::<sys::VkDeviceQueueCreateInfo>::new();
        let priority = 1f32;
        queue_create_infos.push(sys::VkDeviceQueueCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueFamilyIndex: 0,
            queueCount: 4,
            pQueuePriorities: &priority as *const f32,
        });
        let extension_names = [
            sys::VK_KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr(),
        ];
        let create_info = sys::VkDeviceCreateInfo {
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
        if unsafe { sys::vkCreateDevice(vk_physical_device,&create_info,null_mut(),vk_device.as_mut_ptr()) } != sys::VK_SUCCESS {
            println!("unable to create VkDevice");
            unsafe {
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }
        let vk_device = unsafe { vk_device.assume_init() };

        // obtain requested queues, all from queue family 0
        let mut vk_present_queue: sys::VkQueue = null_mut();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,0,&mut vk_present_queue) };
        let mut vk_graphics_queue: sys::VkQueue = null_mut();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,1,&mut vk_graphics_queue) };
        let mut vk_transfer_queue: sys::VkQueue = null_mut();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,2,&mut vk_transfer_queue) };
        let mut vk_compute_queue: sys::VkQueue = null_mut();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,3,&mut vk_compute_queue) };

        // create command pool for queue family 0
        let create_info = sys::VkCommandPoolCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueFamilyIndex: 0,
        };
        let mut vk_command_pool = MaybeUninit::uninit();
        if unsafe { sys::vkCreateCommandPool(vk_device,&create_info,null_mut(),vk_command_pool.as_mut_ptr()) } != sys::VK_SUCCESS {
            println!("unable to create command pool");
            unsafe {
                sys::vkDestroyDevice(vk_device,null_mut());
                sys::vkDestroyInstance(vk_instance,null_mut());
                sys::XCloseDisplay(xdisplay);
            }
            return None;
        }
        let vk_command_pool = unsafe { vk_command_pool.assume_init() };

        Some(System {
            xdisplay: xdisplay,
            xcb_connection: xcb_connection,
            xcb_screen: xcb_screen,
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
            vk_instance: vk_instance,
            vk_physical_device: vk_physical_device,
            vk_device: vk_device,
            vk_present_queue: vk_present_queue,
            vk_graphics_queue: vk_graphics_queue,
            vk_transfer_queue: vk_transfer_queue,
            vk_compute_queue: vk_compute_queue,
            vk_command_pool: vk_command_pool,
        })
    }

#[cfg(not(gpu="vulkan"))]
    None
}

impl<'system> System {

#[cfg(gpu="vulkan")]
    pub(crate) fn create_swapchain(&self,vk_surface: sys::VkSurfaceKHR,vk_renderpass: sys::VkRenderPass,r: Rect<i32>) -> Option<(sys::VkExtent2D,sys::VkSwapchainKHR,Vec<sys::VkImageView>,Vec<sys::VkFramebuffer>)> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::uninit();
        unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.vk_physical_device,vk_surface,capabilities.as_mut_ptr()) };
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
        dprintln!("extent = {},{}",vk_extent.width,vk_extent.height);    
        let mut image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount != 0) && (image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }
        dprintln!("image_count = {}",image_count);

        // make sure VK_FORMAT_B8G8R8A8_SRGB is supported (BGRA8UN)
        let mut count = 0u32;
        unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(self.vk_physical_device,vk_surface,&mut count,null_mut()) };
        if count == 0 {
            println!("no formats supported");
            return None;
        }
        let mut formats = vec![sys::VkSurfaceFormatKHR {
            format: 0,
            colorSpace: 0,
        }; count as usize];
        unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(self.vk_physical_device,vk_surface,&mut count,formats.as_mut_ptr()) };
        let mut format_supported = false;
        for i in 0..formats.len() {
            if (formats[i].format == sys::VK_FORMAT_B8G8R8A8_SRGB) && 
                (formats[i].colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format_supported = true;
            }
        }
        if !format_supported {
            println!("window does not support ARGB8UN");
            return None;
        }
        dprintln!("format = {}",sys::VK_FORMAT_B8G8R8A8_SRGB);

        // select VK_PRESENT_MODE_MAILBOX_KHR, or otherwise VK_PRESENT_MODE_FIFO_KHR
        let mut count = 0u32;
        unsafe { sys::vkGetPhysicalDeviceSurfacePresentModesKHR(self.vk_physical_device,vk_surface,&mut count,null_mut()) };
        if count == 0 {
            println!("unable to select present mode");
            return None;
        }
                
        // create swap chain for this window
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
            presentMode: sys::VK_PRESENT_MODE_IMMEDIATE_KHR,
            clipped: sys::VK_TRUE,
            oldSwapchain: null_mut(),
        };
        let mut vk_swapchain = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSwapchainKHR(self.vk_device,&info,null_mut(),vk_swapchain.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create swap chain (error {})",code);
                return None;
            },
        }
        let vk_swapchain = unsafe { vk_swapchain.assume_init() };

        // get swapchain images
        let mut count = 0u32;
        match unsafe { sys::vkGetSwapchainImagesKHR(self.vk_device,vk_swapchain,&mut count,null_mut()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain image count (error {})",code);
                // TODO: unwind
                return None;
            }
        }
        let mut vk_images = vec![null_mut() as sys::VkImage; count as usize];
        match unsafe { sys::vkGetSwapchainImagesKHR(self.vk_device,vk_swapchain,&mut count,vk_images.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain images (error {})",code);
                // TODO: unwind
                return None;
            },
        }

        // create image views for the swapchain images
        let mut vk_imageviews = Vec::<sys::VkImageView>::new();
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
            let mut vk_imageview = MaybeUninit::uninit();
            match unsafe { sys::vkCreateImageView(self.vk_device,&info,null_mut(),vk_imageview.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create image view (error {})",code);
                    // TODO: unwind
                    return None;
                }
            }
            vk_imageviews.push(unsafe { vk_imageview.assume_init() });
        }

        // create framebuffers for the image views
        let mut vk_framebuffers = Vec::<sys::VkFramebuffer>::new();
        for vk_imageview in &vk_imageviews {

            let info = sys::VkFramebufferCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                renderPass: vk_renderpass,
                attachmentCount: 1,
                pAttachments: &*vk_imageview,
                width: vk_extent.width,
                height: vk_extent.height,
                layers: 1,
            };
            let mut vk_framebuffer = MaybeUninit::uninit();
            match unsafe { sys::vkCreateFramebuffer(self.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create framebuffer (error {})",code);
                    // TODO: unwind
                    return None;
                }
            }
            vk_framebuffers.push(unsafe { vk_framebuffer.assume_init() });
        }

        Some((vk_extent,vk_swapchain,vk_imageviews,vk_framebuffers))
    }

    // create basic window, decorations are handled in the public create_frame and create_popup
    fn create_window(&self,r: Rect<i32>,_absolute: bool) -> Option<Window> {

        // create window
        let xcb_window = unsafe { sys::xcb_generate_id(self.xcb_connection) };
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
        {
            // create surface for this window
            let info = sys::VkXcbSurfaceCreateInfoKHR {
                sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                pNext: null_mut(),
                flags: 0,
                connection: self.xcb_connection as *mut sys::xcb_connection_t,
                window: xcb_window,
            };
            let mut vk_surface = MaybeUninit::uninit();
            match unsafe { sys::vkCreateXcbSurfaceKHR(self.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("Unable to create Vulkan XCB surface (error {})",code);
                    unsafe {
                        sys::xcb_unmap_window(self.xcb_connection,xcb_window as u32);
                        sys::xcb_destroy_window(self.xcb_connection,xcb_window as u32);    
                    }
                    return None;
                },
            }
            let vk_surface = unsafe { vk_surface.assume_init() };

            // create renderpass
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
            let mut vk_renderpass = MaybeUninit::uninit();
            match unsafe { sys::vkCreateRenderPass(self.vk_device,&info,null_mut(),vk_renderpass.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create render pass (error {})",code);
                    // TODO: unwind
                    return None;
                }
            }
            let vk_renderpass = unsafe { vk_renderpass.assume_init() };

            // create swapchain for this window and render pass
            if let Some((vk_extent,vk_swapchain,vk_imageviews,vk_framebuffers)) = self.create_swapchain(vk_surface,vk_renderpass,r) {

                Some(Window {
                    system: &self,
                    r: r,
                    xcb_window: xcb_window,
                    vk_surface: vk_surface,
                    vk_renderpass: vk_renderpass,
                    vk_extent: vk_extent,
                    vk_swapchain: vk_swapchain,
                    vk_imageviews: vk_imageviews,
                    vk_framebuffers: vk_framebuffers,
                })
            }
            else {
                // TODO: unwind
                None
            }
        }

#[cfg(not(gpu="vulkan"))]
        None
    }

    /// Create application frame window.
    pub fn create_frame_window(&self,r: Rect<i32>,title: &str) -> Option<Window> {
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
    pub fn create_popup_window(&self,r: Rect<i32>) -> Option<Window> {
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

    /// Create a command buffer.
    pub fn create_commandbuffer(&self) -> Option<CommandBuffer> {
#[cfg(gpu="vulkan")]
        {
            let info = sys::VkCommandBufferAllocateInfo {
                sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                pNext: null_mut(),
                commandPool: self.vk_command_pool,
                level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount: 1,
            };
            let mut vk_command_buffer = MaybeUninit::uninit();
            if unsafe { sys::vkAllocateCommandBuffers(self.vk_device,&info,vk_command_buffer.as_mut_ptr()) } != sys::VK_SUCCESS {
                return None;
            }
            Some(CommandBuffer {
                system: &self,
                vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            })
        }

#[cfg(not(gpu="vulkan"))]
        None
    }

    /// Create a semaphore.
    pub fn create_semaphore(&self) -> Option<Semaphore> {

        let info = sys::VkSemaphoreCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
        };
        let mut vk_semaphore = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSemaphore(self.vk_device,&info,null_mut(),vk_semaphore.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create semaphore (error {})",code);
                return None;
            },
        }
        Some(Semaphore {
            system: &self,
            vk_semaphore: unsafe { vk_semaphore.assume_init() },
        })
    }

    /// Create a pipeline layout.
    pub fn create_pipeline_layout(&self) -> Option<PipelineLayout> {

#[cfg(gpu="vulkan")]
        {
            let info = sys::VkPipelineLayoutCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                setLayoutCount: 0,
                pSetLayouts: null_mut(),
                pushConstantRangeCount: 0,
                pPushConstantRanges: null_mut(),
            };
            let mut vk_pipeline_layout = MaybeUninit::uninit();
            match unsafe { sys::vkCreatePipelineLayout(self.vk_device,&info,null_mut(),vk_pipeline_layout.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create pipeline layout (error {})",code);
                    return None;
                },
            }
            Some(PipelineLayout {
                system: &self,
                vk_pipeline_layout: unsafe { vk_pipeline_layout.assume_init() },
            })
        }
    }

    /// Create a shader.
    pub fn create_shader(&self,code: &[u8]) -> Option<Shader> {

#[cfg(gpu="vulkan")]
        {
            let create_info = sys::VkShaderModuleCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                codeSize: code.len() as u64,
                pCode: code.as_ptr() as *const u32,
            };

            let mut vk_shader_module = MaybeUninit::uninit();
            match unsafe { sys::vkCreateShaderModule(self.vk_device,&create_info,null_mut(),vk_shader_module.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create Vulkan shader module (error {})",code);
                    return None;
                },
            }
            let vk_shader_module = unsafe { vk_shader_module.assume_init() };

            Some(Shader {
                system: &self,
                vk_shader_module: vk_shader_module,
            })
        }

#[cfg(not(gpu="vulkan"))]
        None
    }

    /// Create a graphics pipeline.
    pub fn create_graphics_pipeline(&self,pipeline_layout: &PipelineLayout,window: &Window,vertex_shader: &Shader,fragment_shader: &Shader) -> Option<GraphicsPipeline> {

#[cfg(gpu="vulkan")]
        {
            let create_info = sys::VkGraphicsPipelineCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                stageCount: 2,
                pStages: [
                    sys::VkPipelineShaderStageCreateInfo {
                        sType: sys::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        pNext: null_mut(),
                        flags: 0,
                        stage: sys::VK_SHADER_STAGE_VERTEX_BIT,
                        module: vertex_shader.vk_shader_module,
                        pName: b"main\0".as_ptr() as *const i8,
                        pSpecializationInfo: null_mut(),
                    },
                    sys::VkPipelineShaderStageCreateInfo {
                        sType: sys::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                        pNext: null_mut(),
                        flags: 0,
                        stage: sys::VK_SHADER_STAGE_FRAGMENT_BIT,
                        module: fragment_shader.vk_shader_module,
                        pName: b"main\0".as_ptr() as *const i8,
                        pSpecializationInfo: null_mut(),
                    }
                ].as_ptr(),
                pVertexInputState: &sys::VkPipelineVertexInputStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    vertexBindingDescriptionCount: 0,
                    pVertexBindingDescriptions: null_mut(),
                    vertexAttributeDescriptionCount: 0,
                    pVertexAttributeDescriptions: null_mut(),
                },
                pInputAssemblyState: &sys::VkPipelineInputAssemblyStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    topology: sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    primitiveRestartEnable: sys::VK_FALSE,
                },
                pTessellationState: &sys::VkPipelineTessellationStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    patchControlPoints: 1,
                },
                pViewportState: &sys::VkPipelineViewportStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    viewportCount: 1,
                    pViewports: &sys::VkViewport {
                        x: 0.0,
                        y: 0.0,
                        width: window.vk_extent.width as f32,
                        height: window.vk_extent.height as f32,
                        minDepth: 0.0,
                        maxDepth: 1.0,
                    },
                    scissorCount: 1,
                    pScissors: &sys::VkRect2D {
                        offset: sys::VkOffset2D {
                            x: 0,
                            y: 0,
                        },
                        extent: window.vk_extent,
                    },
                },
                pRasterizationState: &sys::VkPipelineRasterizationStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    depthClampEnable: sys::VK_FALSE,
                    rasterizerDiscardEnable: sys::VK_FALSE,
                    polygonMode: sys::VK_POLYGON_MODE_FILL,
                    cullMode: sys::VK_CULL_MODE_BACK_BIT,
                    frontFace: sys::VK_FRONT_FACE_CLOCKWISE,
                    depthBiasEnable: sys::VK_FALSE,
                    depthBiasConstantFactor: 0.0,
                    depthBiasClamp: 0.0,
                    depthBiasSlopeFactor: 0.0,
                    lineWidth: 1.0,
                },
                pMultisampleState: &sys::VkPipelineMultisampleStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    rasterizationSamples: sys::VK_SAMPLE_COUNT_1_BIT,
                    sampleShadingEnable: sys::VK_FALSE,
                    minSampleShading: 1.0,
                    pSampleMask: null_mut(),
                    alphaToCoverageEnable: sys::VK_FALSE,
                    alphaToOneEnable: sys::VK_FALSE,
                },
                pDepthStencilState: null_mut(),
                pColorBlendState: &sys::VkPipelineColorBlendStateCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    logicOpEnable: sys::VK_FALSE,
                    logicOp: sys::VK_LOGIC_OP_COPY,
                    attachmentCount: 1,
                    pAttachments: &sys::VkPipelineColorBlendAttachmentState {
                        blendEnable: sys::VK_FALSE,
                        srcColorBlendFactor: sys::VK_BLEND_FACTOR_ONE,
                        dstColorBlendFactor: sys::VK_BLEND_FACTOR_ZERO,
                        colorBlendOp: sys::VK_BLEND_OP_ADD,
                        srcAlphaBlendFactor: sys::VK_BLEND_FACTOR_ONE,
                        dstAlphaBlendFactor: sys::VK_BLEND_FACTOR_ZERO,
                        alphaBlendOp: sys::VK_BLEND_OP_ADD,
                        colorWriteMask: sys::VK_COLOR_COMPONENT_R_BIT |
                            sys::VK_COLOR_COMPONENT_G_BIT |
                            sys::VK_COLOR_COMPONENT_B_BIT |
                            sys::VK_COLOR_COMPONENT_A_BIT,
                    },
                    blendConstants: [0.0,0.0,0.0,0.0],
                },
                pDynamicState: null_mut(),
                layout: pipeline_layout.vk_pipeline_layout,
                renderPass: window.vk_renderpass,
                subpass: 0,
                basePipelineHandle: null_mut(),
                basePipelineIndex: -1,
            };
            let mut vk_graphics_pipeline = MaybeUninit::uninit();
            match unsafe { sys::vkCreateGraphicsPipelines(self.vk_device,null_mut(),1,&create_info,null_mut(),vk_graphics_pipeline.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create Vulkan graphics pipeline (error {})",code);
                    return None;
                },
            }

            Some(GraphicsPipeline {
                system: &self,
                vk_graphics_pipeline: unsafe { vk_graphics_pipeline.assume_init() },
            })
        }

#[cfg(not(gpu="vulkan"))]
        None
    }

#[doc(hidden)]
    fn submit(&self,vk_queue: sys::VkQueue,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        let wait_stage = sys::VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        let info = sys::VkSubmitInfo {
            sType: sys::VK_STRUCTURE_TYPE_SUBMIT_INFO,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            pWaitDstStageMask: &wait_stage,
            commandBufferCount: 1,
            pCommandBuffers: &command_buffer.vk_command_buffer,
            signalSemaphoreCount: 1,
            pSignalSemaphores: &signal_semaphore.vk_semaphore,
        };
        match unsafe { sys::vkQueueSubmit(vk_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to submit to graphics queue (error {})",code);
                false
            },
        }
    }

    /// Submit command buffer to graphics queue.
    pub fn submit_graphics(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        self.submit(self.vk_graphics_queue,command_buffer,wait_semaphore,signal_semaphore)
    }

    // Submit command buffer to transfer queue.
    pub fn submit_transfer(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        self.submit(self.vk_transfer_queue,command_buffer,wait_semaphore,signal_semaphore)
    }

    // Submit command buffer to compute queue.
    pub fn submit_compute(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
        self.submit(self.vk_compute_queue,command_buffer,wait_semaphore,signal_semaphore)
    }

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
        let mut epe = [sys::epoll_event { events: sys::EPOLLIN as u32,data: sys::epoll_data_t { u64_: 0, } }];
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
                sys::vkDestroyDevice(self.vk_device,null_mut());
                sys::vkDestroyInstance(self.vk_instance,null_mut());
            }
            sys::XCloseDisplay(self.xdisplay);
        }
    }
}