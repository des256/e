use {
    crate::*,
    std::{
        rc::Rc,
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

// Supplemental fields for System
pub(crate) struct SystemGpu {
    pub xcb_depth: u8,
    pub xcb_visual_id: u32,
    pub vk_instance: sys::VkInstance,
    pub vk_physical_device: sys::VkPhysicalDevice,
    pub vk_device: sys::VkDevice,
    pub vk_queue: sys::VkQueue,
    pub vk_command_pool: sys::VkCommandPool,
    pub shared_index: usize,
}

pub(crate) fn open_system_gpu(xcb_screen: *mut sys::xcb_screen_t) -> Option<SystemGpu> {

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
    let vk_queue_families = unsafe { std::mem::transmute::<_,Vec<sys::VkQueueFamilyProperties>>(vk_queue_families) };

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
            sys::vkDestroyDevice(vk_device,null_mut());
            sys::vkDestroyInstance(vk_instance,null_mut());
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

    Some(SystemGpu {
        xcb_depth: unsafe { *xcb_screen }.root_depth,
        xcb_visual_id: unsafe { *xcb_screen }.root_visual,
        vk_instance,
        vk_physical_device,
        vk_device,
        vk_queue,
        vk_command_pool,
        shared_index,
    })
}

impl System {
    
    /// Create swapchain resources for surface, render pass and rectangle.
    pub(crate) fn create_swapchain_resources(&self,vk_surface: sys::VkSurfaceKHR,vk_render_pass: sys::VkRenderPass,r: Rect<i32,u32>) -> Option<SwapchainResources> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::uninit();
        unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.gpu.vk_physical_device,
            vk_surface,
            capabilities.as_mut_ptr(),
        ) };
        let capabilities = unsafe { capabilities.assume_init() };

        let extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
            dprintln!("fixed extent = {} x {}",capabilities.currentExtent.width,capabilities.currentExtent.height);
            u32xy {
                x: capabilities.currentExtent.width,
                y: capabilities.currentExtent.height,
            }
        }
        else {
            let mut extent = u32xy { x: r.s.x as u32,y: r.s.y as u32, };
            if extent.x < capabilities.minImageExtent.width {
                extent.x = capabilities.minImageExtent.width;
            }
            if extent.y < capabilities.minImageExtent.height {
                extent.y = capabilities.minImageExtent.height;
            }
            if extent.x > capabilities.maxImageExtent.width {
                extent.x = capabilities.maxImageExtent.width;
            }
            if extent.y > capabilities.maxImageExtent.height {
                extent.y = capabilities.maxImageExtent.height;
            }
            dprintln!("specified extent = {}",extent);
            extent
        };

        // make sure VK_FORMAT_B8G8R8A8_SRGB is supported (BGRA8UN)
        let mut count = 0u32;
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.gpu.vk_physical_device,
            vk_surface,
            &mut count,
            null_mut(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get surface formats (error {})",code);
                return None;
            }
        }
        let mut formats = vec![MaybeUninit::<sys::VkSurfaceFormatKHR>::uninit(); count as usize];
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.gpu.vk_physical_device,
            vk_surface,
            &mut count,
            formats.as_mut_ptr() as *mut sys::VkSurfaceFormatKHR,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get surface formats (error {})",code);
                return None;
            }
        }
        let formats = unsafe { std::mem::transmute::<_,Vec<sys::VkSurfaceFormatKHR>>(formats) };

        let mut format_supported = false;
        for i in 0..formats.len() {
            if (formats[i].format == sys::VK_FORMAT_B8G8R8A8_SRGB) && 
                (formats[i].colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                format_supported = true;
            }
        }
        if !format_supported {
            println!("window does not support BGRA8UN");
            return None;
        }
        dprintln!("format = {}",sys::VK_FORMAT_B8G8R8A8_SRGB);

        // create swap chain for this window
        dprintln!("creating swap chain...");
        let info = sys::VkSwapchainCreateInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: null_mut(),
            flags: 0,
            surface: vk_surface,
            minImageCount: 2,
            imageFormat: sys::VK_FORMAT_B8G8R8A8_SRGB,
            imageColorSpace: sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            imageExtent: sys::VkExtent2D { width: extent.x,height: extent.y, },
            imageArrayLayers: 1,
            imageUsage: sys::VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            imageSharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
            preTransform: capabilities.currentTransform,
            compositeAlpha: sys::VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            presentMode: sys::VK_PRESENT_MODE_FIFO_KHR,
            clipped: sys::VK_TRUE,
            oldSwapchain: null_mut(),
        };
        let mut vk_swapchain = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSwapchainKHR(
            self.gpu.vk_device,
            &info,
            null_mut(),
            vk_swapchain.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create swap chain (error {})",code);
                return None;
            },
        }
        let vk_swapchain = unsafe { vk_swapchain.assume_init() };

        // get swapchain images
        dprintln!("getting swap chain images...");
        let mut count = 0u32;
        match unsafe { sys::vkGetSwapchainImagesKHR(self.gpu.vk_device,vk_swapchain,&mut count,null_mut()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain image count (error {})",code);
                unsafe { sys::vkDestroySwapchainKHR(self.gpu.vk_device,vk_swapchain,null_mut()) };
                return None;
            }
        }
        let mut vk_images = vec![MaybeUninit::<sys::VkImage>::uninit(); count as usize];
        match unsafe { sys::vkGetSwapchainImagesKHR(
            self.gpu.vk_device,
            vk_swapchain,
            &count as *const u32 as *mut u32,
            vk_images.as_mut_ptr() as *mut sys::VkImage,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to get swap chain images (error {})",code);
                unsafe { sys::vkDestroySwapchainKHR(self.gpu.vk_device,vk_swapchain,null_mut()) };
                return None;
            },
        }
        let vk_images = unsafe { std::mem::transmute::<_,Vec<sys::VkImage>>(vk_images) };

        // create image views for the swapchain images
        dprintln!("creating image views onto swap chain images...");
        let mut vk_image_views = Vec::<sys::VkImageView>::new();
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
            let mut vk_image_view = MaybeUninit::uninit();
            match unsafe { sys::vkCreateImageView(self.gpu.vk_device,&info,null_mut(),vk_image_view.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create image view (error {})",code);
                    unsafe {
                        for vk_image_view in &vk_image_views {
                            sys::vkDestroyImageView(self.gpu.vk_device,*vk_image_view,null_mut());
                        }
                        sys::vkDestroySwapchainKHR(self.gpu.vk_device,vk_swapchain,null_mut());
                    }
                    return None;
                }
            }
            vk_image_views.push(unsafe { vk_image_view.assume_init() });
        }

        // create framebuffers for the image views
        dprintln!("creating frame buffers for the image views...");
        let mut vk_framebuffers = Vec::<sys::VkFramebuffer>::new();
        for vk_image_view in &vk_image_views {
            let info = sys::VkFramebufferCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                renderPass: vk_render_pass,
                attachmentCount: 1,
                pAttachments: vk_image_view,
                width: extent.x,
                height: extent.y,
                layers: 1,
            };
            let mut vk_framebuffer = MaybeUninit::uninit();
            match unsafe { sys::vkCreateFramebuffer(self.gpu.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    println!("unable to create framebuffer (error {})",code);
                    unsafe {
                        for vk_framebuffer in &vk_framebuffers {
                            sys::vkDestroyFramebuffer(self.gpu.vk_device,*vk_framebuffer,null_mut());
                        }
                        for vk_image_view in &vk_image_views {
                            sys::vkDestroyImageView(self.gpu.vk_device,*vk_image_view,null_mut());
                        }
                        sys::vkDestroySwapchainKHR(self.gpu.vk_device,vk_swapchain,null_mut());
                    }
                    return None;
                }
            }
            vk_framebuffers.push(unsafe { vk_framebuffer.assume_init() });
        }

        dprintln!("success.");

        Some(SwapchainResources {
            vk_swapchain,
            vk_image_views,
            vk_framebuffers,
        })
    }

    /// Create GPU-specific Window part for surface and rectangle.
    pub(crate) fn create_window_gpu(&self,vk_surface: sys::VkSurfaceKHR,r: Rect<isize,usize>) -> Option<WindowGpu> {

        // check if surface is supported by queue family
        let mut supported = sys::VK_FALSE;
        unsafe { sys::vkGetPhysicalDeviceSurfaceSupportKHR(self.gpu.vk_physical_device,0,vk_surface,&mut supported) };
        if supported != sys::VK_TRUE {
            println!("surface not supported by queue family");
            unsafe { sys::vkDestroySurfaceKHR(self.gpu.vk_instance,vk_surface,null_mut()) };
            return None;
        }

        // create render pass

        // attachments: image descriptions and load/store indications of each attachment
        // subpasses: subpass? descriptions
        // dependencies: how subpasses depend on each othe

        // window needs one, and probably also off-screen rendering contexts

        // the render pass explains what the draw commands operate on

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
        let mut vk_render_pass = MaybeUninit::uninit();
        match unsafe { sys::vkCreateRenderPass(self.gpu.vk_device,&info,null_mut(),vk_render_pass.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create render pass (error {})",code);
                unsafe { sys::vkDestroySurfaceKHR(self.gpu.vk_instance,vk_surface,null_mut()) };
                return None;
            }
        }
        let vk_render_pass = unsafe { vk_render_pass.assume_init() };

        // create swapchain resources
        if let Some(swapchain_resources) = self.create_swapchain_resources(vk_surface,vk_render_pass,Rect { o: Vec2 { x: r.o.x as i32,y: r.o.y as i32, },s: Vec2 { x: r.s.x as u32,y: r.s.y as u32, } }) {
            Some(WindowGpu {
                vk_surface,
                vk_render_pass,
                swapchain_resources,
            })
        }
        else {
            unsafe {
                sys::vkDestroyRenderPass(self.gpu.vk_device,vk_render_pass,null_mut());
                sys::vkDestroySurfaceKHR(self.gpu.vk_instance,vk_surface,null_mut());
            }
            return None;
        }
    }

    /// Create command buffer.
    pub fn create_command_buffer(self: &Rc<System>) -> Option<CommandBuffer> {

        let info = sys::VkCommandBufferAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            pNext: null_mut(),
            commandPool: self.gpu.vk_command_pool,
            level: sys::VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount: 1,
        };
        let mut vk_command_buffer = MaybeUninit::uninit();
        if unsafe { sys::vkAllocateCommandBuffers(self.gpu.vk_device,&info,vk_command_buffer.as_mut_ptr()) } != sys::VK_SUCCESS {
            return None;
        }
        Some(CommandBuffer {
            system: Rc::clone(self),
            vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
        })
    }

    /// Wait for wait_semaphore before submitting command buffer to the queue, and signal signal_semaphore when ready.
    pub fn submit(&mut self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
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
        match unsafe { sys::vkQueueSubmit(self.gpu.vk_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("unable to submit to graphics queue (error {})",code);
                false
            },
        }
    }

    /// Drop all GPU-specific resources.
    pub fn drop_gpu(&self) {
        unsafe {
            sys::vkDestroyCommandPool(self.gpu.vk_device,self.gpu.vk_command_pool,null_mut());
            sys::vkDestroyDevice(self.gpu.vk_device,null_mut());
            sys::vkDestroyInstance(self.gpu.vk_instance,null_mut());
        }
    }
}
