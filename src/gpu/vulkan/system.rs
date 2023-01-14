use {
    crate::*,
    std::{
        rc::Rc,
        ptr::{
            null_mut,
            copy_nonoverlapping,
        },
        mem::MaybeUninit,
        ffi::c_void,
        cell::Cell,
    },
};

trait BaseTypeFormat {
    const FORMAT: sys::VkFormat;
}

impl BaseTypeFormat for u8 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8_UINT; }
impl BaseTypeFormat for u16 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16_UINT; }
impl BaseTypeFormat for u32 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32_UINT; }
impl BaseTypeFormat for u64 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64_UINT; }
impl BaseTypeFormat for i8 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8_SINT; }
impl BaseTypeFormat for i16 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16_SINT; }
impl BaseTypeFormat for i32 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32_SINT; }
impl BaseTypeFormat for i64 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64_SINT; }
impl BaseTypeFormat for f32 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32_SFLOAT; }
impl BaseTypeFormat for f64 { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64_SFLOAT; }
impl BaseTypeFormat for Vec2<u8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8_UINT; }
impl BaseTypeFormat for Vec2<u16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16_UINT; }
impl BaseTypeFormat for Vec2<u32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32_UINT; }
impl BaseTypeFormat for Vec2<u64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64_UINT; }
impl BaseTypeFormat for Vec2<i8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8_SINT; }
impl BaseTypeFormat for Vec2<i16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16_SINT; }
impl BaseTypeFormat for Vec2<i32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32_SINT; }
impl BaseTypeFormat for Vec2<i64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64_SINT; }
impl BaseTypeFormat for Vec2<f32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32_SFLOAT; }
impl BaseTypeFormat for Vec2<f64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64_SFLOAT; }
impl BaseTypeFormat for Vec3<u8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8B8_UINT; }
impl BaseTypeFormat for Vec3<u16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16B16_UINT; }
impl BaseTypeFormat for Vec3<u32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32_UINT; }
impl BaseTypeFormat for Vec3<u64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64_UINT; }
impl BaseTypeFormat for Vec3<i8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8B8_SINT; }
impl BaseTypeFormat for Vec3<i16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16B16_SINT; }
impl BaseTypeFormat for Vec3<i32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32_SINT; }
impl BaseTypeFormat for Vec3<i64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64_SINT; }
impl BaseTypeFormat for Vec3<f32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32_SFLOAT; }
impl BaseTypeFormat for Vec3<f64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64_SFLOAT; }
impl BaseTypeFormat for Vec4<u8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8B8A8_UINT; }
impl BaseTypeFormat for Vec4<u16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16B16A16_UINT; }
impl BaseTypeFormat for Vec4<u32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32A32_UINT; }
impl BaseTypeFormat for Vec4<u64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64A64_UINT; }
impl BaseTypeFormat for Vec4<i8> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R8G8B8A8_SINT; }
impl BaseTypeFormat for Vec4<i16> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R16G16B16A16_SINT; }
impl BaseTypeFormat for Vec4<i32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32A32_SINT; }
impl BaseTypeFormat for Vec4<i64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64A64_SINT; }
impl BaseTypeFormat for Vec4<f32> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R32G32B32A32_SFLOAT; }
impl BaseTypeFormat for Vec4<f64> { const FORMAT: sys::VkFormat = sys::VK_FORMAT_R64G64B64A64_SFLOAT; }

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
    pub(crate) fn create_swapchain_resources(&self,vk_surface: sys::VkSurfaceKHR,vk_render_pass: sys::VkRenderPass,r: Rect<isize,usize>) -> Option<SwapchainResources> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::uninit();
        unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            self.gpu.vk_physical_device,
            vk_surface,
            capabilities.as_mut_ptr(),
        ) };
        let capabilities = unsafe { capabilities.assume_init() };

        let extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
            Vec2 {
                x: capabilities.currentExtent.width,
                y: capabilities.currentExtent.height,
            }
        }
        else {
            let mut extent = Vec2 { x: r.s.x as u32,y: r.s.y as u32, };
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

        // create swap chain for this window
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
        if let Some(swapchain_resources) = self.create_swapchain_resources(vk_surface,vk_render_pass,r) {
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

    /*
    /// Create a graphics pipeline.
    pub fn create_graphics_pipeline<T: Vertex>(
        self: &Rc<Self>,
        window: &Window,
        pipeline_layout: &Rc<PipelineLayout>,
        vertex_shader: &Rc<VertexShader>,
        fragment_shader: &Rc<FragmentShader>,
        topology: PrimitiveTopology,
        restart: PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: DepthClamp,
        primitive_discard: PrimitiveDiscard,
        polygon_mode: PolygonMode,
        cull_mode: CullMode,
        depth_bias: DepthBias,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: SampleShading,
        alpha_to_coverage: AlphaToCoverage,
        alpha_to_one: AlphaToOne,
        depth_test: DepthTest,
        depth_write_mask: bool,
        stencil_test: StencilTest,
        logic_op: LogicOp,
        blend: Blend,
        write_mask: (bool,bool,bool,bool),
        blend_constant: Color<f32>,
    ) -> Option<Rc<GraphicsPipeline>> {

        let vertex_base_fields = T::get_fields();

        let shaders = [
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
        ];

        let mut location = 0u32;
        let mut stride = 0u32;
        let mut attribute_descriptions: Vec<sys::VkVertexInputAttributeDescription> = Vec::new();
        for (_,ty) in &vertex_base_fields {
            attribute_descriptions.push(sys::VkVertexInputAttributeDescription {
                location,
                binding: 0,
                format: base_type_format(ty),
                offset: stride,
            });
            location += 1;
            stride += ty.size() as u32;
        }

        let input = sys::VkPipelineVertexInputStateCreateInfo {
            // TODO: build entirely from T
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: [  // binding 0 is a T::SIZE for each vertex
                sys::VkVertexInputBindingDescription {
                    binding: 0,
                    stride,
                    inputRate: sys::VK_VERTEX_INPUT_RATE_VERTEX,  // or RATE_INSTANCE
                },
            ].as_ptr(),
            vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
            pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
        };

        let assembly = sys::VkPipelineInputAssemblyStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            topology: match topology {
                PrimitiveTopology::Points => sys::VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                PrimitiveTopology::Lines => sys::VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                PrimitiveTopology::LineStrip => sys::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
                PrimitiveTopology::Triangles => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                PrimitiveTopology::TriangleStrip => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                PrimitiveTopology::TriangleFan => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                PrimitiveTopology::LinesAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
                PrimitiveTopology::LineStripAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
                PrimitiveTopology::TrianglesAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
                PrimitiveTopology::TriangleStripAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
                PrimitiveTopology::Patches => sys::VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
            },
            primitiveRestartEnable: match restart {
                PrimitiveRestart::Disabled => sys::VK_FALSE,
                PrimitiveRestart::Enabled => sys::VK_TRUE,
            },
        };

        let tesselation = sys::VkPipelineTessellationStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            patchControlPoints: patch_control_points as u32,
        };

        let viewport = sys::VkPipelineViewportStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            viewportCount: 1,
            pViewports: null_mut(),
            scissorCount: 1,
            pScissors: null_mut(),
        };

        let depth_clamp = match depth_clamp {
            DepthClamp::Disabled => sys::VK_FALSE,
            DepthClamp::Enabled => sys::VK_TRUE,
        };
        let primitive_discard = match primitive_discard {
            PrimitiveDiscard::Disabled => sys::VK_FALSE,
            PrimitiveDiscard::Enabled => sys::VK_TRUE,
        };
        let polygon_mode = match polygon_mode {
            PolygonMode::Point => sys::VK_POLYGON_MODE_POINT,
            PolygonMode::Line => sys::VK_POLYGON_MODE_LINE,
            PolygonMode::Fill => sys::VK_POLYGON_MODE_FILL,
        };
        let (cull_mode,front_face) = match cull_mode {
            CullMode::None => (
                sys::VK_CULL_MODE_NONE,
                sys::VK_FRONT_FACE_CLOCKWISE
            ),
            CullMode::Front(front_face) => (
                sys::VK_CULL_MODE_FRONT_BIT,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
            CullMode::Back(front_face) => (
                sys::VK_CULL_MODE_BACK_BIT,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
            CullMode::FrontAndBack(front_face) => (
                sys::VK_CULL_MODE_FRONT_AND_BACK,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
        };
        let (depth_bias_enable,depth_bias_constant_factor,depth_bias_clamp,depth_bias_slope_factor) = match depth_bias {
            DepthBias::Disabled => (sys::VK_FALSE,0.0,0.0,0.0),
            DepthBias::Enabled(constant_factor,clamp,slope_factor) => (sys::VK_TRUE,constant_factor,clamp,slope_factor),
        };
        let rasterization = sys::VkPipelineRasterizationStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthClampEnable: depth_clamp,
            rasterizerDiscardEnable: primitive_discard,
            polygonMode: polygon_mode,
            cullMode: cull_mode,
            frontFace: front_face,
            depthBiasEnable: depth_bias_enable,
            depthBiasConstantFactor: depth_bias_constant_factor,
            depthBiasClamp: depth_bias_clamp,
            depthBiasSlopeFactor: depth_bias_slope_factor,
            lineWidth: line_width,
        };

        let (sample_shading,min_sample_shading) = match sample_shading {
            SampleShading::Disabled => (sys::VK_FALSE,0.0),
            SampleShading::Enabled(min_sample_shading) => (sys::VK_TRUE,min_sample_shading),
        };
        let alpha_to_coverage = match alpha_to_coverage {
            AlphaToCoverage::Disabled => sys::VK_FALSE,
            AlphaToCoverage::Enabled => sys::VK_TRUE,
        };
        let alpha_to_one = match alpha_to_one {
            AlphaToOne::Disabled => sys::VK_FALSE,
            AlphaToOne::Enabled => sys::VK_TRUE,
        };
        let multisample = sys::VkPipelineMultisampleStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            rasterizationSamples: rasterization_samples as u32,
            sampleShadingEnable: sample_shading,
            minSampleShading: min_sample_shading,
            pSampleMask: null_mut(),
            alphaToCoverageEnable: alpha_to_coverage,
            alphaToOneEnable: alpha_to_one,
        };

        let (depth_test,depth_compare,(depth_bounds,min_depth_bounds,max_depth_bounds)) = match depth_test {
            DepthTest::Disabled => (sys::VK_FALSE,sys::VK_COMPARE_OP_ALWAYS,(sys::VK_FALSE,0.0,0.0)),
            DepthTest::Enabled(depth_compare,depth_bounds) => (
                sys::VK_TRUE,
                match depth_compare {
                    CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                    CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                    CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                    CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                    CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                    CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                    CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                    CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,
                },
                match depth_bounds {
                    DepthBounds::Disabled => (sys::VK_FALSE,0.0,0.0),
                    DepthBounds::Enabled(min,max) => (sys::VK_TRUE,min,max),
                },
            ),
        };
        let depth_write = if depth_write_mask { sys::VK_TRUE } else { sys::VK_FALSE };
        let (
            stencil_test,
            (front_fail,front_pass,front_depth_fail,front_compare,front_compare_mask,front_write_mask,front_reference),
            (back_fail,back_pass,back_depth_fail,back_compare,back_compare_mask,back_write_mask,back_reference),
        ) = match stencil_test {
            StencilTest::Disabled => (
                sys::VK_FALSE,
                (sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_COMPARE_OP_ALWAYS,0,0,0),
                (sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_COMPARE_OP_ALWAYS,0,0,0),
            ),
            StencilTest::Enabled(
                (front_fail,front_pass,front_depth_fail,front_compare,front_compare_mask,front_write_mask,front_reference),
                (back_fail,back_pass,back_depth_fail,back_compare,back_compare_mask,back_write_mask,back_reference),
            ) => (
                sys::VK_TRUE,
                (
                    match front_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_pass {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_depth_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_compare {
                        CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                        CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                        CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                        CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                        CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                        CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                        CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                        CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,    
                    },
                    front_compare_mask,
                    front_write_mask,
                    front_reference,
                ),
                (
                    match back_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_pass {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_depth_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_compare {
                        CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                        CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                        CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                        CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                        CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                        CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                        CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                        CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,    
                    },
                    back_compare_mask,
                    back_write_mask,
                    back_reference,
                ),
            )
        };
        let depth_stencil = sys::VkPipelineDepthStencilStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthTestEnable: depth_test,
            depthWriteEnable: depth_write,
            depthCompareOp: depth_compare,
            depthBoundsTestEnable: depth_bounds,
            stencilTestEnable: stencil_test,
            front: sys::VkStencilOpState {
                failOp: front_fail,
                passOp: front_pass,
                depthFailOp: front_depth_fail,
                compareOp: front_compare,
                compareMask: front_compare_mask,
                writeMask: front_write_mask,
                reference: front_reference,
            },
            back: sys::VkStencilOpState {
                failOp: back_fail,
                passOp: back_pass,
                depthFailOp: back_depth_fail,
                compareOp: back_compare,
                compareMask: back_compare_mask,
                writeMask: back_write_mask,
                reference: back_reference,
            },
            minDepthBounds: min_depth_bounds,
            maxDepthBounds: max_depth_bounds,
        };

        let (logic_op_enable,logic_op) = match logic_op {
            LogicOp::Disabled => (sys::VK_FALSE,sys::VK_LOGIC_OP_COPY),
            LogicOp::Clear => (sys::VK_TRUE,sys::VK_LOGIC_OP_CLEAR),
            LogicOp::And => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND),
            LogicOp::AndReverse => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND_REVERSE),
            LogicOp::Copy => (sys::VK_TRUE,sys::VK_LOGIC_OP_COPY),
            LogicOp::AndInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND_INVERTED),
            LogicOp::NoOp => (sys::VK_TRUE,sys::VK_LOGIC_OP_NO_OP),
            LogicOp::Xor => (sys::VK_TRUE,sys::VK_LOGIC_OP_XOR),
            LogicOp::Or => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR),
            LogicOp::Nor => (sys::VK_TRUE,sys::VK_LOGIC_OP_NOR),
            LogicOp::Equivalent => (sys::VK_TRUE,sys::VK_LOGIC_OP_EQUIVALENT),
            LogicOp::Invert => (sys::VK_TRUE,sys::VK_LOGIC_OP_INVERT),
            LogicOp::OrReverse => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR_REVERSE),
            LogicOp::CopyInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_COPY_INVERTED),
            LogicOp::OrInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR_INVERTED),
            LogicOp::Nand => (sys::VK_TRUE,sys::VK_LOGIC_OP_NAND),
            LogicOp::Set => (sys::VK_TRUE,sys::VK_LOGIC_OP_SET),
        };
        let (
            blend,
            (color_op,src_color,dst_color),
            (alpha_op,src_alpha,dst_alpha),
        ) = match blend {
            Blend::Disabled => (
                sys::VK_FALSE,
                (sys::VK_BLEND_OP_ADD,sys::VK_BLEND_FACTOR_ONE,sys::VK_BLEND_FACTOR_ZERO),
                (sys::VK_BLEND_OP_ADD,sys::VK_BLEND_FACTOR_ONE,sys::VK_BLEND_FACTOR_ZERO),
            ),
            Blend::Enabled((color_op,src_color,dst_color),(alpha_op,src_alpha,dst_alpha)) => (
                sys::VK_TRUE,
                (
                    match color_op {
                        BlendOp::Add => sys::VK_BLEND_OP_ADD,
                        BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                        BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                        BlendOp::Min => sys::VK_BLEND_OP_MIN,
                        BlendOp::Max => sys::VK_BLEND_OP_MAX,
                    },
                    match src_color {
                        BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                        BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                        BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                        BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                        BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                        BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                        BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                        BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                        BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                        BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                        BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                        BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                        BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                        BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                        BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                        BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                        BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                        BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                        BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                    },
                    match dst_color {
                        BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                        BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                        BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                        BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                        BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                        BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                        BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                        BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                        BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                        BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                        BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                        BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                        BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                        BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                        BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                        BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                        BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                        BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                        BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                    },
                ),
                (
                    match alpha_op {
                        BlendOp::Add => sys::VK_BLEND_OP_ADD,
                        BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                        BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                        BlendOp::Min => sys::VK_BLEND_OP_MIN,
                        BlendOp::Max => sys::VK_BLEND_OP_MAX,
                    },
                    match src_alpha {
                        BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                        BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                        BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                        BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                        BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                        BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                        BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                        BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                        BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                        BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                        BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                        BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                        BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                        BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                        BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                        BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                        BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                        BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                        BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                    },
                    match dst_alpha {
                        BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                        BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                        BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                        BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                        BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                        BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                        BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                        BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                        BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                        BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                        BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                        BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                        BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                        BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                        BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                        BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                        BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                        BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                        BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                    },
                ),
            ),
        };
        let color_write_mask = if write_mask.0 { 8 } else { 0 } | if write_mask.1 { 4 } else { 0 } | if write_mask.2 { 2 } else { 0 } | if write_mask.3 { 1 } else { 0 };
        let blend = sys::VkPipelineColorBlendStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            logicOpEnable: logic_op_enable,
            logicOp: logic_op,
            attachmentCount: 1,
            pAttachments: &sys::VkPipelineColorBlendAttachmentState {
                blendEnable: blend,
                srcColorBlendFactor: src_color,
                dstColorBlendFactor: dst_color,
                colorBlendOp: color_op,
                srcAlphaBlendFactor: src_alpha,
                dstAlphaBlendFactor: dst_alpha,
                alphaBlendOp: alpha_op,
                colorWriteMask: color_write_mask,
            },
            blendConstants: [blend_constant.r,blend_constant.g,blend_constant.b,blend_constant.a],
        };

        let dynamic = sys::VkPipelineDynamicStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            pDynamicStates: [
                sys::VK_DYNAMIC_STATE_VIEWPORT,
                sys::VK_DYNAMIC_STATE_SCISSOR,
            ].as_ptr(),
            dynamicStateCount: 2,
        };

        let create_info = sys::VkGraphicsPipelineCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            stageCount: 2,
            pStages: shaders.as_ptr(),
            pVertexInputState: &input,
            pInputAssemblyState: &assembly,
            pTessellationState: &tesselation,
            pViewportState: &viewport,
            pRasterizationState: &rasterization,
            pMultisampleState: &multisample,
            pDepthStencilState: &depth_stencil,
            pColorBlendState: &blend,
            pDynamicState: &dynamic,
            layout: pipeline_layout.vk_pipeline_layout,
            renderPass: window.gpu.vk_render_pass,
            subpass: 0,
            basePipelineHandle: null_mut(),
            basePipelineIndex: -1,
        };

        let mut vk_graphics_pipeline = MaybeUninit::uninit();
        match unsafe { sys::vkCreateGraphicsPipelines(self.gpu.vk_device,null_mut(),1,&create_info,null_mut(),vk_graphics_pipeline.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create graphics pipeline (error {})",code);
                return None;
            },
        }

        Some(Rc::new(GraphicsPipeline {
            system: Rc::clone(self),
            vk_graphics_pipeline: unsafe { vk_graphics_pipeline.assume_init() },
            vertex_shader: Rc::clone(vertex_shader),
            fragment_shader: Rc::clone(fragment_shader),
            pipeline_layout: Rc::clone(pipeline_layout),
        }))
    }

    /// Create a pipeline layout.
    pub fn create_pipeline_layout(self: &Rc<Self>) -> Option<Rc<PipelineLayout>> {

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
        match unsafe { sys::vkCreatePipelineLayout(self.gpu.vk_device,&info,null_mut(),vk_pipeline_layout.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create pipeline layout (error {})",code);
                return None;
            },
        }
        Some(Rc::new(PipelineLayout {
            system: Rc::clone(self),
            vk_pipeline_layout: unsafe { vk_pipeline_layout.assume_init() },
        }))
    }
    */

    /// Create command buffer.
    pub fn create_command_buffer(self: &Rc<Self>) -> Option<Rc<CommandBuffer>> {

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
        Some(Rc::new(CommandBuffer {
            system: Rc::clone(self),
            vk_command_buffer: unsafe { vk_command_buffer.assume_init() },
            vertex_buffer: Cell::new(None),
            index_buffer: Cell::new(None),
            graphics_pipeline: Cell::new(None),
        }))
    }

    /// Wait for wait_semaphore before submitting command buffer to the queue, and signal signal_semaphore when ready.
    pub fn submit(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
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

    /// Create a shader.
    pub fn create_vertex_shader(self: &Rc<System>,code: &[u8]) -> Option<Rc<VertexShader>> {
        let create_info = sys::VkShaderModuleCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            codeSize: code.len() as u64,
            pCode: code.as_ptr() as *const u32,
        };

        let mut vk_shader_module = MaybeUninit::uninit();
        match unsafe { sys::vkCreateShaderModule(self.gpu.vk_device,&create_info,null_mut(),vk_shader_module.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create shader (error {})",code);
                return None;
            },
        }
        let vk_shader_module = unsafe { vk_shader_module.assume_init() };

        Some(Rc::new(VertexShader {
            system: Rc::clone(self),
            vk_shader_module: vk_shader_module,
        }))
    }    

    /// Create a fragment shader.
    pub fn create_fragment_shader(self: &Rc<System>,code: &[u8]) -> Option<Rc<FragmentShader>> {
        let create_info = sys::VkShaderModuleCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            codeSize: code.len() as u64,
            pCode: code.as_ptr() as *const u32,
        };

        let mut vk_shader_module = MaybeUninit::uninit();
        match unsafe { sys::vkCreateShaderModule(self.gpu.vk_device,&create_info,null_mut(),vk_shader_module.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create shader (error {})",code);
                return None;
            },
        }
        let vk_shader_module = unsafe { vk_shader_module.assume_init() };

        Some(Rc::new(FragmentShader {
            system: Rc::clone(self),
            vk_shader_module: vk_shader_module,
        }))
    }

    /*
    /// create a vertex buffer.
    pub fn create_vertex_buffer<T: Vertex>(self: &Rc<Self>,vertices: &Vec<T>) -> Option<Rc<VertexBuffer>> {

        // obtain vertex info
        let vertex_base_fields = T::get_fields();
        let mut vertex_stride = 0usize;
        for (_,ty) in &vertex_base_fields {
            vertex_stride += ty.size();
        }

        // create vertex buffer
        let info = sys::VkBufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            size: (vertices.len() * vertex_stride) as u64,
            usage: sys::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
        };
        let mut vk_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateBuffer(self.gpu.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create vertex buffer (error {})",code);
                return None;
            }
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // allocate shared memory
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (vertices.len() * vertex_stride) as u64,
            memoryTypeIndex: self.gpu.shared_index as u32,
        };
        let mut vk_memory = MaybeUninit::<sys::VkDeviceMemory>::uninit();
        match unsafe { sys::vkAllocateMemory(self.gpu.vk_device,&info,null_mut(),vk_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to allocate memory (error {})",code);
                return None;
            }
        }
        let vk_memory = unsafe { vk_memory.assume_init() };

        // map memory
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            self.gpu.vk_device,
            vk_memory,
            0,
            sys::VK_WHOLE_SIZE as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to map memory (error {})",code);
                return None;
            }
        }
        let data_ptr = unsafe { data_ptr.assume_init() } as *mut T;

        // copy from the input vertices into data
        unsafe { copy_nonoverlapping(vertices.as_ptr(),data_ptr,vertices.len()) };

        // and unmap the memory again
        unsafe { sys::vkUnmapMemory(self.gpu.vk_device,vk_memory) };

        // bind to vertex buffer
        match unsafe { sys::vkBindBufferMemory(self.gpu.vk_device,vk_buffer,vk_memory,0) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to bind memory to vertex buffer (error {})",code);
                return None;
            }
        }

        Some(Rc::new(VertexBuffer {
            system: Rc::clone(self),
            vk_buffer: vk_buffer,
            vk_memory: vk_memory,
        }))
    }
    */

    /// create an index buffer.
    pub fn create_index_buffer<T>(self: &Rc<Self>,indices: &Vec<T>) -> Option<Rc<IndexBuffer>> {

        // create index buffer
        let info = sys::VkBufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            size: (indices.len() * 4) as u64,
            usage: sys::VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            sharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
        };
        let mut vk_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateBuffer(self.gpu.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create index buffer (error {})",code);
                return None;
            }
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // allocate shared memory
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (indices.len() * 4) as u64,
            memoryTypeIndex: self.gpu.shared_index as u32,
        };
        let mut vk_memory = MaybeUninit::<sys::VkDeviceMemory>::uninit();
        match unsafe { sys::vkAllocateMemory(self.gpu.vk_device,&info,null_mut(),vk_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to allocate memory (error {})",code);
                return None;
            }
        }
        let vk_memory = unsafe { vk_memory.assume_init() };

        // map memory
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            self.gpu.vk_device,
            vk_memory,
            0,
            sys::VK_WHOLE_SIZE as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to map memory (error {})",code);
                return None;
            }
        }
        let data_ptr = unsafe { data_ptr.assume_init() } as *mut T;

        // copy from the input vertices into data
        unsafe { copy_nonoverlapping(indices.as_ptr(),data_ptr,indices.len()) };

        // and unmap the memory again
        unsafe { sys::vkUnmapMemory(self.gpu.vk_device,vk_memory) };

        // bind to vertex buffer
        match unsafe { sys::vkBindBufferMemory(self.gpu.vk_device,vk_buffer,vk_memory,0) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to bind memory to index buffer (error {})",code);
                return None;
            }
        }

        Some(Rc::new(IndexBuffer {
            system: Rc::clone(self),
            vk_buffer: vk_buffer,
            vk_memory: vk_memory,
        }))
    }

    /// Create a semaphore.
    pub fn create_semaphore(self: &Rc<Self>) -> Option<Rc<Semaphore>> {

        let info = sys::VkSemaphoreCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
        };
        let mut vk_semaphore = MaybeUninit::uninit();
        match unsafe { sys::vkCreateSemaphore(self.gpu.vk_device,&info,null_mut(),vk_semaphore.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create semaphore (error {})",code);
                return None;
            },
        }
        Some(Rc::new(Semaphore {
            system: Rc::clone(self),
            vk_semaphore: unsafe { vk_semaphore.assume_init() },
        }))
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
