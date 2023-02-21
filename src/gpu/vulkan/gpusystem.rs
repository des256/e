use {
    crate::*,
    std::{
        result::Result,
        rc::Rc,
        ptr::null_mut,
        mem::MaybeUninit,
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
#[derive(Debug)]
pub(crate) struct GpuSystem {
    pub vk_instance: sys::VkInstance,
    pub vk_physical_device: sys::VkPhysicalDevice,
    pub vk_device: sys::VkDevice,
    pub vk_queue: sys::VkQueue,
    pub vk_command_pool: sys::VkCommandPool,
    pub shared_index: usize,
}

impl GpuSystem {

    pub(crate) fn open() -> Result<Rc<GpuSystem>,String> {

        // create instance
        dprintln!("creating instance");
        let extension_names = [
            sys::VK_KHR_SURFACE_EXTENSION_NAME.as_ptr(),
            sys::VK_KHR_XCB_SURFACE_EXTENSION_NAME.as_ptr(),
        ];
        let info = sys::VkInstanceCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo: &sys::VkApplicationInfo {
                sType: sys::VK_STRUCTURE_TYPE_APPLICATION_INFO,
                pNext: null_mut(),
                pApplicationName: b"e::System\0".as_ptr() as *const i8,
                applicationVersion: (1 << 22) as u32,
                pEngineName: b"e::GpuSystem\0".as_ptr() as *const i8,
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
            sys::VK_SUCCESS => { },
            code => return Err(format!("unable to create VkInstance ({})",code)),
        }
        let vk_instance = unsafe { vk_instance.assume_init() };
        dprintln!("vk_instance = {:?}",vk_instance);

        // enumerate physical devices
        dprintln!("enumerating physical devices");
        let mut count = MaybeUninit::<u32>::uninit();
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,count.as_mut_ptr(),null_mut()) };
        let count = unsafe { count.assume_init() };
        if count == 0 {
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return Err("unable to enumerate physical devices".to_string());
        }
        let mut vk_physical_devices = vec![null_mut() as sys::VkPhysicalDevice; count as usize];
        unsafe { sys::vkEnumeratePhysicalDevices(vk_instance,&count as *const u32 as *mut u32,vk_physical_devices.as_mut_ptr()) };

        // get first physical device
        let vk_physical_device = vk_physical_devices[0];
        dprintln!("vk_physical_device = {:?}",vk_physical_device);

        // DEBUG: show the name in debug build
#[cfg(build="debug")]
        {
            let mut properties = MaybeUninit::<sys::VkPhysicalDeviceProperties>::uninit();
            unsafe { sys::vkGetPhysicalDeviceProperties(vk_physical_device,properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            let slice: &[u8] = unsafe { &*(&properties.deviceName as *const [i8] as *const [u8]) };
            dprintln!("first physical device: {}",std::str::from_utf8(slice).unwrap());
        }
        
        // get supported queue families
        dprintln!("obtaining supported queue families");
        let mut count = MaybeUninit::<u32>::uninit();
        unsafe { sys::vkGetPhysicalDeviceQueueFamilyProperties(vk_physical_device,count.as_mut_ptr(),null_mut()) };
        let count = unsafe { count.assume_init() };
        if count == 0 {
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return Err("no queue families supported on this GPU".to_string());
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
        vk_queue_families.iter().for_each(|vk_queue_family| {
            let mut capabilities = String::new();
            if vk_queue_family.queueFlags & sys::VK_QUEUE_GRAPHICS_BIT != 0 {
                capabilities.push_str("graphics ");
            }
            if vk_queue_family.queueFlags & sys::VK_QUEUE_TRANSFER_BIT != 0 {
                capabilities.push_str("transfer ");
            }
            if vk_queue_family.queueFlags & sys::VK_QUEUE_COMPUTE_BIT != 0 {
                capabilities.push_str("compute ");
            }
            if vk_queue_family.queueFlags & sys::VK_QUEUE_SPARSE_BINDING_BIT != 0 {
                capabilities.push_str("sparse ");
            }
            dprintln!("    - {} queues, capable of: {}",vk_queue_family.queueCount,capabilities);
        });

        // assume the first queue family is the one we want for all queues
        let vk_queue_family = vk_queue_families[0];
        let mask = sys::VK_QUEUE_GRAPHICS_BIT | sys::VK_QUEUE_TRANSFER_BIT | sys::VK_QUEUE_COMPUTE_BIT;
        if (vk_queue_family.queueFlags & mask) != mask {
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return Err("queue family 0 of the GPU does not support graphics, transfer and compute operations".to_string());
        }

        // assume that presentation is done on the same family as graphics and create logical device with one queue of queue family 0
        dprintln!("creating device");
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
            pQueueCreateInfos: queue_create_infos.as_ptr(),
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
            unsafe { sys::vkDestroyInstance(vk_instance,null_mut()) };
            return Err("unable to create VkDevice".to_string());
        }
        let vk_device = unsafe { vk_device.assume_init() };
        dprintln!("vk_device = {:?}",vk_device);

        // obtain the queue from queue family 0
        dprintln!("obtaining queue from family 0");
        let mut vk_queue = MaybeUninit::uninit();
        unsafe { sys::vkGetDeviceQueue(vk_device,0,0,vk_queue.as_mut_ptr()) };
        let vk_queue = unsafe { vk_queue.assume_init() };
        dprintln!("vk_queue = {:?}",vk_queue);

        // create command pool for this queue
        dprintln!("create command pool");
        let info = sys::VkCommandPoolCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: null_mut(),
            flags: sys::VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            queueFamilyIndex: 0,
        };
        let mut vk_command_pool = MaybeUninit::uninit();
        if unsafe { sys::vkCreateCommandPool(vk_device,&info,null_mut(),vk_command_pool.as_mut_ptr()) } != sys::VK_SUCCESS {
            unsafe { 
                sys::vkDestroyDevice(vk_device,null_mut());
                sys::vkDestroyInstance(vk_instance,null_mut());
            }
            return Err("unable to create command pool".to_string());
        }
        let vk_command_pool = unsafe { vk_command_pool.assume_init() };
        dprintln!("vk_command_pool = {:?}",vk_command_pool);

        // get memory properties
        dprintln!("obtaining memory properties");
        let mut vk_memory_properties = MaybeUninit::<sys::VkPhysicalDeviceMemoryProperties>::uninit();
        unsafe { sys::vkGetPhysicalDeviceMemoryProperties(vk_physical_device,vk_memory_properties.as_mut_ptr()) };
        let vk_memory_properties = unsafe { vk_memory_properties.assume_init() };
        dprintln!("vk_memory_properties = {:?}",vk_memory_properties);

        // DEBUG: show the entire memory description
#[cfg(build="debug")]
        {
            dprintln!("device memory properties:");
            dprintln!("    memory types:");
            for i in 0..vk_memory_properties.memoryTypeCount as usize {
                let mut flags = String::new();
                let vk_memory_type = &vk_memory_properties.memoryTypes[i];
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
                dprintln!("        - on heap {}, {}",vk_memory_type.heapIndex,flags);
            }
            dprintln!("    memory heaps:");
            for i in 0..vk_memory_properties.memoryHeapCount as usize {
                dprintln!("        - size {} MiB, {:X}",vk_memory_properties.memoryHeaps[i].size / (1024 * 1024),vk_memory_properties.memoryHeaps[i].flags);
            }
        }

        // find shared memory heap and type (later also find device-only index)
        dprintln!("finding shared memory heap and type");
        let mask = sys::VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        let valid_types: Vec<(usize,&sys::VkMemoryType)> = vk_memory_properties.memoryTypes.iter().enumerate().filter(|vk_memory_type| (vk_memory_type.1.propertyFlags & mask) == mask).collect();
        if valid_types.is_empty() {
            return Err("no valid memory types found".to_string());
        }
        let shared_index = valid_types[0].0;
        dprintln!("shared_index = {}",shared_index);

        Ok(Rc::new(GpuSystem {
            vk_instance,
            vk_physical_device,
            vk_device,
            vk_queue,
            vk_command_pool,
            shared_index,
        }))
    }
}
