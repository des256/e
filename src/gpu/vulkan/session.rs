// E - GPU (Vulkan) - Device
// Desmond Germans, 2020

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct Session {
    pub gpu: Rc<Gpu>,
    pub(crate) vk_device: VkDevice,
    pub(crate) vk_command_pools: Vec<VkCommandPool>,
}

impl Gpu {
    
    pub fn create_session(self: &Rc<Gpu>,queues: Vec<(QueueFamilyID,usize)>) -> Option<Rc<Session>> {

        let mut queue_create_infos = Vec::<VkDeviceQueueCreateInfo>::new();
        let mut prioritiesies = Vec::<Vec<f32>>::new();
        for queue in &queues {
            let mut priorities = Vec::<f32>::new();
            for _ in 0..queue.1 {
                priorities.push(1.0);
            }
            let size = prioritiesies.len();
            prioritiesies.push(priorities);
            queue_create_infos.push(VkDeviceQueueCreateInfo {
                sType: VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                queueFamilyIndex: queue.0,
                queueCount: queue.1 as u32,
                pQueuePriorities: prioritiesies[size].as_mut_ptr(),
            });
        }

        let extension_names = [
            VK_KHR_SWAPCHAIN_EXTENSION_NAME.as_ptr(),
        ];
        let create_info = VkDeviceCreateInfo {
            sType: VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueCreateInfoCount: queue_create_infos.len() as u32,
            pQueueCreateInfos: queue_create_infos.as_mut_ptr(),
            enabledLayerCount: 0,
            ppEnabledLayerNames: null_mut(),
            enabledExtensionCount: extension_names.len() as u32,
            ppEnabledExtensionNames: extension_names.as_ptr() as *const *const i8,
            pEnabledFeatures: &VkPhysicalDeviceFeatures {
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
        match unsafe { vkCreateDevice(self.vk_physical_device,&create_info,null_mut(),vk_device.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create Vulkan logical device (error {})",code);
                return None;
            },
        }
        let vk_device = unsafe { vk_device.assume_init() };

        let mut vk_command_pools = Vec::<VkCommandPool>::new();
        for queue in &queues {
            for _ in 0..queue.1 {
                let create_info = VkCommandPoolCreateInfo {
                    sType: VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    queueFamilyIndex: queue.0 as u32,
                };
        
                let mut vk_command_pool = MaybeUninit::uninit();
                match unsafe { vkCreateCommandPool(vk_device,&create_info,null_mut(),vk_command_pool.as_mut_ptr()) } {
                    VK_SUCCESS => { },
                    code => {
#[cfg(feature="debug_output")]
                        println!("Unable to create Vulkan command pool (error {})",code);
                        return None;
                    },
                }
                vk_command_pools.push(unsafe { vk_command_pool.assume_init() });
            }
        }

        Some(Rc::new(Session {
            gpu: Rc::clone(self),
            vk_device: vk_device,
            vk_command_pools: vk_command_pools,
        }))

    }
}

impl Drop for Session {
    fn drop(&mut self) {
        for vk_command_pool in &self.vk_command_pools {
            unsafe { vkDestroyCommandPool(self.vk_device,*vk_command_pool,null_mut()) };
        }
        unsafe { vkDestroyDevice(self.vk_device,null_mut()) };
    }
}
