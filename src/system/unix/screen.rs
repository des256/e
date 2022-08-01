use {
    crate::*,
    std::ptr::null_mut,
};

#[cfg(feature="gpu_vulkan")]
use {
    std::mem::MaybeUninit,
};

/// The screen interface.
pub struct Screen<'system,'gpu> {
    pub gpu: &'gpu Gpu<'system>,
    pub name: String,
#[doc(hidden)]
    pub(crate) xcb_screen: *const sys::xcb_screen_t,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_device: sys::VkDevice,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_present_queue: sys::VkQueue,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_graphics_queue: sys::VkQueue,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_transfer_queue: sys::VkQueue,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_compute_queue: sys::VkQueue,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]    
    pub(crate) vk_graphics_command_pool: sys::VkCommandPool,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]    
    pub(crate) vk_transfer_command_pool: sys::VkCommandPool,
#[doc(hidden)]
#[cfg(feature="gpu_vulkan")]    
    pub(crate) vk_compute_command_pool: sys::VkCommandPool,
}

impl<'system,'gpu> Gpu<'system> {

#[cfg(feature="gpu_vulkan")]
    fn create_command_pool(&self,vk_device: sys::VkDevice,index: usize) -> Option<sys::VkCommandPool> {
        let create_info = sys::VkCommandPoolCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            queueFamilyIndex: index as u32,
        };
        let mut vk_command_pool = MaybeUninit::uninit();
        if unsafe { sys::vkCreateCommandPool(vk_device,&create_info,null_mut(),vk_command_pool.as_mut_ptr()) } == sys::VK_SUCCESS {
            Some(unsafe { vk_command_pool.assume_init() })
        }
        else {
            println!("Unable to create command pool");
            None
        }    
    }

    /// Create interface to one of the enumerated screens.
    pub fn open_screen(&self,index: usize) -> Option<Screen> {

        // make sure index is within range
        if index >= self.screen_descriptors.len() {
            println!("Screen index out of range");
            return None;
        }

        #[cfg(feature="gpu_vulkan")]
        {
            // count how many of which queue family we actually need
            let mut max_index = 0usize;
            for queue_family_descriptor in &self.queue_family_descriptors {
                if queue_family_descriptor.index > max_index {
                    max_index = queue_family_descriptor.index;
                }
            }
            let mut queue_families = vec![0usize; max_index + 1];
            println!("total number of queue families: {}",max_index + 1);
            queue_families[self.screen_descriptors[index].present_queue_family_index] += 1;
            println!("present goes to {}",self.screen_descriptors[index].present_queue_family_index);
            let mut graphics_family_index: Option<usize> = None;
            let mut transfer_family_index: Option<usize> = None;
            let mut compute_family_index: Option<usize> = None;
            for queue_family_descriptor in &self.queue_family_descriptors {
                if queue_family_descriptor.graphics {
                    if let None = graphics_family_index {
                        println!("graphics goes to {}",queue_family_descriptor.index);
                        graphics_family_index = Some(queue_family_descriptor.index);
                    }
                }
                if queue_family_descriptor.transfer {
                    if let None = transfer_family_index {
                        println!("transfer goes to {}",queue_family_descriptor.index);
                        transfer_family_index = Some(queue_family_descriptor.index);
                    }
                }
                if queue_family_descriptor.compute {
                    if let None = compute_family_index {
                        println!("compute goes to {}",queue_family_descriptor.index);
                        compute_family_index = Some(queue_family_descriptor.index);
                    }
                }
            }
            if let None = graphics_family_index {
                println!("cannot find family for graphics queue");
                return None;
            }
            let graphics_family_index = graphics_family_index.unwrap();
            if let None = transfer_family_index {
                println!("cannot find family for transfer queue");
                return None;
            }
            let transfer_family_index = transfer_family_index.unwrap();
            if let None = compute_family_index {
                println!("cannot find family for compute queue");
                return None;
            }
            let compute_family_index = compute_family_index.unwrap();
            let mut graphics_index = queue_families[graphics_family_index];
            queue_families[graphics_family_index] += 1;
            let mut transfer_index = queue_families[transfer_family_index];
            queue_families[transfer_family_index] += 1;
            let mut compute_index = queue_families[compute_family_index];
            queue_families[compute_family_index] += 1;

            println!("so:");

            // only request enough queues of the relevant families
            let mut queue_create_infos = Vec::<sys::VkDeviceQueueCreateInfo>::new();
            let priority = 1f32;
            for i in 0..queue_families.len() {
                if queue_families[i] > 0 {
                    println!("    request {} of family {}",queue_families[i],i);
                    let priority = 1f32;
                    queue_create_infos.push(sys::VkDeviceQueueCreateInfo {
                        sType: sys::VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                        pNext: null_mut(),
                        flags: 0,
                        queueFamilyIndex: i as u32,
                        queueCount: queue_families[i] as u32,
                        pQueuePriorities: &priority as *const f32,
                    });
                }
            }
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
            if unsafe { sys::vkCreateDevice(self.vk_physical_device,&create_info,null_mut(),vk_device.as_mut_ptr()) } == sys::VK_SUCCESS {

                let vk_device = unsafe { vk_device.assume_init() };

                // get the requested queues
                let mut vk_present_queue: sys::VkQueue = null_mut();
                unsafe { sys::vkGetDeviceQueue(vk_device,self.screen_descriptors[index].present_queue_family_index as u32, 0, &mut vk_present_queue) };
                let mut vk_graphics_queue: sys::VkQueue = null_mut();
                unsafe { sys::vkGetDeviceQueue(vk_device,graphics_family_index as u32, graphics_index as u32, &mut vk_graphics_queue) };
                let mut vk_transfer_queue: sys::VkQueue = null_mut();
                unsafe { sys::vkGetDeviceQueue(vk_device,transfer_family_index as u32, transfer_index as u32, &mut vk_transfer_queue) };
                let mut vk_compute_queue: sys::VkQueue = null_mut();
                unsafe { sys::vkGetDeviceQueue(vk_device,compute_family_index as u32, compute_index as u32, &mut vk_compute_queue) };

                // create command pools for all queues except present
                let vk_graphics_command_pool = self.create_command_pool(vk_device,graphics_family_index);
                if let None = vk_graphics_command_pool {
                    unsafe { sys::vkDestroyDevice(vk_device,null_mut()) };
                    return None;
                }
                let vk_graphics_command_pool = vk_graphics_command_pool.unwrap();
                let vk_transfer_command_pool = self.create_command_pool(vk_device,transfer_family_index);
                if let None = vk_transfer_command_pool {
                    unsafe { sys::vkDestroyDevice(vk_device,null_mut()) };
                    return None;
                }
                let vk_transfer_command_pool = vk_transfer_command_pool.unwrap();
                let vk_compute_command_pool = self.create_command_pool(vk_device,compute_family_index);
                if let None = vk_compute_command_pool {
                    unsafe { sys::vkDestroyDevice(vk_device,null_mut()) };
                    return None;
                }
                let vk_compute_command_pool = vk_compute_command_pool.unwrap();
    
                Some(Screen {
                    gpu: &self,
                    name: self.screen_descriptors[index].name.clone(),
                    xcb_screen: self.screen_descriptors[index].xcb_screen,
                    vk_device: vk_device,
                    vk_present_queue: vk_present_queue,
                    vk_graphics_queue: vk_graphics_queue,
                    vk_transfer_queue: vk_transfer_queue,
                    vk_compute_queue: vk_compute_queue,
                    vk_graphics_command_pool: vk_graphics_command_pool,
                    vk_transfer_command_pool: vk_transfer_command_pool,
                    vk_compute_command_pool: vk_compute_command_pool,
                })
            }
            else {
                None
            }
        }

#[cfg(feature="gpu_opengl")]
        Screen {
            gpu: &self,
            name: self.screen_descriptors[index].name.clone(),
            xcb_screen: self.screen_descriptors[index].xcb_screen,
        }
    }
}

impl<'system,'gpu> Screen<'system,'gpu> {
    /*
    pub fn submit_graphics(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
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
        match unsafe { sys::vkQueueSubmit(self.vk_graphics_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("Unable to submit to graphics queue (error {}).",code);
                false
            },
        }
    }
    */

    /*
    pub fn submit_transfer(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
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
        match unsafe { sys::vkQueueSubmit(self.vk_transfer_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("Unable to submit to transfer queue (error {}).",code);
                false
            },
        }
    }
    */

    /*
    pub fn submit_compute(&self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {
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
        match unsafe { sys::vkQueueSubmit(self.vk_compute_queue,1,&info,null_mut()) } {
            sys::VK_SUCCESS => true,
            code => {
                println!("Unable to submit to compute queue (error {}).",code);
                false
            },
        }
    }
    */
}

impl<'system,'gpu> Drop for Screen<'system,'gpu> {
    fn drop(&mut self) {
        unsafe {
            unsafe { sys::vkDestroyCommandPool(self.vk_device,self.vk_graphics_command_pool,null_mut()) };
            unsafe { sys::vkDestroyCommandPool(self.vk_device,self.vk_transfer_command_pool,null_mut()) };
            unsafe { sys::vkDestroyCommandPool(self.vk_device,self.vk_compute_command_pool,null_mut()) };
            unsafe { sys::vkDestroyDevice(self.vk_device,null_mut()) };               
        }
    }
}
