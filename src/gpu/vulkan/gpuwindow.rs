use {
    crate::*,
    std::{
        rc::Rc,
        mem::MaybeUninit,
        ptr::null_mut,
        cell::RefCell,
    },
};

#[derive(Debug)]
pub(crate) struct Swapchain {
    pub system: Rc<System>,
    pub vk_swapchain: sys::VkSwapchainKHR,
    pub vk_framebuffers: Vec<sys::VkFramebuffer>,
    pub vk_image_views: Vec<sys::VkImageView>,
}

impl Swapchain {

    /// Create swapchain for surface, render pass and rectangle.
    pub(crate) fn new(system: &Rc<System>,vk_surface: sys::VkSurfaceKHR,vk_render_pass: sys::VkRenderPass,r: &Rect<i32>) -> Result<Swapchain,String> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::<sys::VkSurfaceCapabilitiesKHR>::uninit();
        match unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            system.gpu_system.vk_physical_device,
            vk_surface,
            capabilities.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to get surface capabilities ({})",vk_code_to_string(code)));
            },
        }
        let capabilities = unsafe { capabilities.assume_init() };

        // get current extent, if any
        let extent = if capabilities.currentExtent.width != 0xFFFFFFFF {
            Vec2 {
                x: capabilities.currentExtent.width,
                y: capabilities.currentExtent.height,
            }
        }

        // otherwise take window size as extent, and make sure it fits the constraints
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
            system.gpu_system.vk_physical_device,
            vk_surface,
            &mut count as *mut u32,
            null_mut(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to get surface formats ({})",vk_code_to_string(code)));
            },
        }
        let mut formats = vec![MaybeUninit::<sys::VkSurfaceFormatKHR>::uninit(); count as usize];
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            system.gpu_system.vk_physical_device,
            vk_surface,
            &mut count,
            formats.as_mut_ptr() as *mut sys::VkSurfaceFormatKHR,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to get surface formats ({})",vk_code_to_string(code)));
            }
        }
        let formats = unsafe { std::mem::transmute::<_,Vec<sys::VkSurfaceFormatKHR>>(formats) };
        let format_supported = formats.iter().any(|vk_format| (vk_format.format == sys::VK_FORMAT_B8G8R8A8_SRGB) && (vk_format.colorSpace == sys::VK_COLOR_SPACE_SRGB_NONLINEAR_KHR));
        if !format_supported {
            return Err("window does not support BGRA8UN at SRGB".to_string());
        }

        // create swapchain for this window
        let info = sys::VkSwapchainCreateInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
            pNext: null_mut(),
            flags: 0,
            surface: vk_surface,
            minImageCount: capabilities.minImageCount,
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
        let mut vk_swapchain: sys::VkSwapchainKHR = null_mut();
        match unsafe { sys::vkCreateSwapchainKHR(
            system.gpu_system.vk_device,
            &info,
            null_mut(),
            &mut vk_swapchain as *mut sys::VkSwapchainKHR,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to create swap chain ({})",vk_code_to_string(code)));
            },
        }

        // get swapchain images
        let mut count = 0u32;
        match unsafe { sys::vkGetSwapchainImagesKHR(system.gpu_system.vk_device,vk_swapchain,&mut count as *mut u32,null_mut()) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySwapchainKHR(system.gpu_system.vk_device,vk_swapchain,null_mut()) };
                return Err(format!("unable to get swap chain image count ({})",vk_code_to_string(code)));
            },
        }
        let mut vk_images = vec![MaybeUninit::<sys::VkImage>::uninit(); count as usize];
        match unsafe { sys::vkGetSwapchainImagesKHR(
            system.gpu_system.vk_device,
            vk_swapchain,
            &count as *const u32 as *mut u32,
            vk_images.as_mut_ptr() as *mut sys::VkImage,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySwapchainKHR(system.gpu_system.vk_device,vk_swapchain,null_mut()) };
                return Err(format!("unable to get swap chain images ({})",vk_code_to_string(code)));
            },
        }
        let vk_images = unsafe { std::mem::transmute::<_,Vec<sys::VkImage>>(vk_images) };

        // create image views for the swapchain images
        let results: Vec<Result<sys::VkImageView,String>> = vk_images.iter().map(|vk_image| {
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
            let mut vk_image_view: sys::VkImageView = null_mut();
            match unsafe { sys::vkCreateImageView(system.gpu_system.vk_device,&info,null_mut(),&mut vk_image_view) } {
                sys::VK_SUCCESS => Ok(vk_image_view),
                code => Err(format!("unable to create image view ({})",vk_code_to_string(code))),
            }
        }).collect();
        if results.iter().any(|result| result.is_err()) {
            results.iter().for_each(|result| if let Ok(vk_image_view) = result { unsafe { sys::vkDestroyImageView(system.gpu_system.vk_device,*vk_image_view,null_mut()) } });
            unsafe { sys::vkDestroySwapchainKHR(system.gpu_system.vk_device,vk_swapchain,null_mut()); }
            return Err("unable to create image view".to_string());
        }
        let vk_image_views: Vec<sys::VkImageView> = results.into_iter().map(|result| result.unwrap()).collect();

        // create framebuffers for the image views
        let results: Vec<Result<sys::VkFramebuffer,String>> = vk_image_views.iter().map(|vk_image_view| {
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
            match unsafe { sys::vkCreateFramebuffer(system.gpu_system.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
                sys::VK_SUCCESS => Ok(unsafe { vk_framebuffer.assume_init() }),
                code => Err(format!("unable to create framebuffer ({})",vk_code_to_string(code))),
            }
        }).collect();
        if results.iter().any(|result| result.is_err()) {
            results.iter().for_each(|result| if let Ok(vk_framebuffer) = result { unsafe { sys::vkDestroyFramebuffer(system.gpu_system.vk_device,*vk_framebuffer,null_mut()) } });
            vk_image_views.iter().for_each(|vk_image_view| unsafe { sys::vkDestroyImageView(system.gpu_system.vk_device,*vk_image_view,null_mut()) });
            return Err("unable to create framebuffer".to_string());
        }
        let vk_framebuffers: Vec<sys::VkFramebuffer> = results.into_iter().map(|result| result.unwrap()).collect();
        Ok(Swapchain {
            system: Rc::clone(system),
            vk_swapchain,
            vk_image_views,
            vk_framebuffers,
        })
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        self.vk_framebuffers.iter().for_each(|vk_framebuffer| unsafe { sys::vkDestroyFramebuffer(self.system.gpu_system.vk_device,*vk_framebuffer,null_mut()) });
        self.vk_image_views.iter().for_each(|vk_image_view| unsafe { sys::vkDestroyImageView(self.system.gpu_system.vk_device,*vk_image_view,null_mut()) });
        unsafe { sys::vkDestroySwapchainKHR(self.system.gpu_system.vk_device,self.vk_swapchain,null_mut()); }
    }
}

pub(crate) struct GpuWindow {
    pub system: Rc<System>,
    pub vk_surface: sys::VkSurfaceKHR,
    pub vk_render_pass: sys::VkRenderPass,
    pub swapchain: RefCell<Swapchain>,
}

impl GpuWindow {

    pub(crate) fn create(system: &Rc<System>,r: Rect<i32>,xcb_connection: *mut sys::xcb_connection_t,xcb_window: u32) -> Result<GpuWindow,String> {

        // create surface for this window
        let info = sys::VkXcbSurfaceCreateInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            pNext: null_mut(),
            flags: 0,
            connection: xcb_connection,
            window: xcb_window,
        };
        let mut vk_surface = MaybeUninit::<sys::VkSurfaceKHR>::uninit();
        match unsafe { sys::vkCreateXcbSurfaceKHR(system.gpu_system.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("Unable to create Vulkan XCB surface ({})",vk_code_to_string(code)));
            },
        }
        let vk_surface = unsafe { vk_surface.assume_init() };

        // verify the surface is supported for the current physical device
        let mut supported = MaybeUninit::<sys::VkBool32>::uninit();
        match unsafe { sys::vkGetPhysicalDeviceSurfaceSupportKHR(system.gpu_system.vk_physical_device,0,vk_surface,supported.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("Surface not supported on physical device ({})",vk_code_to_string(code)));
            },
        }
        let supported = unsafe { supported.assume_init() };
        if supported == sys::VK_FALSE {
            return Err("Surface not supported on physical device".to_string());
        }

        // create render pass

        // A render pass describes the buffers and how they interact for a specific rendering type. This is probably helpful for the GPU to optimize tiling.
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
        match unsafe { sys::vkCreateRenderPass(system.gpu_system.vk_device,&info,null_mut(),vk_render_pass.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySurfaceKHR(system.gpu_system.vk_instance,vk_surface,null_mut()) };
                return Err(format!("unable to create render pass ({})",vk_code_to_string(code)));
            }
        }
        let vk_render_pass = unsafe { vk_render_pass.assume_init() };
        dprintln!("vk_render_pass = {:?}",vk_render_pass);

        // create swapchain
        let swapchain = RefCell::new(Swapchain::new(system,vk_surface,vk_render_pass,&r)?);

        Ok(GpuWindow {
            system: Rc::clone(system),
            vk_surface,
            vk_render_pass,
            swapchain,
        })
    }

    pub fn update_swapchain(&self,r: &Rect<i32>) {
        if let Ok(swapchain) = Swapchain::new(&self.system,self.vk_surface,self.vk_render_pass,r) {
            *(self.swapchain.borrow_mut()) = swapchain;
        }
    }

    pub fn get_framebuffer_count(&self) -> usize {
        self.swapchain.borrow().vk_framebuffers.len()
    }

    pub fn acquire(&self,signal_semaphore: &Semaphore) -> Result<usize,String> {
        let mut index = 0u32;
        match unsafe { sys::vkAcquireNextImageKHR(self.system.gpu_system.vk_device,self.swapchain.borrow().vk_swapchain,0xFFFFFFFFFFFFFFFF,signal_semaphore.vk_semaphore,null_mut(),&mut index,) } {
            sys::VK_SUCCESS => Ok(index as usize),
            code => Err(format!("Unable to acquire next image ({})",vk_code_to_string(code))),
        }
    }

    pub fn present(&self,index: usize,wait_semaphore: &Semaphore) -> Result<(),String> {
        let image_index = index as u32;
        let info = sys::VkPresentInfoKHR {
            sType: sys::VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            pNext: null_mut(),
            waitSemaphoreCount: 1,
            pWaitSemaphores: &wait_semaphore.vk_semaphore,
            swapchainCount: 1,
            pSwapchains: &self.swapchain.borrow().vk_swapchain,
            pImageIndices: &image_index,
            pResults: null_mut(),
        };
        match unsafe { sys::vkQueuePresentKHR(self.system.gpu_system.vk_queue,&info) } {
            sys::VK_SUCCESS => Ok(()),
            code => Err(format!("Unable to present image ({})",vk_code_to_string(code))),
        }
    }
}

impl Drop for GpuWindow {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroySurfaceKHR(self.system.gpu_system.vk_instance,self.vk_surface,null_mut());
            sys::vkDestroyRenderPass(self.system.gpu_system.vk_device,self.vk_render_pass,null_mut());
        }
    }
}
