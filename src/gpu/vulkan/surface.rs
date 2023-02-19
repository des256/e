use {
    crate::*,
    std::{
        rc::Rc,
        mem::MaybeUninit,
        ptr::null_mut,
    },
};

pub(crate) struct SwapchainResources {
    pub gpu: Rc<Gpu>,
    pub vk_swapchain: sys::VkSwapchainKHR,
    pub vk_framebuffers: Vec<sys::VkFramebuffer>,
    pub vk_image_views: Vec<sys::VkImageView>,
}

impl SwapchainResources {

    /// Create swapchain resources for surface, render pass and rectangle.
    pub(crate) fn create(gpu: &Rc<Gpu>,vk_surface: sys::VkSurfaceKHR,vk_render_pass: sys::VkRenderPass,r: Rect<f32>) -> Result<SwapchainResources,String> {

        // get surface capabilities to calculate the extent and image count
        let mut capabilities = MaybeUninit::uninit();
        unsafe { sys::vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
            gpu.vk_physical_device,
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
            gpu.vk_physical_device,
            vk_surface,
            &mut count,
            null_mut(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to get surface formats (error {})",code));
            },
        }
        let mut formats = vec![MaybeUninit::<sys::VkSurfaceFormatKHR>::uninit(); count as usize];
        match unsafe { sys::vkGetPhysicalDeviceSurfaceFormatsKHR(
            gpu.vk_physical_device,
            vk_surface,
            &mut count,
            formats.as_mut_ptr() as *mut sys::VkSurfaceFormatKHR,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to get surface formats (error {})",code));
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
            return Err("window does not support BGRA8UN".to_string());
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
            gpu.vk_device,
            &info,
            null_mut(),
            vk_swapchain.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                return Err(format!("unable to create swap chain (error {})",code));
            },
        }
        let vk_swapchain = unsafe { vk_swapchain.assume_init() };

        // get swapchain images
        let mut count = 0u32;
        match unsafe { sys::vkGetSwapchainImagesKHR(gpu.vk_device,vk_swapchain,&mut count,null_mut()) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySwapchainKHR(gpu.vk_device,vk_swapchain,null_mut()) };
                return Err(format!("unable to get swap chain image count (error {})",code));
            },
        }
        let mut vk_images = vec![MaybeUninit::<sys::VkImage>::uninit(); count as usize];
        match unsafe { sys::vkGetSwapchainImagesKHR(
            gpu.vk_device,
            vk_swapchain,
            &count as *const u32 as *mut u32,
            vk_images.as_mut_ptr() as *mut sys::VkImage,
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySwapchainKHR(gpu.vk_device,vk_swapchain,null_mut()) };
                return Err(format!("unable to get swap chain images (error {})",code));
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
            match unsafe { sys::vkCreateImageView(gpu.vk_device,&info,null_mut(),vk_image_view.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    unsafe {
                        for vk_image_view in &vk_image_views {
                            sys::vkDestroyImageView(gpu.vk_device,*vk_image_view,null_mut());
                        }
                        sys::vkDestroySwapchainKHR(gpu.vk_device,vk_swapchain,null_mut());
                    }
                    return Err(format!("unable to create image view (error {})",code));
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
            match unsafe { sys::vkCreateFramebuffer(gpu.vk_device,&info,null_mut(),vk_framebuffer.as_mut_ptr()) } {
                sys::VK_SUCCESS => { },
                code => {
                    unsafe {
                        for vk_framebuffer in &vk_framebuffers {
                            sys::vkDestroyFramebuffer(gpu.vk_device,*vk_framebuffer,null_mut());
                        }
                        for vk_image_view in &vk_image_views {
                            sys::vkDestroyImageView(gpu.vk_device,*vk_image_view,null_mut());
                        }
                        sys::vkDestroySwapchainKHR(gpu.vk_device,vk_swapchain,null_mut());
                    }
                    return Err(format!("unable to create framebuffer (error {})",code));
                }
            }
            vk_framebuffers.push(unsafe { vk_framebuffer.assume_init() });
        }

        Ok(SwapchainResources {
            gpu: Rc::clone(&gpu),
            vk_swapchain,
            vk_image_views,
            vk_framebuffers,
        })
    }
}

impl Drop for SwapchainResources {
    fn drop(&mut self) {
        unsafe {
            for vk_framebuffer in &self.vk_framebuffers {
                sys::vkDestroyFramebuffer(self.gpu.vk_device,*vk_framebuffer,null_mut());
            }
            for vk_image_view in &self.vk_image_views {
                sys::vkDestroyImageView(self.gpu.vk_device,*vk_image_view,null_mut());
            }
            sys::vkDestroySwapchainKHR(self.gpu.vk_device,self.vk_swapchain,null_mut());
        }    
    }
}

pub(crate) struct Surface {
    pub gpu: Rc<Gpu>,
    pub vk_surface: sys::VkSurfaceKHR,
    pub vk_render_pass: sys::VkRenderPass,
    pub swapchain_resources: SwapchainResources,
}

impl Surface {

    pub(crate) fn create(gpu: &Rc<Gpu>,vk_surface: sys::VkSurfaceKHR,r: Rect<f32>) -> Result<Surface,String> {

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
        match unsafe { sys::vkCreateRenderPass(gpu.vk_device,&info,null_mut(),vk_render_pass.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                unsafe { sys::vkDestroySurfaceKHR(gpu.vk_instance,vk_surface,null_mut()) };
                return Err(format!("unable to create render pass (error {})",code));
            }
        }
        let vk_render_pass = unsafe { vk_render_pass.assume_init() };

        // create swapchain resources
        let swapchain_resources = SwapchainResources::create(gpu,vk_surface,vk_render_pass,r)?;

        Ok(Surface {
            gpu: Rc::clone(&gpu),
            vk_surface,
            vk_render_pass,
            swapchain_resources,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroySurfaceKHR(self.gpu.vk_instance,self.vk_surface,null_mut());
            sys::vkDestroyRenderPass(self.gpu.vk_device,self.vk_render_pass,null_mut());
        }
    }
}
