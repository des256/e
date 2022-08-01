use {
    crate::*,
    std::ptr::null_mut,
};

#[cfg(feature="gpu_vulkan")]
use std::mem::MaybeUninit;

#[doc(hidden)]
pub(crate) struct ScreenDescriptor {
    pub(crate) xcb_screen: *const sys::xcb_screen_t,
    pub(crate) name: String,
#[cfg(feature="gpu_vulkan")]
    pub(crate) present_queue_family_index: usize,
}

/// The GPU interface.
pub struct Gpu<'system> {
    pub(crate) system: &'system System,
    pub(crate) name: String,
#[doc(hidden)]    
    pub(crate) screen_descriptors: Vec<ScreenDescriptor>,
#[cfg(feature="gpu_vulkan")]
    pub(crate) vk_physical_device: sys::VkPhysicalDevice,
#[cfg(feature="gpu_vulkan")]
    pub(crate) queue_family_descriptors: Vec<QueueFamilyDescriptor>,
}

impl<'system> System {

    /// Create interface to one of the enumerated GPUs.
    pub fn create_gpu(&self,index: usize) -> Option<Gpu> {

        // make sure index is within range
#[cfg(feature="gpu_vulkan")]
        if index >= self.gpu_descriptors.len() {
            println!("GPU index out of range");
            return None;
        }

#[cfg(feature="gpu_opengl")]
        if index != 0 {
            println!("GPU index out of range");
            return None;
        }

        // build screen descriptors
        let mut screen_descriptors = Vec::<ScreenDescriptor>::new();
        let xcb_setup = unsafe { sys::xcb_get_setup(self.xcb_connection) };
        if xcb_setup == null_mut() {
            println!("Unable to obtain X server setup.");
            return None;
        }
        let mut iterator = unsafe { sys::xcb_setup_roots_iterator(xcb_setup) };
        while iterator.rem != 0 {
            let xcb_screen = iterator.data;

#[cfg(feature="gpu_vulkan")]
            {
                // create test window on this screen
                let xcb_window = unsafe { sys::xcb_generate_id(self.xcb_connection) };
                unsafe {
                    sys::xcb_create_window(
                        self.xcb_connection,
                        (*xcb_screen).root_depth as u8,
                        xcb_window as u32,
                        //if let Some(id) = parent { id as u32 } else { system.rootwindow as u32 },
                        (*xcb_screen).root as u32,
                        0i16,0i16,1u16,1u16,
                        0,
                        sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                        (*xcb_screen).root_visual as u32,
                        sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
                        &[
                            sys::XCB_EVENT_MASK_EXPOSURE
                            | sys::XCB_EVENT_MASK_KEY_PRESS
                            | sys::XCB_EVENT_MASK_KEY_RELEASE
                            | sys::XCB_EVENT_MASK_BUTTON_PRESS
                            | sys::XCB_EVENT_MASK_BUTTON_RELEASE
                            | sys::XCB_EVENT_MASK_POINTER_MOTION
                            | sys::XCB_EVENT_MASK_STRUCTURE_NOTIFY,
                            (*xcb_screen).default_colormap,
                        ] as *const u32 as *const std::os::raw::c_void,
                    );
                    sys::xcb_map_window(self.xcb_connection,xcb_window as u32);
                    sys::xcb_flush(self.xcb_connection);
                }

                // create surface for the window
                let mut vk_surface = MaybeUninit::uninit();
                if unsafe { sys::vkCreateXcbSurfaceKHR(
                    self.vk_instance,
                    &sys::VkXcbSurfaceCreateInfoKHR {
                        sType: sys::VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
                        pNext: null_mut(),
                        flags: 0,
                        connection: self.xcb_connection,
                        window: xcb_window,
                    },
                    null_mut(),
                    vk_surface.as_mut_ptr(),
                ) } == sys::VK_SUCCESS {

                    let vk_surface = unsafe { vk_surface.assume_init() };

                    // make sure we can present from the first available graphics queue family
                    let mut present_queue_family_index: Option<usize> = None;
                    for queue_family_descriptor in &self.gpu_descriptors[index].queue_family_descriptors {
                        let mut present_support: sys::VkBool32 = 0;
                        unsafe { sys::vkGetPhysicalDeviceSurfaceSupportKHR(self.gpu_descriptors[index].vk_physical_device,queue_family_descriptor.index as u32,vk_surface,&mut present_support as *mut sys::VkBool32) };
                        if present_support != 0 {
                            present_queue_family_index = Some(queue_family_descriptor.index);
                            break;
                        }
                    }

                    // if that worked out, add this screen to the list
                    if let Some(present_queue_family_index) = present_queue_family_index {
                        screen_descriptors.push(ScreenDescriptor {
                            xcb_screen: xcb_screen,
                            name: format!("{} screen {}",self.gpu_descriptors[index].name,iterator.index),
                            present_queue_family_index: present_queue_family_index,
                        });
                    }

                    unsafe { sys::vkDestroySurfaceKHR(self.vk_instance,vk_surface,null_mut()) };
                }

                unsafe {
                    sys::xcb_unmap_window(self.xcb_connection,xcb_window as u32);
                    sys::xcb_destroy_window(self.xcb_connection,xcb_window as u32);
                }
            }

#[cfg(feature="gpu_opengl")]
            {
                // just add all screens, assuming that OpenGL can reach them
                screen_descriptors.push(ScreenDescriptor {
                    xcb_screen: xcb_screen,
                    name: format!("Default Screen {}",iterator.index),
                });
            }

            unsafe { sys::xcb_screen_next(&mut iterator as *mut sys::xcb_screen_iterator_t) };
        }

        Some(Gpu {
            system: &self,
#[cfg(feature="gpu_vulkan")]
            name: self.gpu_descriptors[index].name.clone(),
#[cfg(feature="gpu_opengl")]
            name: "Default GPU".to_string(),
            screen_descriptors: screen_descriptors,
#[cfg(feature="gpu_vulkan")]
            vk_physical_device: self.gpu_descriptors[index].vk_physical_device,
#[cfg(feature="gpu_vulkan")]
            queue_family_descriptors: self.gpu_descriptors[index].queue_family_descriptors.clone(),
        })
    }        
}

impl<'system,'gpu> Gpu<'system> {

    /// Enumerate all screens connected to this GPU.
    pub fn enumerate_screens(&self) -> Vec<String> {
        let mut screens: Vec<String> = Vec::new();
        for screen_descriptor in &self.screen_descriptors {
            screens.push(screen_descriptor.name.clone());
        }
        screens
    }
}