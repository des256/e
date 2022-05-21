use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct Screen {
    pub system: Rc<System>,
    pub gpu: Rc<GPU>,
    pub name: String,
    pub graphics_queue_id: QueueFamilyID,
    pub present_queue_id: QueueFamilyID,
    pub(crate) xcb_screen: *const xcb_screen_t,
}

impl System {

#[cfg(feature="gpu_vulkan")]
    fn find_gpu_window_queue_families(&self,gpu: &GPU,xcb_window: xcb_window_t) -> Option<(QueueFamilyID,QueueFamilyID)> {

        // create surface for the window
        let info = VkXcbSurfaceCreateInfoKHR {
            sType: VK_STRUCTURE_TYPE_XCB_SURFACE_CREATE_INFO_KHR,
            pNext: null_mut(),
            flags: 0,
            connection: self.xcb_connection,
            window: xcb_window,
        };
        let mut vk_surface = MaybeUninit::uninit();
        match unsafe { vkCreateXcbSurfaceKHR(self.vk_instance,&info,null_mut(),vk_surface.as_mut_ptr()) } {
            VK_SUCCESS => { },
            _ => {
                return None;
            },
        }
        let vk_surface = unsafe { vk_surface.assume_init() };

        // find queue that can accept graphics command buffers
        let mut vk_graphics_queue_id: Option<QueueFamilyID> = None;
        for queue_family in &gpu.queue_families {
            if queue_family.0.graphics {
                vk_graphics_queue_id = Some(queue_family.0.id);
                break;
            }
        }
        if let None = vk_graphics_queue_id {
            return None;
        }

        // find queue that can be used for presentation
        let mut vk_present_queue_id: Option<QueueFamilyID> = None;
        for queue_family in &gpu.queue_families {
            let mut support: VkBool32 = 0;
            unsafe { vkGetPhysicalDeviceSurfaceSupportKHR(gpu.vk_physical_device,queue_family.0.id,vk_surface,&mut support) };
            if support != 0 {
                vk_present_queue_id = Some(queue_family.0.id);
                break;
            }
        }
        if let None = vk_present_queue_id {
            return None;
        }

        // destroy the surface
        unsafe { vkDestroySurfaceKHR(self.vk_instance,vk_surface,null_mut()) };

        // and return the indices
        Some((vk_graphics_queue_id.unwrap(),vk_present_queue_id.unwrap()))
    }

    pub fn find_screens(self: &Rc<Self>,gpus: &Vec<Rc<GPU>>) -> Vec<Rc<Screen>> {

        // TODO: somehow enumerate all attached screens

        // just take the first screen for now
        let xcb_setup = unsafe { xcb_get_setup(self.xcb_connection) };
        if xcb_setup == null_mut() {
#[cfg(feature="debug_output")]
            println!("Unable to obtain X server setup.");
            return Vec::new();
        }
        let xcb_screen = unsafe { xcb_setup_roots_iterator(xcb_setup).data };
        if xcb_screen == null_mut() {
#[cfg(feature="debug_output")]
            println!("Unable to obtain X root screen.");
            return Vec::new();
        }

#[cfg(feature="gpu_vulkan")]
        let (best_gpu,vk_graphics_queue_id,vk_present_queue_id) = {

            // create window
            let xcb_window = unsafe { xcb_generate_id(self.xcb_connection) };
            let values = [XCB_EVENT_MASK_EXPOSURE
                | XCB_EVENT_MASK_KEY_PRESS
                | XCB_EVENT_MASK_KEY_RELEASE
                | XCB_EVENT_MASK_BUTTON_PRESS
                | XCB_EVENT_MASK_BUTTON_RELEASE
                | XCB_EVENT_MASK_POINTER_MOTION
                | XCB_EVENT_MASK_STRUCTURE_NOTIFY,
                unsafe { *xcb_screen }.default_colormap,
            ];
            unsafe {
                xcb_create_window(
                    self.xcb_connection,
                    (*xcb_screen).root_depth as u8,
                    xcb_window as u32,
                    //if let Some(id) = parent { id as u32 } else { system.rootwindow as u32 },
                    (*xcb_screen).root as u32,
                    0i16,0i16,1u16,1u16,
                    0,
                    XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                    (*xcb_screen).root_visual as u32,
                    XCB_CW_EVENT_MASK | XCB_CW_COLORMAP,
                    &values as *const u32 as *const std::os::raw::c_void
                );
                xcb_map_window(self.xcb_connection,xcb_window as u32);
                xcb_flush(self.xcb_connection);
            }

            // find best matching GPU
            let mut best_gpu: Option<Rc<GPU>> = None;
            let mut vk_graphics_queue_id = 0;
            let mut vk_present_queue_id = 0;
            for gpu in gpus {
                if let Some((g,p)) = self.find_gpu_window_queue_families(gpu,xcb_window) {
                    best_gpu = Some(Rc::clone(gpu));
                    vk_graphics_queue_id = g;
                    vk_present_queue_id = p;
                    break;
                }    
            }
        
            // destroy window
            unsafe {
                xcb_unmap_window(self.xcb_connection,xcb_window as u32);
                xcb_destroy_window(self.xcb_connection,xcb_window as u32);
            }

            // make sure a GPU was found
            if let None = best_gpu {
#[cfg(feature="debug_output")]
                println!("Unable to find suitable GPU.");
                return Vec::new();
            }

            (best_gpu.unwrap(),vk_graphics_queue_id,vk_present_queue_id)
        };

        vec![Rc::new(Screen {
            system: Rc::clone(self),
            gpu: best_gpu,
            name: ":0.0".to_string(),
            graphics_queue_id: vk_graphics_queue_id,
            present_queue_id: vk_present_queue_id,
            xcb_screen: xcb_screen,
        })]
    }
}
