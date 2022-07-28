use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub type QueueFamilyID = u32;

pub struct QueueFamily {
    pub id: QueueFamilyID,
    pub graphics: bool,
    pub compute: bool,
    pub transfer: bool,
    pub sparse: bool,
}

pub struct Gpu {
    pub system: Rc<System>,
    pub name: String,
    pub queue_families: Vec<(QueueFamily,usize)>,
    pub(crate) vk_physical_device: VkPhysicalDevice,
}

impl System {

    pub fn find_gpus(self: &Rc<Self>) -> Vec<Rc<Gpu>> {

        let vk_physical_devices = {
            let mut count = 0u32;
            unsafe { vkEnumeratePhysicalDevices(self.vk_instance,&mut count,null_mut()) };
            if count == 0 {
#[cfg(feature="debug_output")]
                println!("Unable to find Vulkan physical devices.");
                return Vec::new();
            }
            let mut devices = vec![null_mut() as VkPhysicalDevice; count as usize];
            unsafe { vkEnumeratePhysicalDevices(self.vk_instance,&mut count,devices.as_mut_ptr()) };
            devices
        };

        let mut gpus = Vec::<Rc<Gpu>>::new();
        for vk_physical_device in &vk_physical_devices {

            // get GPU properties
            let mut properties = MaybeUninit::uninit();
            unsafe { vkGetPhysicalDeviceProperties(*vk_physical_device,properties.as_mut_ptr()) };
            let properties = unsafe { properties.assume_init() };
            let slice: &[u8] = unsafe { &*(&properties.deviceName as *const [i8] as *const [u8]) };
            let name = std::str::from_utf8(slice).unwrap();

            // get supported queue families
            let mut count = 0u32;
            let mut queue_families = Vec::<(QueueFamily,usize)>::new();
            unsafe { vkGetPhysicalDeviceQueueFamilyProperties(*vk_physical_device,&mut count,null_mut()) };
            if count > 0 {
                let mut vk_queue_families = vec![VkQueueFamilyProperties {
                    queueFlags: 0,
                    queueCount: 0,
                    timestampValidBits: 0,
                    minImageTransferGranularity: VkExtent3D {
                        width: 0,
                        height: 0,
                        depth: 0,
                    },
                }; count as usize];
                unsafe { vkGetPhysicalDeviceQueueFamilyProperties(*vk_physical_device,&mut count,vk_queue_families.as_mut_ptr()) };
                for i in 0..vk_queue_families.len() {
                    queue_families.push((QueueFamily {
                        id: i as QueueFamilyID,
                        graphics: (vk_queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0,
                        compute: (vk_queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0,
                        transfer: (vk_queue_families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) != 0,
                        sparse: (vk_queue_families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) != 0,
                    },vk_queue_families[i].queueCount as usize));
                }
            }

            gpus.push(Rc::new(Gpu {
                system: Rc::clone(self),
                name: name.to_string(),
                queue_families: queue_families,
                vk_physical_device: *vk_physical_device,
            }));
        }

        gpus
    }
}
