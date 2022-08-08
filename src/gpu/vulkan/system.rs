use {
    crate::*,
};

pub struct SystemVulkan {
    pub instance: VkInstance,
    pub physical_device: VkPhysicalDevice,
    pub device: VkDevice,
    pub queue: VkQueue,
    pub command_pool: VkCommandPool,
}

impl System {

    pub fn open_system_vulkan() -> Option<SystemVulkan> {

        // create instance and get first physical device
        let instance = VkInstance::create()?;
        let physical_devices = instance.enumerate_physical_devices()?;
        let physical_device = physical_devices[0];

        // DEBUG: show the name in debug build
#[cfg(build="debug")]
        dprintln!("physical device: {}",physical_device.get_name());
            
        // get supported queue families
        let queue_families = physical_device.get_queue_families()?;

        // DEBUG: display the number of queues and capabilities
#[cfg(build="debug")]
        for i in 0..queue_families.len() {
            let mut capabilities = String::new();
            if queue_families[i].queueFlags & sys::VK_QUEUE_GRAPHICS_BIT != 0 {
                capabilities.push_str("graphics ");
            }
            if queue_families[i].queueFlags & sys::VK_QUEUE_TRANSFER_BIT != 0 {
                capabilities.push_str("transfer ");
            }
            if queue_families[i].queueFlags & sys::VK_QUEUE_COMPUTE_BIT != 0 {
                capabilities.push_str("compute ");
            }
            if queue_families[i].queueFlags & sys::VK_QUEUE_SPARSE_BINDING_BIT != 0 {
                capabilities.push_str("sparse ");
            }
            dprintln!("    {}: {} queues, capable of: {}",i,queue_families[i].queueCount,capabilities);
        }

        // assume the first queue family is the one we want for all queues
        let queue_family = queue_families[0];
        let mask = sys::VK_QUEUE_GRAPHICS_BIT | sys::VK_QUEUE_TRANSFER_BIT | sys::VK_QUEUE_COMPUTE_BIT;
        if (queue_family.queueFlags & mask) != mask {
            println!("queue family 0 of the GPU does not support graphics, transfer and compute operations");
            return None;
        }

        // assume that presentation is done on the same family as graphics and create logical device with one queue of queue family 0
        let device = physical_device.create_device()?;

        // obtain the queue from queue family 0
        let queue = device.get_queue();

        // create command pool for queue family 0
        let command_pool = device.create_command_pool()?;

        Some(SystemVulkan {
            instance,
            physical_device,
            device,
            queue,
            command_pool,
        })
    }
}