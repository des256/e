use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        ffi::c_void,
    },
};

pub struct VertexBuffer<'system> {
    pub system: &'system System,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_memory: sys::VkDeviceMemory,
}

impl System {

    /// create a vertex buffer.
    pub fn create_vertex_buffer<T: Vertex>(&self,vertices: &Vec<T>) -> Option<VertexBuffer> {

        // create vertex buffer
        println!("creating vertex buffer");
        let info = sys::VkBufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            size: (vertices.len() * T::SIZE) as u64,
            usage: sys::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
        };
        let mut vk_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateBuffer(self.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create vertex buffer (error {})",code);
                return None;
            }
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // get buffer memory requirements
        println!("getting memory requirements");
        let mut vk_memory_requirements = MaybeUninit::uninit();
        unsafe { sys::vkGetBufferMemoryRequirements(
            self.vk_device,
            vk_buffer,
            vk_memory_requirements.as_mut_ptr()
        ) };
        let vk_memory_requirements = unsafe { vk_memory_requirements.assume_init() };
        println!("memory requirements: size {}, alignment {}, type {:08X}",vk_memory_requirements.size,vk_memory_requirements.alignment,vk_memory_requirements.memoryTypeBits);

        // allocate shared memory
        println!("allocating memory");
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (vertices.len() * T::SIZE) as u64,
            memoryTypeIndex: vk_memory_requirements.memoryTypeBits | sys::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | sys::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        };
        let mut vk_memory = MaybeUninit::uninit();
        match unsafe { sys::vkAllocateMemory(self.vk_device,&info,null_mut(),vk_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to allocate memory (error {})",code);
                return None;
            }
        }
        let vk_memory = unsafe { vk_memory.assume_init() };

        println!("mapping memory");
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            self.vk_device,
            vk_memory,
            0,
            (vertices.len() * T::SIZE) as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to map memory (error {})",code);
                return None;
            }
        }
        let data_ptr = unsafe { data_ptr.assume_init() };
        println!("mapped pointer = {:?}",data_ptr);

        // TODO: copy from the input vertices into data

        println!("unmapping memory");
        unsafe { sys::vkUnmapMemory(self.vk_device,vk_memory) };
        println!("binding memory buffer to vertex buffer");
        match unsafe { sys::vkBindBufferMemory(self.vk_device,vk_buffer,vk_memory,0) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to bind memory to vertex buffer (error {})",code);
                return None;
            }
        }

        Some(VertexBuffer {
            system: &self,
            vk_buffer: vk_buffer,
            vk_memory: vk_memory,
        })
    }
}

impl<'system> Drop for VertexBuffer<'system> {
    fn drop(&mut self) {
        // TODO: drop buffer and memory
    }
}