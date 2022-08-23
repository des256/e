use {
    crate::*,
    std::{
        ptr::{
            null_mut,
            copy_nonoverlapping,
        },
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

        // obtain vertex info
        let vertex_base_types = T::get_types();
        let mut vertex_stride = 0usize;
        for base_type in &vertex_base_types {
            vertex_stride += base_type.size();
        }

        // create vertex buffer
        println!("creating vertex buffer");
        let info = sys::VkBufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            size: (vertices.len() * vertex_stride) as u64,
            usage: sys::VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            sharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
        };
        let mut vk_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateBuffer(self.gpu.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create vertex buffer (error {})",code);
                return None;
            }
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // allocate shared memory
        println!("allocating memory");
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (vertices.len() * vertex_stride) as u64,
            memoryTypeIndex: self.gpu.shared_index as u32,
        };
        let mut vk_memory = MaybeUninit::<sys::VkDeviceMemory>::uninit();
        match unsafe { sys::vkAllocateMemory(self.gpu.vk_device,&info,null_mut(),vk_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to allocate memory (error {})",code);
                return None;
            }
        }
        let vk_memory = unsafe { vk_memory.assume_init() };

        // map memory
        println!("mapping memory");
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            self.gpu.vk_device,
            vk_memory,
            0,
            sys::VK_WHOLE_SIZE as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to map memory (error {})",code);
                return None;
            }
        }
        let data_ptr = unsafe { data_ptr.assume_init() } as *mut T;
        println!("mapped pointer = {:?}",data_ptr);

        // copy from the input vertices into data
        unsafe { copy_nonoverlapping(vertices.as_ptr(),data_ptr,vertices.len()) };

        // and unmap the memory again
        println!("unmapping memory");
        unsafe { sys::vkUnmapMemory(self.gpu.vk_device,vk_memory) };

        // bind to vertex buffer
        println!("binding memory buffer to vertex buffer");
        match unsafe { sys::vkBindBufferMemory(self.gpu.vk_device,vk_buffer,vk_memory,0) } {
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
        unsafe {
            sys::vkDestroyBuffer(self.system.gpu.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.system.gpu.vk_device,self.vk_memory,null_mut());
        }
    }

    /*
    Cannot free VkBuffer 0xd000000000d[] that is in use by a command buffer.
    The Vulkan spec states: All submitted commands that refer to buffer,
    either directly or via a VkBufferView, must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroyBuffer-buffer-00922)
    */

    /*
    Cannot call vkFreeMemory on VkDeviceMemory 0xe000000000e[] that is
    currently in use by a command buffer. The Vulkan spec states: All
    submitted commands that refer to memory (via images or buffers) must have
    completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkFreeMemory-memory-00677)
    */
}