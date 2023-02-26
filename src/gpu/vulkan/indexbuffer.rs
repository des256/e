use {
    crate::*,
    std::{
        ptr::{
            null_mut,
            copy_nonoverlapping,
        },
        rc::Rc,
        mem::MaybeUninit,
        ffi::c_void,
    },
};

#[derive(Debug)]
pub struct IndexBuffer {
    pub system: Rc<System>,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_memory: sys::VkDeviceMemory,
}

impl IndexBuffer {
    pub fn new<T>(system: &Rc<System>,indices: &Vec<T>) -> Result<IndexBuffer,String> {

        // create index buffer
        let info = sys::VkBufferCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            size: (indices.len() * 4) as u64,
            usage: sys::VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            sharingMode: sys::VK_SHARING_MODE_EXCLUSIVE,
            queueFamilyIndexCount: 0,
            pQueueFamilyIndices: null_mut(),
        };
        let mut vk_buffer = MaybeUninit::uninit();
        match unsafe { sys::vkCreateBuffer(system.gpu_system.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to create index buffer ({})",vk_code_to_string(code))),
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // allocate shared memory
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (indices.len() * 4) as u64,
            memoryTypeIndex: system.gpu_system.shared_index as u32,
        };
        let mut vk_memory = MaybeUninit::<sys::VkDeviceMemory>::uninit();
        match unsafe { sys::vkAllocateMemory(system.gpu_system.vk_device,&info,null_mut(),vk_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to allocate memory ({})",vk_code_to_string(code))),
        }
        let vk_memory = unsafe { vk_memory.assume_init() };

        // map memory
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            system.gpu_system.vk_device,
            vk_memory,
            0,
            sys::VK_WHOLE_SIZE as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to map memory ({})",vk_code_to_string(code))),
        }
        let data_ptr = unsafe { data_ptr.assume_init() } as *mut T;

        // copy from the input vertices into data
        unsafe { copy_nonoverlapping(indices.as_ptr(),data_ptr,indices.len()) };

        // and unmap the memory again
        unsafe { sys::vkUnmapMemory(system.gpu_system.vk_device,vk_memory) };

        // bind to vertex buffer
        match unsafe { sys::vkBindBufferMemory(system.gpu_system.vk_device,vk_buffer,vk_memory,0) } {
            sys::VK_SUCCESS => Ok(IndexBuffer {
                system: Rc::clone(system),
                vk_buffer,
                vk_memory,
            }),
            code => Err(format!("Unable to bind memory to index buffer ({})",vk_code_to_string(code))),
        }
    }
}

impl Drop for IndexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroyBuffer(self.system.gpu_system.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.system.gpu_system.vk_device,self.vk_memory,null_mut());
        }
    }
}