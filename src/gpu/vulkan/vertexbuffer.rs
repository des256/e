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
pub struct VertexBuffer {
    pub system: Rc<System>,
    pub(crate) vk_buffer: sys::VkBuffer,
    pub(crate) vk_device_memory: sys::VkDeviceMemory,
}

impl VertexBuffer {

    pub fn new<T: Vertex>(system: &Rc<System>,vertices: &Vec<T>) -> Result<VertexBuffer,String> {

        // obtain vertex info
        let vertex_base_fields = T::get_fields();
        let mut vertex_stride = 0usize;
        for (_,ty) in &vertex_base_fields {
            vertex_stride += ty.size();
        }

        // create vertex buffer
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
        match unsafe { sys::vkCreateBuffer(system.gpu_system.vk_device, &info, null_mut(), vk_buffer.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to create vertex buffer ({})",vk_code_to_string(code))),
        }
        let vk_buffer = unsafe { vk_buffer.assume_init() };

        // allocate shared memory
        let info = sys::VkMemoryAllocateInfo {
            sType: sys::VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            pNext: null_mut(),
            allocationSize: (vertices.len() * vertex_stride) as u64,
            memoryTypeIndex: system.gpu_system.shared_index as u32,
        };
        let mut vk_device_memory = MaybeUninit::<sys::VkDeviceMemory>::uninit();
        match unsafe { sys::vkAllocateMemory(system.gpu_system.vk_device,&info,null_mut(),vk_device_memory.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to allocate memory ({})",vk_code_to_string(code))),
        }
        let vk_device_memory = unsafe { vk_device_memory.assume_init() };

        // map memory
        let mut data_ptr = MaybeUninit::<*mut c_void>::uninit();
        match unsafe { sys::vkMapMemory(
            system.gpu_system.vk_device,
            vk_device_memory,
            0,
            sys::VK_WHOLE_SIZE as u64,
            0,
            data_ptr.as_mut_ptr(),
        ) } {
            sys::VK_SUCCESS => { },
            code => return Err(format!("Unable to map mempry ({})",vk_code_to_string(code))),
        }
        let data_ptr = unsafe { data_ptr.assume_init() } as *mut T;

        // copy from the input vertices into data
        unsafe { copy_nonoverlapping(vertices.as_ptr(),data_ptr,vertices.len()) };

        // and unmap the memory again
        unsafe { sys::vkUnmapMemory(system.gpu_system.vk_device,vk_device_memory) };

        // bind to vertex buffer
        match unsafe { sys::vkBindBufferMemory(system.gpu_system.vk_device,vk_buffer,vk_device_memory,0) } {
            sys::VK_SUCCESS => Ok(VertexBuffer {
                system: Rc::clone(system),
                vk_buffer,
                vk_device_memory,
            }),
            code => Err(format!("Unable to bind memory to vertex buffer ({})",vk_code_to_string(code))),
        }
    }
}

impl Drop for VertexBuffer {
    fn drop(&mut self) {
        unsafe {
            sys::vkDestroyBuffer(self.system.gpu_system.vk_device,self.vk_buffer,null_mut());
            sys::vkFreeMemory(self.system.gpu_system.vk_device,self.vk_device_memory,null_mut());
        }
    }
}