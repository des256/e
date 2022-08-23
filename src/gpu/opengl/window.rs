use {
    crate::*,
    std::cell::Cell,
};

pub struct WindowGpu {
    pub(crate) visible_index: Cell<usize>,
}

impl Window {

    /// Update swapchain resources after window resize.
    pub fn update_swapchain_resources(&mut self,r: Rect<isize,usize>) {
    }
        
    /// Return number of framebuffers in the swapchain.
    pub fn get_framebuffer_count(&self) -> usize {
        2
    }

    /// Acquire index of the next available framebuffer in the window's swap chain. Also indicate to trigger signal_semaphore when this frame is ready to be drawn to.
    pub fn acquire_next(&self,signal_semaphore: &Semaphore) -> usize {
        let index = self.gpu.visible_index.get();
        self.gpu.visible_index.set(1 - index);
        index
    }

    /// Present a newly created framebuffer in the swapchain as soon as wait_semaphore gets triggered.
    pub fn present(&self,index: usize,wait_semaphore: &Semaphore) {
        unsafe {
            sys::glFlush();
            sys::glXSwapBuffers(self.system.xdisplay,self.xcb_window as u64);
        }
    }

    /// Drop GPU-specific Window part.
    pub fn drop_gpu(&self) {
    }
}
