use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::{
        rc::Rc,
        cell::Cell,
    },
};

#[derive(Debug)]
pub struct Surface {
    pub gpu: Rc<Gpu>,
    pub window: Rc<Window>,
    pub visible_index: Cell<usize>,
}

impl Surface {

}

impl gpu::Surface for Surface {

    fn set_rect(&mut self,_r: Rect<i32>) -> Result<(),String> {
        Ok(())
    }

    fn get_swapchain_count(&self) -> usize {
        2
    }

    fn acquire(&self) -> Result<usize,String> {
        let index = self.visible_index.get();
        self.visible_index.set(1 - index);
        Ok(index)
    }

    fn present(&self,_index: usize) -> Result<(),String> {
        unsafe {
            checkgl!(sys::glFlush());
#[cfg(system="linux")]
            sys::glXSwapBuffers(self.gpu.system.xdisplay,self.window.xcb_window as u64);
        }
        Ok(())
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
    }
}
