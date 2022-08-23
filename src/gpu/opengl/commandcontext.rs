use {
    crate::*,
    std::ptr::null_mut,
};

pub struct CommandContext<'system,'window> {
    pub window: &'window Window<'system>,
    pub index: usize,
}

impl<'system> Window<'system> {

    /// Acquire command context for next available frame in the window's swap chain. Also indicate to trigger signal_semaphore when this frame is ready to be drawn to.
    pub fn acquire_next(&self,signal_semaphore: &Semaphore) -> CommandContext {
        let mut index = 0u32;
        // TODO
        CommandContext {
            window: &self,
            index: index as usize,
        }
    }

    /// Present work from a context to the window as soon as wait_semaphore gets triggered.
    pub fn present(&self,context: CommandContext,wait_semaphore: &Semaphore) {
        let image_index = context.index as u32;
        // TODO
    }
}
