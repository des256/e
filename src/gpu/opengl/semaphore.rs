use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct Semaphore<'system> {
    pub system: &'system System,
}

impl System {

    /// Create a semaphore.
    pub fn create_semaphore(&self) -> Option<Semaphore> {

        // TODO
        Some(Semaphore {
            system: &self,
        })
    }
}
