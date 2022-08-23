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

pub struct IndexBuffer<'system> {
    pub system: &'system System,
}

impl System {

    /// create a vertex buffer.
    pub fn create_index_buffer<T>(&self,indices: &Vec<T>) -> Option<IndexBuffer> {

        // TODO
        Some(IndexBuffer {
            system: &self,
        })
    }
}
