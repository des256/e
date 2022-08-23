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
}

impl System {

    /// create a vertex buffer.
    pub fn create_vertex_buffer<T: Vertex>(&self,vertices: &Vec<T>) -> Option<VertexBuffer> {

        // TODO

        Some(VertexBuffer {
            system: &self,
        })
    }
}
