use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct ShaderModule<'system> {
    pub system: &'system System,
}

impl System {

    /// Create a shader.
    pub fn create_shader_module(&self,code: &[u8]) -> Option<ShaderModule> {

        // TODO
        Some(ShaderModule {
            system: &self,
        })
    }    
}
