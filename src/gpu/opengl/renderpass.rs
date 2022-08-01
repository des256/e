use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct RenderPass<'system,'gpu,'screen> {
    pub screen: &'screen Screen<'system,'gpu>
}

impl <'system,'gpu> Screen<'system,'gpu> {

    pub fn create_render_pass(&self) -> Option<RenderPass> {
        Some(RenderPass { screen: &self, })
    }
}
