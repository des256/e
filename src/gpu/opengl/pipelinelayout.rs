// Pipeline Layout
// descriptor set layouts: which type of information in the pipeline can be accessed by which shader stage
// push-constant ranges: which push-constants can be accessed by which shader stage

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct PipelineLayout<'system> {
    pub system: &'system System,
}

impl<'system> System {

    /// Create a pipeline layout.
    pub fn create_pipeline_layout(&self) -> Option<PipelineLayout> {

        // TODO

        Some(PipelineLayout {
            system: &self,
        })
    }
}
