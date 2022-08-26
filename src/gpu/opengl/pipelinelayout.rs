// Pipeline Layout
// descriptor set layouts: which type of information in the pipeline can be accessed by which shader stage
// push-constant ranges: which push-constants can be accessed by which shader stage

use {
    crate::*,
    std::rc::Rc,
};

pub struct PipelineLayout {
    pub(crate) system: Rc<System>,
}
