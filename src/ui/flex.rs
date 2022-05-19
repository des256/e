use {
    crate::*,
    std::{
        sync::Arc,
    },
};

pub struct Flex {
    pub children: Vec<Arc<dyn Widget>>,
    pub main_axis_alignment: MainAxisAlignment,
    pub cross_axis_alignment: CrossAxisAlignment,
    pub direction: Axis,
}
