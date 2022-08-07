use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct RenderPass {
    pub(crate) owned: bool,
    pub(crate) vk_device: sys::VkDevice,
    pub(crate) vk_renderpass: sys::VkRenderPass,
}

impl RenderPass {

    /// Create a graphics pipeline for this render pass.
    pub fn create_graphics_pipeline(&self,pipeline_layout: &PipelineLayout,vertex_shader: &Shader,fragment_shader: &Shader) -> Option<GraphicsPipeline> {

        let create_info = sys::VkGraphicsPipelineCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            stageCount: 2,
            pStages: [
                sys::VkPipelineShaderStageCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    stage: sys::VK_SHADER_STAGE_VERTEX_BIT,
                    module: vertex_shader.vk_shader_module,
                    pName: b"main\0".as_ptr() as *const i8,
                    pSpecializationInfo: null_mut(),
                },
                sys::VkPipelineShaderStageCreateInfo {
                    sType: sys::VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    stage: sys::VK_SHADER_STAGE_FRAGMENT_BIT,
                    module: fragment_shader.vk_shader_module,
                    pName: b"main\0".as_ptr() as *const i8,
                    pSpecializationInfo: null_mut(),
                }
            ].as_ptr(),
            pVertexInputState: &sys::VkPipelineVertexInputStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                vertexBindingDescriptionCount: 0,
                pVertexBindingDescriptions: null_mut(),
                vertexAttributeDescriptionCount: 0,
                pVertexAttributeDescriptions: null_mut(),
            },
            pInputAssemblyState: &sys::VkPipelineInputAssemblyStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                topology: sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                primitiveRestartEnable: sys::VK_FALSE,
            },
            pTessellationState: &sys::VkPipelineTessellationStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                patchControlPoints: 1,
            },
            pViewportState: &sys::VkPipelineViewportStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                viewportCount: 1,
                pViewports: null_mut(),
                scissorCount: 1,
                pScissors: null_mut(),
            },
            pRasterizationState: &sys::VkPipelineRasterizationStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                depthClampEnable: sys::VK_FALSE,
                rasterizerDiscardEnable: sys::VK_FALSE,
                polygonMode: sys::VK_POLYGON_MODE_FILL,
                cullMode: sys::VK_CULL_MODE_BACK_BIT,
                frontFace: sys::VK_FRONT_FACE_CLOCKWISE,
                depthBiasEnable: sys::VK_FALSE,
                depthBiasConstantFactor: 0.0,
                depthBiasClamp: 0.0,
                depthBiasSlopeFactor: 0.0,
                lineWidth: 1.0,
            },
            pMultisampleState: &sys::VkPipelineMultisampleStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                rasterizationSamples: sys::VK_SAMPLE_COUNT_1_BIT,
                sampleShadingEnable: sys::VK_FALSE,
                minSampleShading: 1.0,
                pSampleMask: null_mut(),
                alphaToCoverageEnable: sys::VK_FALSE,
                alphaToOneEnable: sys::VK_FALSE,
            },
            pDepthStencilState: null_mut(),
            pColorBlendState: &sys::VkPipelineColorBlendStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                logicOpEnable: sys::VK_FALSE,
                logicOp: sys::VK_LOGIC_OP_COPY,
                attachmentCount: 1,
                pAttachments: &sys::VkPipelineColorBlendAttachmentState {
                    blendEnable: sys::VK_FALSE,
                    srcColorBlendFactor: sys::VK_BLEND_FACTOR_ONE,
                    dstColorBlendFactor: sys::VK_BLEND_FACTOR_ZERO,
                    colorBlendOp: sys::VK_BLEND_OP_ADD,
                    srcAlphaBlendFactor: sys::VK_BLEND_FACTOR_ONE,
                    dstAlphaBlendFactor: sys::VK_BLEND_FACTOR_ZERO,
                    alphaBlendOp: sys::VK_BLEND_OP_ADD,
                    colorWriteMask: sys::VK_COLOR_COMPONENT_R_BIT |
                        sys::VK_COLOR_COMPONENT_G_BIT |
                        sys::VK_COLOR_COMPONENT_B_BIT |
                        sys::VK_COLOR_COMPONENT_A_BIT,
                },
                blendConstants: [0.0,0.0,0.0,0.0],
            },
            pDynamicState: &sys::VkPipelineDynamicStateCreateInfo {
                sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                pDynamicStates: [
                    sys::VK_DYNAMIC_STATE_VIEWPORT,
                    sys::VK_DYNAMIC_STATE_SCISSOR,
                ].as_ptr(),
                dynamicStateCount: 2,
            },
            layout: pipeline_layout.vk_pipeline_layout,
            renderPass: self.vk_renderpass,
            subpass: 0,
            basePipelineHandle: null_mut(),
            basePipelineIndex: -1,
        };
        let mut vk_graphics_pipeline = MaybeUninit::uninit();
        match unsafe { sys::vkCreateGraphicsPipelines(self.vk_device,null_mut(),1,&create_info,null_mut(),vk_graphics_pipeline.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create Vulkan graphics pipeline (error {})",code);
                return None;
            },
        }

        Some(GraphicsPipeline {
            vk_device: self.vk_device,
            vk_graphics_pipeline: unsafe { vk_graphics_pipeline.assume_init() },
        })
    }
}

impl Drop for RenderPass {

    fn drop(&mut self) {
        if self.owned {
            unsafe { sys::vkDestroyRenderPass(self.vk_device,self.vk_renderpass,null_mut()) };
        }
    }
}
