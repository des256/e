use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
        rc::Rc,
    },
    sys_sys::*,
};

pub struct GraphicsPipeline {
    pub session: Rc<Session>,
    pub(crate) vk_graphics_pipeline: VkPipeline,
}

impl Session {

    pub fn create_graphics_pipeline(self: &Rc<Self>,pipeline_layout: &Rc<PipelineLayout>,render_pass: &Rc<RenderPass>,vertex_shader: &Rc<Shader>,fragment_shader: &Rc<Shader>) -> Option<Rc<GraphicsPipeline>> {

        let create_info = VkGraphicsPipelineCreateInfo {
            sType: VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            stageCount: 2,
            pStages: [
                VkPipelineShaderStageCreateInfo {
                    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_VERTEX_BIT,
                    module: vertex_shader.vk_shader_module,
                    pName: b"main\0".as_ptr() as *const i8,
                    pSpecializationInfo: null_mut(),
                },
                VkPipelineShaderStageCreateInfo {
                    sType: VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    pNext: null_mut(),
                    flags: 0,
                    stage: VK_SHADER_STAGE_FRAGMENT_BIT,
                    module: fragment_shader.vk_shader_module,
                    pName: b"main\0".as_ptr() as *const i8,
                    pSpecializationInfo: null_mut(),
                }
            ].as_ptr(),
            pVertexInputState: &VkPipelineVertexInputStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                vertexBindingDescriptionCount: 0,
                pVertexBindingDescriptions: null_mut(),
                vertexAttributeDescriptionCount: 0,
                pVertexAttributeDescriptions: null_mut(),
            },
            pInputAssemblyState: &VkPipelineInputAssemblyStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                topology: VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                primitiveRestartEnable: VK_FALSE,
            },
            pTessellationState: &VkPipelineTessellationStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                patchControlPoints: 1,
            },
            pViewportState: &VkPipelineViewportStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                viewportCount: 1,
                pViewports: &VkViewport {
                    x: 0.0,
                    y: 0.0,
                    width: 640.0,
                    height: 480.0,
                    minDepth: 0.0,
                    maxDepth: 1.0,
                },
                scissorCount: 1,
                pScissors: &VkRect2D {
                    offset: VkOffset2D {
                        x: 0,
                        y: 0,
                    },
                    extent: VkExtent2D {
                        width: 640,
                        height: 480,
                    },
                },
            },
            pRasterizationState: &VkPipelineRasterizationStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                depthClampEnable: VK_FALSE,
                rasterizerDiscardEnable: VK_FALSE,
                polygonMode: VK_POLYGON_MODE_FILL,
                cullMode: VK_CULL_MODE_BACK_BIT,
                frontFace: VK_FRONT_FACE_CLOCKWISE,
                depthBiasEnable: VK_FALSE,
                depthBiasConstantFactor: 0.0,
                depthBiasClamp: 0.0,
                depthBiasSlopeFactor: 0.0,
                lineWidth: 1.0,
            },
            pMultisampleState: &VkPipelineMultisampleStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                rasterizationSamples: VK_SAMPLE_COUNT_1_BIT,
                sampleShadingEnable: VK_FALSE,
                minSampleShading: 1.0,
                pSampleMask: null_mut(),
                alphaToCoverageEnable: VK_FALSE,
                alphaToOneEnable: VK_FALSE,
            },
            pDepthStencilState: null_mut(),
            pColorBlendState: &VkPipelineColorBlendStateCreateInfo {
                sType: VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                pNext: null_mut(),
                flags: 0,
                logicOpEnable: VK_FALSE,
                logicOp: VK_LOGIC_OP_COPY,
                attachmentCount: 1,
                pAttachments: &VkPipelineColorBlendAttachmentState {
                    blendEnable: VK_FALSE,
                    srcColorBlendFactor: VK_BLEND_FACTOR_ONE,
                    dstColorBlendFactor: VK_BLEND_FACTOR_ZERO,
                    colorBlendOp: VK_BLEND_OP_ADD,
                    srcAlphaBlendFactor: VK_BLEND_FACTOR_ONE,
                    dstAlphaBlendFactor: VK_BLEND_FACTOR_ZERO,
                    alphaBlendOp: VK_BLEND_OP_ADD,
                    colorWriteMask: VK_COLOR_COMPONENT_R_BIT |
                        VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT |
                        VK_COLOR_COMPONENT_A_BIT,
                },
                blendConstants: [0.0,0.0,0.0,0.0],
            },
            pDynamicState: null_mut(),
            layout: pipeline_layout.vk_pipeline_layout,
            renderPass: render_pass.vk_render_pass,
            subpass: 0,
            basePipelineHandle: null_mut(),
            basePipelineIndex: -1,
        };
        let mut vk_graphics_pipeline = MaybeUninit::uninit();
        match unsafe { vkCreateGraphicsPipelines(self.vk_device,null_mut(),1,&create_info,null_mut(),vk_graphics_pipeline.as_mut_ptr()) } {
            VK_SUCCESS => { },
            code => {
#[cfg(feature="debug_output")]
                println!("Unable to create Vulkan graphics pipeline (error {})",code);
                return None;
            },
        }        

        Some(Rc::new(GraphicsPipeline {
            session: Rc::clone(self),
            vk_graphics_pipeline: unsafe { vk_graphics_pipeline.assume_init() },
        }))
    }
}

impl Drop for GraphicsPipeline {

    fn drop(&mut self) {
        unsafe { vkDestroyPipeline(self.session.vk_device,self.vk_graphics_pipeline,null_mut()) };
    }
}
