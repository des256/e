// Graphics Pipeline
// shader stages: vertex, tesselation control, tesselation evaluation, geometry, fragment, compute stages
// vertex input: from the vertex buffer memory, which data of what type describes which attributes to the vertex
// input assembly: primitive topology and restart enable
// tesselation: patch control points
// rasterization: depth clamp enable, rasterizer discard enable, polygon mode, cull mode, front face, depth bias enable, depth bias constant, clamp, slope factor, line width
// multisample: samples, sample shading enable, min sample shading, sample mask, alpha to coverage, alpha to one
// depth/stencil: depth test enable, depth write enable, depth compare op, depth bounds test enable, stencil test enable, front, back, min depth bounds, max depth bounds
// color blend: logic op enable, logic op, [blend enable, src/dst color/alpha factors, color/alpha blend op, color write mask] for each attachment

use {
    crate::*,
    std::{
        ptr::null_mut,
        mem::MaybeUninit,
    },
};

pub struct GraphicsPipeline<'system> {
    pub(crate) system: &'system System,
    pub(crate) vk_graphics_pipeline: sys::VkPipeline,
}

impl System {

    /// Create a graphics pipeline.
    pub fn create_graphics_pipeline<T: Vertex>(
        &self,
        window: &Window,
        pipeline_layout: &PipelineLayout,
        vertex_shader: &Shader,
        fragment_shader: &Shader,
        topology: PrimitiveTopology,
        restart: PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: DepthClamp,
        primitive_discard: PrimitiveDiscard,
        polygon_mode: PolygonMode,
        cull_mode: CullMode,
        front_face: FrontFace,
        depth_bias: DepthBias,
        depth_bias_constant: f32,
        depth_bias_clamp: f32,
        depth_bias_slope: f32,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: SampleShading,
        min_sample_shading: f32,
        alpha_to_coverage: AlphaToCoverage,
        alpha_to_one: AlphaToOne,
        depth_test: DepthTest,
        depth_write: DepthWrite,
        depth_compare: CompareOp,
        depth_bounds: DepthBounds,
        stencil_test: StencilTest,
        front_fail: StencilOp,
        front_pass: StencilOp,
        front_depth_fail: StencilOp,
        front_compare: CompareOp,
        front_compare_mask: u32,
        front_write_mask: u32,
        front_reference: u32,
        back_fail: StencilOp,
        back_pass: StencilOp,
        back_depth_fail: StencilOp,
        back_compare: CompareOp,
        back_compare_mask: u32,
        back_write_mask: u32,
        back_reference: u32,
        min_depth_bounds: f32,
        max_depth_bounds: f32,
        logic_op: LogicOp,
        blend: Blend,
        src_color: BlendFactor,
        dst_color: BlendFactor,
        color_op: BlendOp,
        src_alpha: BlendFactor,
        dst_alpha: BlendFactor,
        alpha_op: BlendOp,
        write_mask: u8,
        blend_constant: Vec4<f32>,
    ) -> Option<GraphicsPipeline> {
        let shaders = [
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
        ];
        let input = sys::VkPipelineVertexInputStateCreateInfo {
            // TODO: build entirely from T
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: [
                sys::VkVertexInputBindingDescription {
                    binding: 0,
                    stride: T::SIZE as u32,
                    inputRate: sys::VK_VERTEX_INPUT_RATE_VERTEX,
                },
            ].as_ptr(),
            vertexAttributeDescriptionCount: 1,
            pVertexAttributeDescriptions: [
                sys::VkVertexInputAttributeDescription {
                    binding: 0,
                    location: 0,
                    format: sys::VK_FORMAT_R32G32_SFLOAT,
                    offset: 0,
                },
            ].as_ptr(),
        };
        let assembly = sys::VkPipelineInputAssemblyStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            topology: match topology {
                PrimitiveTopology::Points => sys::VK_PRIMITIVE_TOPOLOGY_POINT_LIST,
                PrimitiveTopology::Lines => sys::VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                PrimitiveTopology::LineStrip => sys::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP,
                PrimitiveTopology::Triangles => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                PrimitiveTopology::TriangleStrip => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
                PrimitiveTopology::TriangleFan => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN,
                PrimitiveTopology::LinesAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY,
                PrimitiveTopology::LineStripAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY,
                PrimitiveTopology::TrianglesAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY,
                PrimitiveTopology::TriangleStripAdjacency => sys::VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY,
                PrimitiveTopology::Patches => sys::VK_PRIMITIVE_TOPOLOGY_PATCH_LIST,
            },
            primitiveRestartEnable: match restart {
                PrimitiveRestart::Disabled => sys::VK_FALSE,
                PrimitiveRestart::Enabled => sys::VK_TRUE,
            },
        };
        let tesselation = sys::VkPipelineTessellationStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            patchControlPoints: patch_control_points as u32,
        };
        let viewport = sys::VkPipelineViewportStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            viewportCount: 1,
            pViewports: null_mut(),
            scissorCount: 1,
            pScissors: null_mut(),
        };
        let rasterization = sys::VkPipelineRasterizationStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthClampEnable: match depth_clamp {
                DepthClamp::Disabled => sys::VK_FALSE,
                DepthClamp::Enabled => sys::VK_TRUE,
            },
            rasterizerDiscardEnable: match primitive_discard {
                PrimitiveDiscard::Disabled => sys::VK_FALSE,
                PrimitiveDiscard::Enabled => sys::VK_TRUE,
            },
            polygonMode: match polygon_mode {
                PolygonMode::Point => sys::VK_POLYGON_MODE_POINT,
                PolygonMode::Line => sys::VK_POLYGON_MODE_LINE,
                PolygonMode::Fill => sys::VK_POLYGON_MODE_FILL,
            },
            cullMode: match cull_mode {
                CullMode::None => sys::VK_CULL_MODE_NONE,
                CullMode::Front => sys::VK_CULL_MODE_FRONT_BIT,
                CullMode::Back => sys::VK_CULL_MODE_BACK_BIT,
                CullMode::FrontAndBack => sys::VK_CULL_MODE_FRONT_AND_BACK,
            },
            frontFace: match front_face {
                FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
            },
            depthBiasEnable: match depth_bias {
                DepthBias::Disabled => sys::VK_FALSE,
                DepthBias::Enabled => sys::VK_TRUE,
            },
            depthBiasConstantFactor: depth_bias_constant,
            depthBiasClamp: depth_bias_clamp,
            depthBiasSlopeFactor: depth_bias_slope,
            lineWidth: line_width,
        };
        let multisample = sys::VkPipelineMultisampleStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            rasterizationSamples: rasterization_samples as u32,
            sampleShadingEnable: match sample_shading {
                SampleShading::Disabled => sys::VK_FALSE,
                SampleShading::Enabled => sys::VK_TRUE,
            },
            minSampleShading: min_sample_shading,
            pSampleMask: null_mut(),
            alphaToCoverageEnable: match alpha_to_coverage {
                AlphaToCoverage::Disabled => sys::VK_FALSE,
                AlphaToCoverage::Enabled => sys::VK_TRUE,
            },
            alphaToOneEnable: match alpha_to_one {
                AlphaToOne::Disabled => sys::VK_FALSE,
                AlphaToOne::Enabled => sys::VK_TRUE,
            },
        };
        let depth_stencil = sys::VkPipelineDepthStencilStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthTestEnable: match depth_test {
                DepthTest::Disabled => sys::VK_FALSE,
                DepthTest::Enabled => sys::VK_TRUE,
            },
            depthWriteEnable: match depth_write {
                DepthWrite::Disabled => sys::VK_FALSE,
                DepthWrite::Enabled => sys::VK_TRUE,
            },
            depthCompareOp: match depth_compare {
                CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,
            },
            depthBoundsTestEnable: match depth_bounds {
                DepthBounds::Disabled => sys::VK_FALSE,
                DepthBounds::Enabled => sys::VK_TRUE,
            },
            stencilTestEnable: match stencil_test {
                StencilTest::Disabled => sys::VK_FALSE,
                StencilTest::Enabled => sys::VK_TRUE,
            },
            front: sys::VkStencilOpState {
                failOp: match front_fail {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                passOp: match front_pass {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                depthFailOp: match front_depth_fail {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                compareOp: match front_compare {
                    CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                    CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                    CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                    CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                    CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                    CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                    CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                    CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,
                },
                compareMask: front_compare_mask,
                writeMask: front_write_mask,
                reference: front_reference,
            },
            back: sys::VkStencilOpState {
                failOp: match back_fail {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                passOp: match back_pass {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                depthFailOp: match back_depth_fail {
                    StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                    StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                    StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                    StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                    StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                    StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                    StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                    StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,
                },
                compareOp: match back_compare {
                    CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                    CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                    CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                    CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                    CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                    CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                    CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                    CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,
                },
                compareMask: back_compare_mask,
                writeMask: back_write_mask,
                reference: back_reference,
            },
            minDepthBounds: min_depth_bounds,
            maxDepthBounds: max_depth_bounds,
        };
        let blend = sys::VkPipelineColorBlendStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            logicOpEnable: match logic_op {
                LogicOp::Disabled => sys::VK_FALSE,
                _ => sys::VK_TRUE,
            },
            logicOp: match logic_op {
                LogicOp::Disabled => sys::VK_LOGIC_OP_COPY,
                LogicOp::Clear => sys::VK_LOGIC_OP_CLEAR,
                LogicOp::And => sys::VK_LOGIC_OP_AND,
                LogicOp::AndReverse => sys::VK_LOGIC_OP_AND_REVERSE,
                LogicOp::Copy => sys::VK_LOGIC_OP_COPY,
                LogicOp::AndInverted => sys::VK_LOGIC_OP_AND_INVERTED,
                LogicOp::NoOp => sys::VK_LOGIC_OP_NO_OP,
                LogicOp::Xor => sys::VK_LOGIC_OP_XOR,
                LogicOp::Or => sys::VK_LOGIC_OP_OR,
                LogicOp::Nor => sys::VK_LOGIC_OP_NOR,
                LogicOp::Equivalent => sys::VK_LOGIC_OP_EQUIVALENT,
                LogicOp::Invert => sys::VK_LOGIC_OP_INVERT,
                LogicOp::OrReverse => sys::VK_LOGIC_OP_OR_REVERSE,
                LogicOp::CopyInverted => sys::VK_LOGIC_OP_COPY_INVERTED,
                LogicOp::OrInverted => sys::VK_LOGIC_OP_OR_INVERTED,
                LogicOp::Nand => sys::VK_LOGIC_OP_NAND,
                LogicOp::Set => sys::VK_LOGIC_OP_SET,
            },
            attachmentCount: 1,
            pAttachments: &sys::VkPipelineColorBlendAttachmentState {
                blendEnable: match blend {
                    Blend::Disabled => sys::VK_FALSE,
                    Blend::Enabled => sys::VK_TRUE,
                },
                srcColorBlendFactor: match src_color {
                    BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                    BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                    BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                },
                dstColorBlendFactor: match dst_color {
                    BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                    BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                    BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                },
                colorBlendOp: match color_op {
                    BlendOp::Add => sys::VK_BLEND_OP_ADD,
                    BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                    BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                    BlendOp::Min => sys::VK_BLEND_OP_MIN,
                    BlendOp::Max => sys::VK_BLEND_OP_MAX,
                },
                srcAlphaBlendFactor: match src_alpha {
                    BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                    BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                    BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                },
                dstAlphaBlendFactor: match dst_alpha {
                    BlendFactor::Zero => sys::VK_BLEND_FACTOR_ZERO,
                    BlendFactor::One => sys::VK_BLEND_FACTOR_ONE,
                    BlendFactor::SrcColor => sys::VK_BLEND_FACTOR_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::VK_BLEND_FACTOR_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::VK_BLEND_FACTOR_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::VK_BLEND_FACTOR_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::VK_BLEND_FACTOR_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::VK_BLEND_FACTOR_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::VK_BLEND_FACTOR_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::VK_BLEND_FACTOR_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::VK_BLEND_FACTOR_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA,
                },
                alphaBlendOp: match alpha_op {
                    BlendOp::Add => sys::VK_BLEND_OP_ADD,
                    BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                    BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                    BlendOp::Min => sys::VK_BLEND_OP_MIN,
                    BlendOp::Max => sys::VK_BLEND_OP_MAX,
                },
                colorWriteMask: write_mask as u32,
            },
            blendConstants: [blend_constant.x,blend_constant.y,blend_constant.z,blend_constant.w],
        };
        let dynamic = sys::VkPipelineDynamicStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            pDynamicStates: [
                sys::VK_DYNAMIC_STATE_VIEWPORT,
                sys::VK_DYNAMIC_STATE_SCISSOR,
            ].as_ptr(),
            dynamicStateCount: 2,
        };
        let create_info = sys::VkGraphicsPipelineCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            stageCount: 2,
            pStages: shaders.as_ptr(),
            pVertexInputState: &input,
            pInputAssemblyState: &assembly,
            pTessellationState: &tesselation,
            pViewportState: &viewport,
            pRasterizationState: &rasterization,
            pMultisampleState: &multisample,
            pDepthStencilState: &depth_stencil,
            pColorBlendState: &blend,
            pDynamicState: &dynamic,
            layout: pipeline_layout.vk_pipeline_layout,
            renderPass: window.vk_render_pass,
            subpass: 0,
            basePipelineHandle: null_mut(),
            basePipelineIndex: -1,
        };
        let mut vk_graphics_pipeline = MaybeUninit::uninit();
        match unsafe { sys::vkCreateGraphicsPipelines(self.vk_device,null_mut(),1,&create_info,null_mut(),vk_graphics_pipeline.as_mut_ptr()) } {
            sys::VK_SUCCESS => { },
            code => {
                println!("unable to create graphics pipeline (error {})",code);
                return None;
            },
        }

        Some(GraphicsPipeline {
            system: &self,
            vk_graphics_pipeline: unsafe { vk_graphics_pipeline.assume_init() },
        })
    }
}

impl<'system> Drop for GraphicsPipeline<'system> {

    fn drop(&mut self) {
        unsafe { sys::vkDestroyPipeline(self.system.vk_device,self.vk_graphics_pipeline,null_mut()) };
    }

    /*
    Cannot call vkDestroyPipeline on VkPipeline 0x120000000012[] that is
    currently in use by a command buffer. The Vulkan spec states: All submitted
    commands that refer to pipeline must have completed execution
    (https://vulkan.lunarg.com/doc/view/1.2.162.1~rc2/linux/1.2-extensions/vkspec.html#VUID-vkDestroyPipeline-pipeline-00765)
    */
}
