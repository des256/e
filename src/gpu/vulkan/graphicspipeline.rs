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

impl BaseType {
    pub fn format(&self) -> sys::VkFormat {
        match self {
            BaseType::U8 => sys::VK_FORMAT_R8_UINT,
            BaseType::U16 => sys::VK_FORMAT_R16_UINT,
            BaseType::U32 => sys::VK_FORMAT_R32_UINT,
            BaseType::U64 => sys::VK_FORMAT_R64_UINT,
            BaseType::I8 => sys::VK_FORMAT_R8_SINT,
            BaseType::I16 => sys::VK_FORMAT_R16_SINT,
            BaseType::I32 => sys::VK_FORMAT_R32_SINT,
            BaseType::I64 => sys::VK_FORMAT_R64_SINT,
            BaseType::F16 => sys::VK_FORMAT_R16_SFLOAT,
            BaseType::F32 => sys::VK_FORMAT_R32_SFLOAT,
            BaseType::F64 => sys::VK_FORMAT_R64_SFLOAT,
            BaseType::Vec2U8 => sys::VK_FORMAT_R8G8_UINT,
            BaseType::Vec2U16 => sys::VK_FORMAT_R16G16_UINT,
            BaseType::Vec2U32 => sys::VK_FORMAT_R32G32_UINT,
            BaseType::Vec2U64 => sys::VK_FORMAT_R64G64_UINT,
            BaseType::Vec2I8 => sys::VK_FORMAT_R8G8_SINT,
            BaseType::Vec2I16 => sys::VK_FORMAT_R16G16_SINT,
            BaseType::Vec2I32 => sys::VK_FORMAT_R32G32_SINT,
            BaseType::Vec2I64 => sys::VK_FORMAT_R64G64_SINT,
            BaseType::Vec2F16 => sys::VK_FORMAT_R16G16_SFLOAT,
            BaseType::Vec2F32 => sys::VK_FORMAT_R32G32_SFLOAT,
            BaseType::Vec2F64 => sys::VK_FORMAT_R64G64_SFLOAT,
            BaseType::Vec3U8 => sys::VK_FORMAT_R8G8B8_UINT,
            BaseType::Vec3U16 => sys::VK_FORMAT_R16G16B16_UINT,
            BaseType::Vec3U32 => sys::VK_FORMAT_R32G32B32_UINT,
            BaseType::Vec3U64 => sys::VK_FORMAT_R64G64B64_UINT,
            BaseType::Vec3I8 => sys::VK_FORMAT_R8G8B8_SINT,
            BaseType::Vec3I16 => sys::VK_FORMAT_R16G16B16_SINT,
            BaseType::Vec3I32 => sys::VK_FORMAT_R32G32B32_SINT,
            BaseType::Vec3I64 => sys::VK_FORMAT_R64G64B64_SINT,
            BaseType::Vec3F16 => sys::VK_FORMAT_R16G16B16_SFLOAT,
            BaseType::Vec3F32 => sys::VK_FORMAT_R32G32B32_SFLOAT,
            BaseType::Vec3F64 => sys::VK_FORMAT_R64G64B64_SFLOAT,
            BaseType::Vec4U8 => sys::VK_FORMAT_R8G8B8A8_UINT,
            BaseType::Vec4U16 => sys::VK_FORMAT_R16G16B16A16_UINT,
            BaseType::Vec4U32 => sys::VK_FORMAT_R32G32B32A32_UINT,
            BaseType::Vec4U64 => sys::VK_FORMAT_R64G64B64A64_UINT,
            BaseType::Vec4I8 => sys::VK_FORMAT_R8G8B8A8_SINT,
            BaseType::Vec4I16 => sys::VK_FORMAT_R16G16B16A16_SINT,
            BaseType::Vec4I32 => sys::VK_FORMAT_R32G32B32A32_SINT,
            BaseType::Vec4I64 => sys::VK_FORMAT_R64G64B64A64_SINT,
            BaseType::Vec4F16 => sys::VK_FORMAT_R16G16B16A16_SFLOAT,
            BaseType::Vec4F32 => sys::VK_FORMAT_R32G32B32A32_SFLOAT,
            BaseType::Vec4F64 => sys::VK_FORMAT_R64G64B64A64_SFLOAT,
            BaseType::ColorU8 => sys::VK_FORMAT_R8G8B8A8_UNORM,
            BaseType::ColorU16 => sys::VK_FORMAT_R16G16B16A16_UNORM,
            BaseType::ColorF16 => sys::VK_FORMAT_R16G16B16A16_SFLOAT,
            BaseType::ColorF32 => sys::VK_FORMAT_R32G32B32A32_SFLOAT,
            BaseType::ColorF64 => sys::VK_FORMAT_R64G64B64A64_SFLOAT,
        }
    }
}

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
        vertex_shader: &ShaderModule,
        fragment_shader: &ShaderModule,
        topology: PrimitiveTopology,
        restart: PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: DepthClamp,
        primitive_discard: PrimitiveDiscard,
        polygon_mode: PolygonMode,
        cull_mode: CullMode,
        depth_bias: DepthBias,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: SampleShading,
        alpha_to_coverage: AlphaToCoverage,
        alpha_to_one: AlphaToOne,
        depth_test: DepthTest,
        depth_write: DepthWrite,
        stencil_test: StencilTest,
        logic_op: LogicOp,
        blend: Blend,
        write_mask: u8,
        blend_constant: Vec4<f32>,
    ) -> Option<GraphicsPipeline> {

        let vertex_base_types = T::get_types();

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
        let mut location = 0u32;
        let mut stride = 0u32;
        let mut attribute_descriptions: Vec<sys::VkVertexInputAttributeDescription> = Vec::new();
        for base_type in vertex_base_types {
            attribute_descriptions.push(sys::VkVertexInputAttributeDescription {
                location,
                binding: 0,
                format: base_type.format(),
                offset: stride,
            });
            location += 1;
            stride += base_type.size() as u32;
        }
        let input = sys::VkPipelineVertexInputStateCreateInfo {
            // TODO: build entirely from T
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            vertexBindingDescriptionCount: 1,
            pVertexBindingDescriptions: [  // binding 0 is a T::SIZE for each vertex
                sys::VkVertexInputBindingDescription {
                    binding: 0,
                    stride,
                    inputRate: sys::VK_VERTEX_INPUT_RATE_VERTEX,  // or RATE_INSTANCE
                },
            ].as_ptr(),
            vertexAttributeDescriptionCount: attribute_descriptions.len() as u32,
            pVertexAttributeDescriptions: attribute_descriptions.as_ptr(),
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
        let depth_clamp = match depth_clamp {
            DepthClamp::Disabled => sys::VK_FALSE,
            DepthClamp::Enabled => sys::VK_TRUE,
        };
        let primitive_discard = match primitive_discard {
            PrimitiveDiscard::Disabled => sys::VK_FALSE,
            PrimitiveDiscard::Enabled => sys::VK_TRUE,
        };
        let polygon_mode = match polygon_mode {
            PolygonMode::Point => sys::VK_POLYGON_MODE_POINT,
            PolygonMode::Line => sys::VK_POLYGON_MODE_LINE,
            PolygonMode::Fill => sys::VK_POLYGON_MODE_FILL,
        };
        let (cull_mode,front_face) = match cull_mode {
            CullMode::None => (
                sys::VK_CULL_MODE_NONE,
                sys::VK_FRONT_FACE_CLOCKWISE
            ),
            CullMode::Front(front_face) => (
                sys::VK_CULL_MODE_FRONT_BIT,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
            CullMode::Back(front_face) => (
                sys::VK_CULL_MODE_BACK_BIT,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
            CullMode::FrontAndBack(front_face) => (
                sys::VK_CULL_MODE_FRONT_AND_BACK,
                match front_face {
                    FrontFace::CounterClockwise => sys::VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    FrontFace::Clockwise => sys::VK_FRONT_FACE_CLOCKWISE,
                },
            ),
        };
        let (depth_bias_enable,depth_bias_constant_factor,depth_bias_clamp,depth_bias_slope_factor) = match depth_bias {
            DepthBias::Disabled => (sys::VK_FALSE,0.0,0.0,0.0),
            DepthBias::Enabled(constant_factor,clamp,slope_factor) => (sys::VK_TRUE,constant_factor,clamp,slope_factor),
        };
        let rasterization = sys::VkPipelineRasterizationStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthClampEnable: depth_clamp,
            rasterizerDiscardEnable: primitive_discard,
            polygonMode: polygon_mode,
            cullMode: cull_mode,
            frontFace: front_face,
            depthBiasEnable: depth_bias_enable,
            depthBiasConstantFactor: depth_bias_constant_factor,
            depthBiasClamp: depth_bias_clamp,
            depthBiasSlopeFactor: depth_bias_slope_factor,
            lineWidth: line_width,
        };
        let (sample_shading,min_sample_shading) = match sample_shading {
            SampleShading::Disabled => (sys::VK_FALSE,0.0),
            SampleShading::Enabled(min_sample_shading) => (sys::VK_TRUE,min_sample_shading),
        };
        let alpha_to_coverage = match alpha_to_coverage {
            AlphaToCoverage::Disabled => sys::VK_FALSE,
            AlphaToCoverage::Enabled => sys::VK_TRUE,
        };
        let alpha_to_one = match alpha_to_one {
            AlphaToOne::Disabled => sys::VK_FALSE,
            AlphaToOne::Enabled => sys::VK_TRUE,
        };
        let multisample = sys::VkPipelineMultisampleStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            rasterizationSamples: rasterization_samples as u32,
            sampleShadingEnable: sample_shading,
            minSampleShading: min_sample_shading,
            pSampleMask: null_mut(),
            alphaToCoverageEnable: alpha_to_coverage,
            alphaToOneEnable: alpha_to_one,
        };
        let (depth_test,depth_compare,(depth_bounds,min_depth_bounds,max_depth_bounds)) = match depth_test {
            DepthTest::Disabled => (sys::VK_FALSE,sys::VK_COMPARE_OP_ALWAYS,(sys::VK_FALSE,0.0,0.0)),
            DepthTest::Enabled(depth_compare,depth_bounds) => (
                sys::VK_TRUE,
                match depth_compare {
                    CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                    CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                    CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                    CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                    CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                    CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                    CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                    CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,
                },
                match depth_bounds {
                    DepthBounds::Disabled => (sys::VK_FALSE,0.0,0.0),
                    DepthBounds::Enabled(min,max) => (sys::VK_TRUE,min,max),
                },
            ),
        };
        let depth_write = match depth_write {
            DepthWrite::Disabled => sys::VK_FALSE,
            DepthWrite::Enabled => sys::VK_TRUE,
        };
        let (
            stencil_test,
            (front_fail,front_pass,front_depth_fail,front_compare,front_compare_mask,front_write_mask,front_reference),
            (back_fail,back_pass,back_depth_fail,back_compare,back_compare_mask,back_write_mask,back_reference),
        ) = match stencil_test {
            StencilTest::Disabled => (
                sys::VK_FALSE,
                (sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_COMPARE_OP_ALWAYS,0,0,0),
                (sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_STENCIL_OP_KEEP,sys::VK_COMPARE_OP_ALWAYS,0,0,0),
            ),
            StencilTest::Enabled(
                (front_fail,front_pass,front_depth_fail,front_compare,front_compare_mask,front_write_mask,front_reference),
                (back_fail,back_pass,back_depth_fail,back_compare,back_compare_mask,back_write_mask,back_reference),
            ) => (
                sys::VK_TRUE,
                (
                    match front_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_pass {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_depth_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match front_compare {
                        CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                        CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                        CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                        CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                        CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                        CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                        CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                        CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,    
                    },
                    front_compare_mask,
                    front_write_mask,
                    front_reference,
                ),
                (
                    match back_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_pass {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_depth_fail {
                        StencilOp::Keep => sys::VK_STENCIL_OP_KEEP,
                        StencilOp::Zero => sys::VK_STENCIL_OP_ZERO,
                        StencilOp::Replace => sys::VK_STENCIL_OP_REPLACE,
                        StencilOp::IncClamp => sys::VK_STENCIL_OP_INCREMENT_AND_CLAMP,
                        StencilOp::DecClamp => sys::VK_STENCIL_OP_DECREMENT_AND_CLAMP,
                        StencilOp::Invert => sys::VK_STENCIL_OP_INVERT,
                        StencilOp::IncWrap => sys::VK_STENCIL_OP_INCREMENT_AND_WRAP,
                        StencilOp::DecWrap => sys::VK_STENCIL_OP_DECREMENT_AND_WRAP,    
                    },
                    match back_compare {
                        CompareOp::Never => sys::VK_COMPARE_OP_NEVER,
                        CompareOp::Less => sys::VK_COMPARE_OP_LESS,
                        CompareOp::Equal => sys::VK_COMPARE_OP_EQUAL,
                        CompareOp::LessOrEqual => sys::VK_COMPARE_OP_LESS_OR_EQUAL,
                        CompareOp::Greater => sys::VK_COMPARE_OP_GREATER,
                        CompareOp::NotEqual => sys::VK_COMPARE_OP_NOT_EQUAL,
                        CompareOp::GreaterOrEqual => sys::VK_COMPARE_OP_GREATER_OR_EQUAL,
                        CompareOp::Always => sys::VK_COMPARE_OP_ALWAYS,    
                    },
                    back_compare_mask,
                    back_write_mask,
                    back_reference,
                ),
            )
        };
        let depth_stencil = sys::VkPipelineDepthStencilStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            depthTestEnable: depth_test,
            depthWriteEnable: depth_write,
            depthCompareOp: depth_compare,
            depthBoundsTestEnable: depth_bounds,
            stencilTestEnable: stencil_test,
            front: sys::VkStencilOpState {
                failOp: front_fail,
                passOp: front_pass,
                depthFailOp: front_depth_fail,
                compareOp: front_compare,
                compareMask: front_compare_mask,
                writeMask: front_write_mask,
                reference: front_reference,
            },
            back: sys::VkStencilOpState {
                failOp: back_fail,
                passOp: back_pass,
                depthFailOp: back_depth_fail,
                compareOp: back_compare,
                compareMask: back_compare_mask,
                writeMask: back_write_mask,
                reference: back_reference,
            },
            minDepthBounds: min_depth_bounds,
            maxDepthBounds: max_depth_bounds,
        };
        let (logic_op_enable,logic_op) = match logic_op {
            LogicOp::Disabled => (sys::VK_FALSE,sys::VK_LOGIC_OP_COPY),
            LogicOp::Clear => (sys::VK_TRUE,sys::VK_LOGIC_OP_CLEAR),
            LogicOp::And => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND),
            LogicOp::AndReverse => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND_REVERSE),
            LogicOp::Copy => (sys::VK_TRUE,sys::VK_LOGIC_OP_COPY),
            LogicOp::AndInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_AND_INVERTED),
            LogicOp::NoOp => (sys::VK_TRUE,sys::VK_LOGIC_OP_NO_OP),
            LogicOp::Xor => (sys::VK_TRUE,sys::VK_LOGIC_OP_XOR),
            LogicOp::Or => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR),
            LogicOp::Nor => (sys::VK_TRUE,sys::VK_LOGIC_OP_NOR),
            LogicOp::Equivalent => (sys::VK_TRUE,sys::VK_LOGIC_OP_EQUIVALENT),
            LogicOp::Invert => (sys::VK_TRUE,sys::VK_LOGIC_OP_INVERT),
            LogicOp::OrReverse => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR_REVERSE),
            LogicOp::CopyInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_COPY_INVERTED),
            LogicOp::OrInverted => (sys::VK_TRUE,sys::VK_LOGIC_OP_OR_INVERTED),
            LogicOp::Nand => (sys::VK_TRUE,sys::VK_LOGIC_OP_NAND),
            LogicOp::Set => (sys::VK_TRUE,sys::VK_LOGIC_OP_SET),
        };
        let (
            blend,
            (color_op,src_color,dst_color),
            (alpha_op,src_alpha,dst_alpha),
        ) = match blend {
            Blend::Disabled => (
                sys::VK_FALSE,
                (sys::VK_BLEND_OP_ADD,sys::VK_BLEND_FACTOR_ONE,sys::VK_BLEND_FACTOR_ZERO),
                (sys::VK_BLEND_OP_ADD,sys::VK_BLEND_FACTOR_ONE,sys::VK_BLEND_FACTOR_ZERO),
            ),
            Blend::Enabled((color_op,src_color,dst_color),(alpha_op,src_alpha,dst_alpha)) => (
                sys::VK_TRUE,
                (
                    match color_op {
                        BlendOp::Add => sys::VK_BLEND_OP_ADD,
                        BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                        BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                        BlendOp::Min => sys::VK_BLEND_OP_MIN,
                        BlendOp::Max => sys::VK_BLEND_OP_MAX,
                    },
                    match src_color {
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
                    match dst_color {
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
                ),
                (
                    match alpha_op {
                        BlendOp::Add => sys::VK_BLEND_OP_ADD,
                        BlendOp::Subtract => sys::VK_BLEND_OP_SUBTRACT,
                        BlendOp::ReverseSubtract => sys::VK_BLEND_OP_REVERSE_SUBTRACT,
                        BlendOp::Min => sys::VK_BLEND_OP_MIN,
                        BlendOp::Max => sys::VK_BLEND_OP_MAX,
                    },
                    match src_alpha {
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
                    match dst_alpha {
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
                ),
            ),
        };
        let blend = sys::VkPipelineColorBlendStateCreateInfo {
            sType: sys::VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            pNext: null_mut(),
            flags: 0,
            logicOpEnable: logic_op_enable,
            logicOp: logic_op,
            attachmentCount: 1,
            pAttachments: &sys::VkPipelineColorBlendAttachmentState {
                blendEnable: blend,
                srcColorBlendFactor: src_color,
                dstColorBlendFactor: dst_color,
                colorBlendOp: color_op,
                srcAlphaBlendFactor: src_alpha,
                dstAlphaBlendFactor: dst_alpha,
                alphaBlendOp: alpha_op,
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
