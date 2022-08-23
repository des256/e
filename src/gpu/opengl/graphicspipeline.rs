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

        // TODO

        Some(GraphicsPipeline {
            system: &self,
        })
    }
}
