use {
    e::*,
    gpu_macros::*,
    std::{
        result::Result,
        rc::Rc,
        io::*,
        fs::*,
    },
};

#[derive(Vertex)]
struct MyVertex {
    pub pos: Vec2<f32>,
}

fn main() -> Result<(),String> {
    let system = Rc::new(e::System::open()?);
    let mut r = Rect {
        o: Vec2 { x: 10i32,y: 10i32, },
        s: Vec2 { x: 800i32,y: 600i32, },
    };
    let frame_window = Rc::new(e::Window::new_frame(&system,r,"Single Triangle",)?);
    let count = frame_window.get_framebuffer_count();
    let mut command_buffers: Vec<CommandBuffer> = Vec::new();
    for _ in 0..count {
        command_buffers.push(CommandBuffer::new(&system)?);
    }
    let mut f = File::open("test_triangle_vert.spv").expect("Unable to open vertex shader");
    let mut code = Vec::<u8>::new();
    f.read_to_end(&mut code).expect("Unable to read vertex shader");
    let vertex_shader = Rc::new(VertexShader::new(&system,&code).expect("Unable to create vertex shader"));
    let mut f = File::open("test_triangle_frag.spv").expect("Unable to open fragment shader");
    let mut code = Vec::<u8>::new();
    f.read_to_end(&mut code).expect("Unable to read fragment shader");
    let fragment_shader = Rc::new(FragmentShader::new(&system,&code).expect("Unable to create fragment shader"));
    let mut vertices = Vec::<MyVertex>::new();
    vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: -0.5, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: -0.5, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: 0.5, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: 0.5, }, });
    let vertex_buffer = Rc::new(VertexBuffer::new(&system,&vertices)?);
    let mut indices = Vec::<u32>::new();
    indices.push(0);
    indices.push(1);
    indices.push(2);
    indices.push(0);
    indices.push(2);
    indices.push(3);
    let index_buffer = Rc::new(IndexBuffer::new(&system,&indices)?);
    let pipeline_layout = Rc::new(PipelineLayout::new(&system)?);
    let graphics_pipeline = Rc::new(GraphicsPipeline::new::<MyVertex>(&system,
        &frame_window,
        &pipeline_layout,
        &vertex_shader,
        &fragment_shader,
        PrimitiveTopology::Triangles,
        PrimitiveRestart::Disabled,
        1,
        DepthClamp::Disabled,
        PrimitiveDiscard::Disabled,
        PolygonMode::Fill,
        CullMode::None,
        DepthBias::Disabled,
        1.0,
        1,
        SampleShading::Disabled,
        AlphaToCoverage::Disabled,
        AlphaToOne::Disabled,
        DepthTest::Disabled,
        false,
        StencilTest::Disabled,
        LogicOp::Disabled,
        Blend::Disabled,
        (true,true,true,true),
        Vec4::<f32>::ZERO,
    )?);
    let image_available = Semaphore::new(&system)?;
    let render_finished = Semaphore::new(&system)?;

    let mut close_clicked = false;
    while !close_clicked {
        system.wait();
        system.flush().into_iter().for_each(|(_,event)| {
            dprintln!("event {}",event);
            if let Event::Configure(new_r) = event {
                if r.s != new_r.s {
                    frame_window.update_swapchain(&new_r);
                }
                r = new_r;
            }
            if let Event::Close = event {
                close_clicked = true;
            }
        });
        let index = frame_window.acquire(&image_available)?;
        let cb = &mut command_buffers[index];
        cb.begin()?;
        cb.bind_graphics_pipeline(&graphics_pipeline);
        cb.bind_vertex_buffer(&vertex_buffer);
        cb.bind_index_buffer(&index_buffer);
        cb.begin_render_pass(&frame_window,index,Rect { o: Vec2::<i32>::ZERO,s: r.s, });
        cb.set_viewport(
            Rect {
                o: Vec2::<f32>::ZERO,
                s: Vec2::<f32> {
                    x: r.s.x as f32,
                    y: r.s.y as f32,
                },
            },
            0.0,
            1.0, 
        );
        cb.set_scissor(
            Rect {
                o: Vec2::<f32>::ZERO,
                s: Vec2::<f32> {
                    x: r.s.x as f32,
                    y: r.s.y as f32,
                },
            });
        cb.draw_indexed(6,1,0,0,0);
        cb.end_render_pass();
        cb.end();
        system.submit_command_buffer(cb,&image_available,&render_finished)?;
        frame_window.present(index,&render_finished);
    }
    Ok(())
}
