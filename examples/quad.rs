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
    pub color: Vec3<f32>,
}

fn main() -> Result<(),String> {
    let system = Rc::new(e::System::open()?);
    let mut r = Rect {
        o: Vec2 { x: 10i32,y: 10i32, },
        s: Vec2 { x: 800i32,y: 600i32, },
    };
    let gpu = vulkan::Gpu::open()?;
    let frame_window = Rc::new(e::Window::new_frame(&system,r,"Single Quad",)?);
    let mut surface = gpu.create_surface(&frame_window,r)?;
    let count = surface.get_swapchain_count();
    let mut command_buffers: Vec<<vulkan::Gpu as Gpu>::CommandBuffer> = Vec::new();
    for _ in 0..count {
        command_buffers.push(gpu.create_command_buffer()?);
    }
    let mut f = File::open("assets/quad-vs.spv").expect("Unable to open vertex shader");
    let mut code = Vec::<u8>::new();
    f.read_to_end(&mut code).expect("Unable to read vertex shader");
    let vertex_shader = Rc::new(gpu.create_vertex_shader(&code)?);
    let mut f = File::open("assets/quad-fs.spv").expect("Unable to open fragment shader");
    let mut code = Vec::<u8>::new();
    f.read_to_end(&mut code).expect("Unable to read fragment shader");
    let fragment_shader = Rc::new(gpu.create_fragment_shader(&code)?);
    let mut vertices = Vec::<MyVertex>::new();
    vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: -0.5, }, color: Vec3::<f32> { x: 1.0,y: 1.0,z: 0.0, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: -0.5, }, color: Vec3::<f32> { x: 0.0,y: 1.0,z: 1.0, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: 0.5, }, color: Vec3::<f32> { x: 1.0,y: 0.0,z: 1.0, }, });
    vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: 0.5, }, color: Vec3::<f32> { x: 0.0,y: 1.0,z: 0.0, }, });
    let vertex_buffer = gpu.create_vertex_buffer(&vertices)?;
    let mut indices = Vec::<u32>::new();
    indices.push(0);
    indices.push(1);
    indices.push(2);
    indices.push(0);
    indices.push(2);
    indices.push(3);
    let index_buffer = gpu.create_index_buffer(&indices)?;
    let pipeline_layout = Rc::new(gpu.create_pipeline_layout()?);
    let graphics_pipeline = gpu.create_graphics_pipeline::<MyVertex>(
        &surface,
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
    )?;
    let mut close_clicked = false;
    while !close_clicked {
        system.wait();
        system.flush().into_iter().try_for_each(|(_,event)| {
            dprintln!("event {}",event);
            if let Event::Configure(new_r) = event {
                if new_r.s != r.s {
                    surface.set_rect(new_r)?;
                }
                r = new_r;
            }
            if let Event::Close = event {
                close_clicked = true;
            }
            Ok::<(),String>(())
        })?;
        let index = surface.acquire()?;
        let cb = &mut command_buffers[index];
        cb.begin()?;
        cb.bind_graphics_pipeline(&graphics_pipeline);
        cb.bind_vertex_buffer(&vertex_buffer);
        cb.bind_index_buffer(&index_buffer);
        cb.begin_render_pass(&surface,index,Rect { o: Vec2::<i32>::ZERO,s: r.s, });
        cb.set_viewport(Rect { o: Vec2::<i32>::ZERO,s: r.s, }, 0.0, 1.0, );
        cb.set_scissor(Rect { o: Vec2::<i32>::ZERO,s: r.s, });
        cb.draw_indexed(6,1,0,0,0);
        cb.end_render_pass();
        cb.end();
        gpu.submit_command_buffer(cb)?;
        surface.present(index)?;
    }
    Ok(())
}
