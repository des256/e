use {
    e::*,
    std::{
        rc::Rc,
        time::Instant,
        collections::HashMap,
    },
    gpu_macros::*,
};

#[derive(Vertex)]
struct MyVertex {
    pub pos: Vec2<f32>,
    pub color: Color<f32>,
}

#[vertex_shader]
mod my_vertex_shader {
    fn main(vertex: MyVertex) -> (Vec4<f32>,Color<f32>) {
        (
            Vec4::<f32> {
                x: vertex.pos.x,
                y: vertex.pos.y,
                z: 0.0,
                w: 1.0,
            },
            vertex.color,
        )
    }
}

#[fragment_shader]
mod my_fragment_shader {
    fn main(varying: Color<f32>) -> Color<f32> {
        varying
    }
}

fn main() {

    // initial rectangle
    let mut r = Rect { o: Vec2::<isize>::ZERO,s: Vec2::<usize> { x: 640,y: 480, }, };

    let system = open_system().expect("unable to access system");

    // create frame window
    let mut window = system.create_frame_window(Rect { o: Vec2::<isize> { x: 100,y: 100, },s: r.s, },"test triangle").expect("unable to create frame window");

    // get number of associated framebuffers
    let count = window.get_framebuffer_count();

    // create command buffers, one for each framebuffer
    let mut command_buffers: Vec<Rc<CommandBuffer>> = Vec::new();
    for _ in 0..count {
        if let Some(command_buffer) = system.create_command_buffer() {
            command_buffers.push(command_buffer);
        }
        else {
            panic!("cannot create command buffer");
        }
    }

    // create the vertices
    let mut vertices = Vec::<MyVertex>::new();
    vertices.push(MyVertex { pos: f32xy { x: -0.5,y: -0.5, }, color: Color::<f32>::from(0xFFFF0000), });
    vertices.push(MyVertex { pos: f32xy { x: 0.5,y: -0.5, }, color: Color::<f32>::from(0xFF00FF00), });
    vertices.push(MyVertex { pos: f32xy { x: 0.5,y: 0.5, }, color: Color::<f32>::from(0xFF0000FF), });
    vertices.push(MyVertex { pos: f32xy { x: -0.5,y: 0.5, }, color: Color::<f32>::from(0xFFFFFF00), });
    let vertex_buffer = system.create_vertex_buffer(&vertices).expect("unable to create vertex buffer");

    // create the indices
    let mut indices = Vec::<u32>::new();
    indices.push(0);
    indices.push(1);
    indices.push(2);
    indices.push(0);
    indices.push(2);
    indices.push(3);
    let index_buffer = system.create_index_buffer(&indices).expect("unable to create index buffer");

    // create pipeline layout
    let pipeline_layout = system.create_pipeline_layout().expect("unable to create pipeline layout");

    // read vertex shader
    //let mut f = File::open("test_triangle_vert.spv").expect("unable to open vertex shader");
    //let mut code = Vec::<u8>::new();
    //f.read_to_end(&mut code).expect("unable to read vertex shader");

    let code = my_vertex_shader::code().expect("unable to compile vertex shader");

    // create vertex shader
    let vertex_shader = system.create_vertex_shader(&code).expect("unable to create vertex shader");

    // read fragment shader
    //let mut f = File::open("test_triangle_frag.spv").expect("unable to open fragment shader");
    //let mut code = Vec::<u8>::new();
    //f.read_to_end(&mut code).expect("unable to read fragment shader");

    let code = my_fragment_shader::code().expect("unable to compile fragment shader");

    // create fragment shader
    let fragment_shader = system.create_fragment_shader(&code).expect("unable to create fragment shader");
    
    // create graphics pipeline with this layout
    let graphics_pipeline = system.create_graphics_pipeline::<MyVertex>(
        &window,
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
        Color::<f32>::ZERO,
    ).expect("Unable to create graphics pipeline.");
    
    // create the semaphores
    let image_available = system.create_semaphore().expect("unable to create image available semaphore");
    let render_finished = system.create_semaphore().expect("unable to create render finished semaphore");

    // and go
    let mut running = true;
    let mut time = Instant::now();
    while running {

        let duration = time.elapsed();
        let fps = if duration.as_micros() != 0 { 1000000 / duration.as_micros() } else { 0 };
        println!("{} FPS",fps);
        time = Instant::now();

        // wait for UX events to happen
        println!("waiting for UX events...");
        system.wait();
        let events = system.flush();
        println!("done waiting.");

        // process the UX events
        for (id,event) in events {

            dprintln!("event: {} ({})",event,id);

            // if the event was meant for the window
            if id == window.id() {

                // if the window changed size, rebuild the framebuffer resources
                if let Event::Configure(new_r) = event {
                    if r.s != new_r.s {
                        window.update_swapchain_resources(new_r);
                    }
                    r = new_r;
                }

                // if the user closes the window, stop running
                if let Event::Close = event {
                    running = false;
                }
            }
        }

        // obtain index for the current accessible framebuffer
        let index = window.acquire_next(&image_available);

        // start associated command buffer
        let cb = &mut command_buffers[index];
        cb.begin();

        // switch to the shader pipeline (select the shaders and blending, etc.)
        cb.bind_pipeline(&graphics_pipeline);

        // bind the vertexbuffer
        cb.bind_vertex_buffer(&vertex_buffer);

        // bind the indexbuffer
        cb.bind_index_buffer(&index_buffer);

        // render the triangle onto the window's framebuffer
        cb.begin_render_pass(&window,index,Rect { o: Vec2::<isize>::ZERO,s: r.s, });

        cb.set_viewport(Hyper {
            o: Vec3::<f32>::ZERO,
            s: Vec3 { x: r.s.x as f32,y: r.s.y as f32,z: 1.0, },
        });

        cb.set_scissor(Rect { o: Vec2::<isize>::ZERO,s: r.s, });

        cb.draw_indexed(6,1,0,0,0);
        
        cb.end_render_pass();

        // end the command buffer
        cb.end();

        // and submit the work
        system.submit(cb,&image_available,&render_finished);

        // and present the frame
        window.present(index,&render_finished);
    }
}
