use {
    e::*,
    std::{
        fs::File,
        io::prelude::*,
        time::Instant,
    },
    vertexformat::*,
    shader::*,
};

#[derive(VertexFormat)]
struct TestVertex {
    pub pos: f32xy,
    pub color: f32rgba,
}

#[shader(vertex)]
mod my_vertex_shader {
    fn main(vertex: TestVertex) -> (f32xyzw,f32rgba) {
        (vertex.pos,vertex.color)
    }
}

#[shader(fragment)]
mod my_fragment_shader {
    fn main(color: f32rgba) -> f32rgba {
        color
    }
}

fn main() {

    // create frame window
    let system = open_system().expect("unable to access system");
    let mut r = i32r { o: i32xy::ZERO,s: u32xy { x: 640,y: 480, }, };

    let mut window = system.create_frame_window(i32r { o: i32xy { x: 100,y: 100, },s: r.s, },"test triangle").expect("unable to create frame window");

    // create the vertices
    let mut vertices = Vec::<TestVertex>::new();
    vertices.push(TestVertex { pos: f32xy { x: -0.5,y: -0.5, }, color: f32rgba::from(0xFFFF0000), });
    vertices.push(TestVertex { pos: f32xy { x: 0.5,y: -0.5, }, color: f32rgba::from(0xFF00FF00), });
    vertices.push(TestVertex { pos: f32xy { x: 0.5,y: 0.5, }, color: f32rgba::from(0xFF0000FF), });
    vertices.push(TestVertex { pos: f32xy { x: -0.5,y: 0.5, }, color: f32rgba::from(0xFFFFFF00), });
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

    // read vertex shader
    let mut f = File::open("test_triangle_vert.spv").expect("unable to open vertex shader");
    let mut b = Vec::<u8>::new();
    f.read_to_end(&mut b).expect("unable to read vertex shader");
    let vertex_shader = system.create_shader_module(&b).expect("unable to create vertex shader");

    // read fragment shader
    let mut f = File::open("test_triangle_frag.spv").expect("unable to open fragment shader");
    let mut b = Vec::<u8>::new();
    f.read_to_end(&mut b).expect("unable to read fragment shader");
    let fragment_shader = system.create_shader_module(&b).expect("unable to create fragment shader");

    // create pipeline layout
    let pipeline_layout = system.create_pipeline_layout().expect("unable to create pipeline layout");
    
    // create graphics pipeline with this layout
    // TODO: currenty doesn't work because of one of these settings
    // TODO: compress settings further, so when a feature is not used, the parameters don't appear in this list
    let graphics_pipeline = system.create_graphics_pipeline::<TestVertex>(
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
        DepthWrite::Disabled,
        StencilTest::Disabled,
        LogicOp::Disabled,
        Blend::Disabled,
        0x0F,
        f32xyzw::ZERO,
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

        // get all UX events that might have gathered
        let events = system.flush();

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

        // obtain context to the next available frame from the window and signal image_available
        let context = window.acquire_next(&image_available);

        // put GPU commands on the queue
        if let Some(buffer) = context.begin() {

            // set the viewport and scissor to whatever the current window rectangle is
            buffer.set_viewport(f32h {
                o: f32xyz::ZERO,
                s: f32xyz { x: r.s.x as f32,y: r.s.y as f32,z: 1.0, },
            });
            buffer.set_scissor(i32r { o: i32xy::ZERO,s: r.s, });

            // switch to the shader pipeline (select the shaders and blending, etc.)
            buffer.bind_pipeline(&graphics_pipeline);

            // bind the vertexbuffer
            buffer.bind_vertex_buffer(&vertex_buffer);

            // bind the indexbuffer
            buffer.bind_index_buffer(&index_buffer);

            // render the triangle using the window's render pass
            buffer.begin_render_pass(i32r { o: i32xy::ZERO,s: r.s, });
            buffer.draw_indexed(6,1,0,0,0);
            buffer.end_render_pass();

            // and finish the command buffer
            if !buffer.end_submit(&image_available,&render_finished) {
                println!("unable to end command buffer");
            }
        }
        else {
            println!("unable to begin command buffer");
        }

        // only when render_finished present the frame
        window.present(context,&render_finished);
    }
}
