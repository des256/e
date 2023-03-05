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

struct Environment<T: Gpu> {
    gpu: Rc<T>,
    r: Rect<i32>,
    surface: <T::CommandBuffer as CommandBuffer>::Surface,
    command_buffers: Vec<T::CommandBuffer>,
    vertex_buffer: Rc<<T::CommandBuffer as CommandBuffer>::VertexBuffer>,
    index_buffer: Rc<<T::CommandBuffer as CommandBuffer>::IndexBuffer>,
    graphics_pipeline: Rc<<T::CommandBuffer as CommandBuffer>::GraphicsPipeline>,
}

impl<T: Gpu> Environment<T> {

    fn new(system: &Rc<System>,gpu: &Rc<T>,ext: &'static str,title: &'static str,r: Rect<i32>) -> Result<Environment<T>,String> {
    
        // open frame window
        let window = Rc::new(e::Window::new_frame(&system,r,&format!("Single Quad ({})",title),)?);
    
        // create surface for the window
        let surface = gpu.create_surface(&window,r)?;
    
        // get number of frames in the swapchain of the surface
        let count = surface.get_swapchain_count();
    
        // create a command buffer for each swapchain frame
        let mut command_buffers: Vec<T::CommandBuffer> = Vec::new();
        for _ in 0..count {
            command_buffers.push(gpu.create_command_buffer()?);
        }
    
        // load and create vertex shader
        let mut f = File::open(format!("assets/quad-vs.{}",ext)).expect("Unable to open vertex shader");
        let mut code = Vec::<u8>::new();
        f.read_to_end(&mut code).expect("Unable to read vertex shader");
        let vertex_shader = Rc::new(gpu.create_vertex_shader(&code)?);
    
        // load and create fragment shader
        let mut f = File::open(format!("assets/quad-fs.{}",ext)).expect("Unable to open fragment shader");
        let mut code = Vec::<u8>::new();
        f.read_to_end(&mut code).expect("Unable to read fragment shader");
        let fragment_shader = Rc::new(gpu.create_fragment_shader(&code)?);
    
        // create vertex buffer
        let mut vertices = Vec::<MyVertex>::new();
        vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: -0.5, }, color: Vec3::<f32> { x: 1.0,y: 1.0,z: 0.0, }, });
        vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: -0.5, }, color: Vec3::<f32> { x: 0.0,y: 1.0,z: 1.0, }, });
        vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: 0.5, }, color: Vec3::<f32> { x: 1.0,y: 0.0,z: 1.0, }, });
        vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: 0.5, }, color: Vec3::<f32> { x: 0.0,y: 1.0,z: 0.0, }, });
        let vertex_buffer = Rc::new(gpu.create_vertex_buffer(&vertices)?);
    
        // create index buffer
        let mut indices = Vec::<u32>::new();
        indices.push(0);
        indices.push(1);
        indices.push(2);
        indices.push(0);
        indices.push(2);
        indices.push(3);
        let index_buffer = Rc::new(gpu.create_index_buffer(&indices)?);
    
        // create pipeline layout
        let pipeline_layout = Rc::new(gpu.create_pipeline_layout()?);
    
        // create graphics pipeline
        let graphics_pipeline = Rc::new(gpu.create_graphics_pipeline::<MyVertex>(
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
        )?);
    
        Ok(Environment {
            gpu: Rc::clone(&gpu),
            r,
            surface,
            command_buffers,
            vertex_buffer,
            index_buffer,
            graphics_pipeline,
        })
    }

    fn render(&mut self) -> Result<(),String> {

        // get index of next available swapchain buffer
        let index = self.surface.acquire()?;

        // build command buffer
        let cb = &mut self.command_buffers[index];
        cb.begin()?;
        cb.bind_graphics_pipeline(&self.graphics_pipeline);
        cb.bind_vertex_buffer(&self.vertex_buffer);
        cb.bind_index_buffer(&self.index_buffer);
        cb.begin_render_pass(&self.surface,index,Rect { o: Vec2::<i32>::ZERO,s: self.r.s, });
        cb.set_viewport(Rect { o: Vec2::<i32>::ZERO,s: self.r.s, }, 0.0, 1.0, );
        cb.set_scissor(Rect { o: Vec2::<i32>::ZERO,s: self.r.s, });
        cb.draw_indexed(6,1,0,0,0);
        cb.end_render_pass();
        cb.end();

        // submit command buffer
        self.gpu.submit_command_buffer(cb)?;

        // and present this buffer
        self.surface.present(index)?;

        Ok(())
    }
}

fn main() -> Result<(),String> {

    // open system
    let system = Rc::new(e::System::open()?);

    // define rectangles for the windows
    let r0 = Rect {
        o: Vec2 { x: 10i32,y: 10i32, },
        s: Vec2 { x: 512i32,y: 384i32, },
    };

    let r1 = Rect {
        o: Vec2 { x: 600i32,y: 10i32, },
        s: Vec2 { x: 512i32,y: 384i32, },
    };

    // create environments
    let mut vulkan = Environment::new(&system,&vulkan::Gpu::open(&system)?,"spv","Vulkan",r0)?;
    let mut opengl = Environment::new(&system,&opengl::Gpu::open(&system)?,"glsl","OpenGL",r1)?;

    // main loop
    let mut close_clicked = false;
    while !close_clicked {

        // wait for system events
        system.wait();

        // handle the system events
        system.flush().into_iter().try_for_each(|(_,event)| {
            if let Event::Close = event {
                close_clicked = true;
            }
            Ok::<(),String>(())
        })?;

        vulkan.render()?;
        opengl.render()?;
    }
    Ok(())
}
