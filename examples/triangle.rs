use {
    e::*,
    macros::*,
    std::{
        result::Result,
        rc::Rc,
        io::*,
        fs::*,
    },
};

// the vertex format
#[derive(Vertex)]
struct MyVertex {
    pos: Vec2<f32>,
}

#[vertex_shader]
mod triangle_vs {
    fn main(vertex: MyVertex) -> Vec4<f32> {
        vertex.pos
    }
}

#[fragment_shader]
mod triangle_fs {
    fn main() -> Vec4<f32> {
        Vec4 { x: 1.0,y: 0.5,z: 0.0,w: 1.0, }
    }
}

// environment template that fits on an arbitrary GPU
struct Environment<T: Gpu> {
    gpu: Rc<T>,
    r: Rect<i32>,
    surface: <T::CommandBuffer as CommandBuffer>::Surface,
    command_buffers: Vec<T::CommandBuffer>,
    vertex_buffer: Rc<<T::CommandBuffer as CommandBuffer>::VertexBuffer>,
    graphics_pipeline: Rc<<T::CommandBuffer as CommandBuffer>::GraphicsPipeline>,
}

impl<T: Gpu> Environment<T> {

    fn new(system: &Rc<System>,gpu: &Rc<T>,ext: &'static str,title: &'static str,r: Rect<i32>) -> Result<Environment<T>,String> {
    
        // open frame window
        let window = Rc::new(e::Window::new_frame(&system,r,&format!("Triangle ({})",title),)?);
    
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
        let mut f = File::open(format!("assets/triangle-vs.{}",ext)).expect("Unable to open vertex shader");
        let mut code = Vec::<u8>::new();
        f.read_to_end(&mut code).expect("Unable to read vertex shader");
        let vertex_shader = Rc::new(gpu.create_vertex_shader(&triangle_vs::ast())?);
    
        // load and create fragment shader
        let mut f = File::open(format!("assets/triangle-fs.{}",ext)).expect("Unable to open fragment shader");
        let mut code = Vec::<u8>::new();
        f.read_to_end(&mut code).expect("Unable to read fragment shader");
        let fragment_shader = Rc::new(gpu.create_fragment_shader(&triangle_fs::ast())?);
    
        // create vertex buffer
        let mut vertices = Vec::<MyVertex>::new();
        vertices.push(MyVertex { pos: Vec2::<f32> { x: -0.5,y: -0.3, }, });
        vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.0,y: 0.8, }, });
        vertices.push(MyVertex { pos: Vec2::<f32> { x: 0.5,y: -0.3, }, });
        let vertex_buffer = Rc::new(gpu.create_vertex_buffer(&vertices)?);
    
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
        cb.begin_render_pass(&self.surface,index,Rect { o: Vec2::<i32>::ZERO,s: self.r.s, });
        cb.set_viewport(Rect { o: Vec2::<i32>::ZERO,s: self.r.s, }, 0.0, 1.0, );
        cb.set_scissor(Rect { o: Vec2::<i32>::ZERO,s: self.r.s, });
        cb.draw(3,1,0,0);
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

    // create environments
#[cfg(any(system="linux"))]
    //let mut vulkan = Environment::new(&system,&vulkan::Gpu::open(&system)?,"spv","Vulkan",Rect::<i32> { o: Vec2::ZERO,s: Vec2 { x: 512,y: 384, }, })?;
#[cfg(any(system="linux"))]
    let mut opengl = Environment::new(&system,&opengl::Gpu::open(&system)?,"glsl","OpenGL",Rect::<i32> { o: Vec2::ZERO,s: Vec2 { x: 512,y: 384, }, })?;

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

        // render stuff
#[cfg(any(system="linux"))]
        //vulkan.render()?;
#[cfg(any(system="linux"))]
        opengl.render()?;
    }
    Ok(())
}
