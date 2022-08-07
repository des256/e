use {
    e::*,
    std::{
        fs::File,
        io::prelude::*,
    },
};

fn main() {

    // create frame window
    let system = open_system().expect("unable to access system");
    let mut r: Rect<i32> = rect!(0,0,640,480);

    let mut window = system.create_frame_window(rect!(100,100,r.s.x,r.s.y),"test triangle").expect("unable to create frame window");

    // read vertex shader
    let mut f = File::open("test-triangle-vert.spv").expect("unable to open vertex shader");
    let mut b = Vec::<u8>::new();
    f.read_to_end(&mut b).expect("unable to read vertex shader.");
    let vertex_shader = system.create_shader(&b).expect("unable to create vertex shader");

    // read fragment shader
    let mut f = File::open("test-triangle-frag.spv").expect("unable to open fragment shader");
    let mut b = Vec::<u8>::new();
    f.read_to_end(&mut b).expect("unable to read fragment shader");
    let fragment_shader = system.create_shader(&b).expect("unable to create fragment shader");

    // get the window's render pass
    let render_pass = window.get_render_pass();

    // create pipeline layout
    let pipeline_layout = system.create_pipeline_layout().expect("unable to create pipeline layout");
    
    // create graphics pipeline with this layout
    let graphics_pipeline = render_pass.create_graphics_pipeline(&pipeline_layout,&vertex_shader,&fragment_shader).expect("Unable to create graphics pipeline.");
    
    // create command buffers, one for each framebuffer
    let mut command_buffers = Vec::<CommandBuffer>::new();

    // create a command buffer for each frame buffer
    for _framebuffer in &window.get_framebuffers() {
        let command_buffer = system.create_commandbuffer().expect("unable to create command buffer");
        command_buffers.push(command_buffer);
    }

    // create the semaphores
    let image_available = system.create_semaphore().expect("Unable to create image available semaphore.");
    let render_finished = system.create_semaphore().expect("Unable to create render finished semaphore.");

    // and go
    let mut running = true;
    while running {
        let index = window.acquire_next(&image_available);

        if command_buffers[index].begin() {
            let framebuffers = window.get_framebuffers();
            command_buffers[index].set_viewport(hyper!(0.0,0.0,0.0,r.s.x as f32,r.s.y as f32,1.0));
            command_buffers[index].set_scissor(rect!(0,0,r.s.x,r.s.y));
            command_buffers[index].bind_pipeline(&graphics_pipeline);
            command_buffers[index].begin_render_pass(&window.get_render_pass(),&framebuffers[index],rect!(0,0,r.s.x,r.s.y));
            command_buffers[index].draw(3,1,0,0);
            command_buffers[index].end_render_pass();
            if !command_buffers[index].end() {
                println!("unable to end command buffer");
            }
        }
        else {
            println!("unable to begin command buffer");
        }

        if !system.submit(&command_buffers[index],&image_available,&render_finished) {
             println!("unable to submit command buffer");
        }

        window.present(index,&render_finished);

        system.wait();

        let events = system.flush();
        for (id,event) in events {
            dprintln!("event: {} ({})",event,id);
            if id == window.id() {
                if let Event::Configure(new_r) = event {
                    r = new_r;
                    window.update_configure(r);
                }
                if let Event::Close = event {
                    running = false;
                }
            }
        }
    }
}
