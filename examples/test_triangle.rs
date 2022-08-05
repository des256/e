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
    let mut window = system.create_frame_window(rect!(50,50,640,350),"test triangle").expect("unable to create frame window");

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

    // create pipeline layout
    let pipeline_layout = system.create_pipeline_layout().expect("unable to create pipeline layout");

    // create graphics pipeline for the window using the pipeline layout
    let graphics_pipeline = system.create_graphics_pipeline(&pipeline_layout,&window,&vertex_shader,&fragment_shader).expect("Unable to create graphics pipeline.");
    
    // create command buffers, one for each framebuffer
    let mut command_buffers = Vec::<CommandBuffer>::new();

    // fill the command buffers with render commands
    for framebuffer in &window.get_framebuffers() {
        let command_buffer = system.create_commandbuffer().expect("unable to create command buffer");
        if command_buffer.begin() {
            command_buffer.bind_pipeline(&graphics_pipeline);
            command_buffer.begin_render_pass(&window,framebuffer);
            command_buffer.draw(3,1,0,0);
            command_buffer.end_render_pass();
            if !command_buffer.end() {
                println!("unable to end command buffer");
            }
        }
        else {
            println!("unable to begin command buffer");
        }
        command_buffers.push(command_buffer);
    }

    // create the semaphores
    let image_available = system.create_semaphore().expect("Unable to create image available semaphore.");
    let render_finished = system.create_semaphore().expect("Unable to create render finished semaphore.");

    // and go
    let mut running = true;
    while running {
        dprintln!("acquiring next frame...");
        let index = window.acquire_next(&image_available);

        dprintln!("submitting graphics to frame {}...",index);
        if !system.submit_graphics(&command_buffers[index],&image_available,&render_finished) {
             println!("unable to submit command buffer");
        }

        dprintln!("presenting...");
        window.present(index,&render_finished);

        //system.wait();

        dprintln!("flushing system queue...");
        let events = system.flush();
        for (id,event) in events {
            if id == window.id() {
                if let Event::Configure(r) = event {
                    window.update_configure(r);
                }
                if let Event::Close = event {
                    running = false;
                }
            }
        }
    }
}
