use e::*;

fn main() {

    let system = open_system().expect("Unable to access system.");
    let gpus = system.enumerate_gpus();
    println!("GPUs:");
    for gpu in &gpus {
        println!("    {}",gpu);
    }
    //let screens = gpus[0].enumerate_screens();
    //let screen = &screens[0];
    //let window = screen.create_frame(rect!(50,50,640,350),"Test Window").expect("Unable to create window.");
    /*
    window.set_handler(move |event| {
        match event {
            Event::Close => {
                window_running.set(false);
            },
            _ => {
                println!("{}",event);
            },
        }
    });

    while running.get() {
        system.wait();
        system.flush();
    }
    */
}
