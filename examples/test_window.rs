use e::*;
use std::{
    rc::Rc,
    cell::Cell,
};

fn main() {

    let system = System::new().expect("Unable to access system.");
    let gpus = system.find_gpus();
    let screens = system.find_screens(&gpus);
    let screen = &screens[0];

    let running = Rc::new(Cell::new(true));
    let window_running = Rc::clone(&running);
    let window = screen.create_frame(rect!(50,50,640,350),"Test Window").expect("Unable to create window.");
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
}
