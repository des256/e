use {
    e::*,
    std::{
        rc::Rc,
        result::Result,
    },
};

fn main() -> Result<(),String> {
    let system = Rc::new(e::System::open()?);
    let _frame_window = e::Window::new_frame(&system,
        Rect {
            o: Vec2 { x: 10i32,y: 10i32, },
            s: Vec2 { x: 800i32,y: 600i32, },
        },
        "Hello, World!",
    )?;
    let mut close_clicked = false;
    while !close_clicked {
        system.wait();
        system.flush().into_iter().for_each(|(id,event)| {
            dprintln!("event {} for window {}",event,id);
            if let Event::Close = event {
                close_clicked = true;
            }
        });
    }
    Ok(())
}
