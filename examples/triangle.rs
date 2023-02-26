use {
    e::*,
    gpu_macros::*,
    std::{
        result::Result,
        rc::Rc,
    },
};

#[derive(Vertex)]
struct MyVertex {
    pub pos: Vec2<f32>,
}

fn main() -> Result<(),String> {
    let system = Rc::new(e::System::open()?);
    let frame_window = e::Window::new_frame(&system,
        Rect {
            o: Vec2 { x: 10.0f32,y: 10.0f32, },
            s: Vec2 { x: 800.0f32,y: 600.0f32, },
        },
        "Single Triangle",
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
