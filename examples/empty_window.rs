use {
    e::*,
    std::result::Result,
};

fn main() -> Result<(),String> {
    let system = e::System::open()?;
    let frame_window = system.create_frame_window(
        Rect {
            o: Vec2 { x: 10.0f32,y: 10.0f32, },
            s: Vec2 { x: 800.0f32,y: 600.0f32, },
        },
        "Hello, World!",
    )?;
    let popup_window = system.create_popup_window(
        Rect {
            o: Vec2 { x: 820.0f32,y: 10.0f32, },
            s: Vec2 { x: 200.0f32,y: 50.0f32, },
        },
    );
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
