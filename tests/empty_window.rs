use {
    e::*,
    std::result::Result,
};

#[test]
fn main() -> Result<(),String> {
    //let executor = Executor::build()?;
    dprintln!("opening system");
    let system = e::System::open()?;
    dprintln!("creating window");
    let window = system.create_frame_window(Rect { o: Vec2 { x: 50.0f32,y: 50.0f32, },s: Vec2 { x: 800.0f32,y: 600.0f32, }, },"Hello, World!")?;
    //executor.run();
    dprintln!("done.");
    Ok(())
}
