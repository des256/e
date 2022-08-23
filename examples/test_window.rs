use e::*;

fn main() {

    let system = open_system().expect("Unable to access system.");
    let window = system.create_frame_window(
        Rect {
            o: Vec2::<isize> {
                x: 50,
                y: 50,
            },
            s: Vec2::<usize> {
                x: 640,
                y: 350,
            },
        },
        "Test Window"
    ).expect("Unable to create window.");
    let mut running = true;
    while running {
        system.wait();
        let events = system.flush();
        for (id,event) in events {
            if id == window.id() {
                dprintln!("{}",event);
                if let Event::Close = event {
                    running = false;
                }
            }
        }
    }
}
