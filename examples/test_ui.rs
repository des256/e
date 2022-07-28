use {
    e::*,
    std::{
        rc::Rc,
        time::Duration,
    },
};

fn main() {
    let ui = Rc::new(Container::new(Rc::new(Text::new("Hello!"))));
    run_ui_loop(ui);
}
