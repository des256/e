use {
    e::*,
    std::rc::Rc,
};

fn main() {
    let _ui = Rc::new(Container::new(Rc::new(Text::new("Hello!"))));
    //run_ui_loop(ui);
}
