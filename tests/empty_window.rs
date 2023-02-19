use {
    e,
    std::result::Result,
};

#[test]
fn main() -> Result<(),String> {
    //let executor = Executor::build()?;
    let system = e::System::open()?;
    //let window = system.create_frame()?;
    //executor.run();
    Ok(())
}
