use {
    e::*,
    std::time::Duration,
};

fn main() {
    let (executor,spawner) = new_executor_and_spawner();
    spawner.spawn(async {
        println!("howdy!");
        Timer::new(Duration::new(2,0)).await;
        println!("done!");
    });
    drop(spawner);
    executor.run();
}
