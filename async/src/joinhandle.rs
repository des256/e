pub struct JoinHandle<T> {}

impl<T> JoinHandle<T> {
    pub fn abort(&self) {}
    pub fn is_finished(&self) -> bool {
        true
    }
}
