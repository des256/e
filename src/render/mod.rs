use {
    crate::*,
};

trait RGBA {
    fn set(c: u32);
    fn get() -> u32;
}
