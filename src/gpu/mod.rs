trait RGBA {
    fn set(c: u32);
    fn get() -> u32;
    fn set_rgba(r: u8,g: u8,b: u8,a: u8);
    fn get_rgba() -> (u8,u8,u8,u8);
    fn set_rgbaf(r: f32,g: f32,b: f32,a: f32);
    fn get_rgbaf() -> (f32,f32,f32,f32);
}
