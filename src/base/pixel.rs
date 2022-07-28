use crate::*;

pub trait Pixel: Copy + Clone + Zero {
    fn set(&mut self,r: u8,g: u8,b: u8,a: u8);
    fn get(&self) -> (u8,u8,u8,u8);
}

pub mod pixelformat {

    use crate::*;

    #[derive(Copy,Clone)]
    pub struct RGBA8UN { r: u8,g: u8,b: u8,a: u8, }

    impl Pixel for RGBA8UN {
        fn set(&mut self,r: u8,g: u8,b: u8,a: u8) { self.r = r; self.g = g; self.b = b; self.a = a; }
        fn get(&self) -> (u8,u8,u8,u8) { (self.r,self.g,self.b,self.a) }
    }

    impl Zero for RGBA8UN { const ZERO: RGBA8UN = RGBA8UN { r: 0,g: 0,b: 0,a: 0, }; }

    #[derive(Copy,Clone)]
    pub struct BGRA8UN { b: u8,g: u8,r: u8,a: u8, }

    impl Pixel for BGRA8UN {
        fn set(&mut self,r: u8,g: u8,b: u8,a: u8) { self.r = r; self.g = g; self.b = b; self.a = a; }
        fn get(&self) -> (u8,u8,u8,u8) { (self.r,self.g,self.b,self.a) }
    }

    impl Zero for BGRA8UN { const ZERO: BGRA8UN = BGRA8UN { r: 0,g: 0,b: 0,a: 0, }; }
}