use crate::*;

pub trait Pixel: Copy + Clone + Zero {
    fn set(&mut self,r: u8,g: u8,b: u8,a: u8);
    fn get(&self) -> (u8,u8,u8,u8);
}

#[derive(Copy,Clone)]
pub struct BGRA8UN { b: u8,g: u8,r: u8,a: u8, }

impl Zero for BGRA8UN { const ZERO: BGRA8UN = BGRA8UN { b: 0,g: 0,r: 0,a: 0, }; }

impl Pixel for BGRA8UN {
    fn set(&mut self,r: u8,g: u8,b: u8,a: u8) {
        self.r = r;
        self.g = g;
        self.b = b;
        self.a = a;
    }

    fn get(&self) -> (u8,u8,u8,u8) {
        (self.r,self.g,self.b,self.a)
    }
}
