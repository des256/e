use {
    super::*,
    std::ops::{
        Index,
        IndexMut,
    },
};

#[derive(Clone,Debug)]
pub struct Image<T> where T : Pixel {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<T>,
}

impl<T: Pixel> Image<T> {
    pub fn new(width: usize,height: usize) -> Image<T> {
        Image {
            width,
            height,
            pixels: vec![T::ZERO; width * height],
        }
    }
}

impl<T: Pixel> Index<(usize,usize)> for Image<T> {
    type Output = T;
    fn index(&self,index: (usize,usize)) -> &T {
        &self.pixels[index.1 * self.width + index.0]
    }
}

impl<T: Pixel> IndexMut<(usize,usize)> for Image<T> {
    fn index_mut(&mut self,index: (usize,usize)) -> &mut T {
        &mut self.pixels[index.1 * self.width + index.0]
    }
}
