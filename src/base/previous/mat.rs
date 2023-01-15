use {
    crate::*,
    std::{
        marker::PhantomData,
        ops::{
            Index,
            IndexMut,
        },
    },
};

#[derive(Clone)]
pub struct Mat<T: Clone + Copy + Zero> {
    pub size: Vec2<usize>,
    data: Box<[T]>,
    phantom: PhantomData<T>,
}

impl<T: Clone + Copy + Zero> Mat<T> {
    pub fn new(size: Vec2<usize>) -> Mat<T> {
        Mat {
            size: size,
            data: vec![T::ZERO; (size.x * size.y) as usize].into_boxed_slice(),
            phantom: PhantomData,
        }
    }

    pub fn data(&self) -> &[T] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Clone + Copy + Zero> Index<(usize,usize)> for Mat<T> {
    type Output = T;
    fn index(&self,index: (usize,usize)) -> &Self::Output {
        &self.data[index.1 * self.size.x + index.0]
    }
}

impl<T: Clone + Copy + Zero> IndexMut<(usize,usize)> for Mat<T> {
    fn index_mut(&mut self,index: (usize,usize)) -> &mut Self::Output {
        &mut self.data[index.1 * self.size.x + index.0]
    }
}

impl<T: Clone + Copy + Zero> Index<usize> for Mat<T> {
    type Output = T;
    fn index(&self,index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T: Clone + Copy + Zero> IndexMut<usize> for Mat<T> {
    fn index_mut(&mut self,index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Clone + Copy + Zero> Index<Vec2<usize>> for Mat<T> {
    type Output = T;
    fn index(&self,index: Vec2<usize>) -> &Self::Output {
        &self.data[(index.y as usize) * self.size.x + (index.x as usize)]
    }
}

impl<T: Clone + Copy + Zero> IndexMut<Vec2<usize>> for Mat<T> {
    fn index_mut(&mut self,index: Vec2<usize>) -> &mut Self::Output {
        &mut self.data[index.y * self.size.x + index.x]
    }
}
