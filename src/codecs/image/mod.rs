use {
    crate::*,
    std::{
        fs::File,
        io::prelude::*,
    }
};

pub mod bmp;
pub mod png;
pub mod jpeg;

#[allow(dead_code)]
pub fn test(src: &[u8]) -> Option<(u32,u32)> {
    if let Some(size) = bmp::test(src) {
        Some(size)
    }
    else if let Some(size) = png::test(src) {
        Some(size)
    }
    else if let Some(size) = jpeg::test(src) {
        Some(size)
    }
    else {
        None
    }
}

#[allow(dead_code)]
pub fn decode<T: Pixel + Default>(src: &[u8]) -> Option<Mat<T>> {
    if let Some(image) = bmp::decode::<T>(src) {
        Some(image)
    }
    else if let Some(image) = png::decode::<T>(src) {
        Some(image)
    }
    else if let Some(image) = jpeg::decode::<T>(src) {
        Some(image)
    }
    else {
        None
    }
}

pub fn load<T: Pixel + Default>(filename: &str) -> Option<Mat<T>> {
    let mut file = match File::open(filename) {
        Ok(file) => file,
        Err(_) => { return None; },
    };
    let mut buffer: Vec<u8> = Vec::new();
    if let Ok(_) = file.read_to_end(&mut buffer) {
        decode::<T>(&buffer)
    }
    else {
        None
    }
}