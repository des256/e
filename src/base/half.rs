#[allow(non_camel_case_types)]
pub struct f16(pub u16);

impl f16 {
    pub fn from_f32(value: f32) -> f16 {
        f16(0)
    }

    pub fn from_f64(value: f64) -> f16 {
        f16(0)
    }

    pub fn as_f32(&self) -> f32 {
        0.0
    }

    pub fn as_f64(&self) -> f64 {
        0.0
    }
}