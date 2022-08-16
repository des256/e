use {
    crate::*,
    std::{
        cmp::PartialEq,
        fmt::{
            Display,
            Debug,
            Formatter,
            Result,
        },
        ops::{
            Add,
            Sub,
            Mul,
            Div,
            AddAssign,
            SubAssign,
            MulAssign,
            DivAssign,
        },
    },
};

#[derive(Copy,Clone,Debug)]
pub struct Color<T> {
    pub r: T,
    pub g: T,
    pub b: T,
    pub a: T,
}

impl<T: Copy> Color<T> {
    pub fn new(r: T,g: T,b: T,a: T) -> Self {
        Color { r: r,g: g,b: b,a: a, }
    }
}

macro_rules! color_argb_int {
    ($($t:ty)+) => {
        $(
            impl const From<u32> for Color<$t> {
                fn from(argb: u32) -> Self {
                    let mut r = ((argb >> 16) & 255) as $t;
                    let mut g = ((argb >> 8) & 255) as $t;
                    let mut b = (argb & 255) as $t;
                    let mut a = (argb >> 24) as $t;
                    let mut bits = 8;
                    while bits < <$t>::BITS {
                        r = (r << bits) | r;
                        g = (g << bits) | g;
                        b = (b << bits) | b;
                        a = (a << bits) | a;
                        bits <<= 1;
                    }
                    Color {
                        r: r,
                        g: g,
                        b: b,
                        a: a,
                    }
                }
            }

            impl const From<Color<$t>> for u32 {
                fn from(color: Color<$t>) -> Self {
                    let mut r = (color.r >> (<$t>::BITS - 8)) as u32;
                    let mut g = (color.g >> (<$t>::BITS - 8)) as u32;
                    let mut b = (color.b >> (<$t>::BITS - 8)) as u32;
                    let mut a = (color.a >> (<$t>::BITS - 8)) as u32;
                    (a << 24) | (r << 16) | (g << 8) | b
                }
            }
        )+
    }
}

color_argb_int!(u8 u16);

macro_rules! color_argb_float {
    ($($t:ty)+) => {
        $(
            impl const From<u32> for Color<$t> {
                fn from(argb: u32) -> Self {
                    let r = (((argb >> 16) & 255) as $t) / 255.0;
                    let g = (((argb >> 8) & 255) as $t) / 255.0;
                    let b = ((argb & 255) as $t) / 255.0;
                    let a = ((argb >> 24) as $t) / 255.0;
                    Color {
                        r: r,
                        g: g,
                        b: b,
                        a: a,
                    }
                }
            }

            impl const From<Color<$t>> for u32 {
                fn from(color: Color<$t>) -> Self {
                    let mut r = if (color.r <= 0.0) { 0u32 } else if (color.r >= 1.0) { 255u32 } else { (color.r * 255.0) as u32 };
                    let mut g = if (color.g <= 0.0) { 0u32 } else if (color.g >= 1.0) { 255u32 } else { (color.g * 255.0) as u32 };
                    let mut b = if (color.b <= 0.0) { 0u32 } else if (color.b >= 1.0) { 255u32 } else { (color.b * 255.0) as u32 };
                    let mut a = if (color.a <= 0.0) { 0u32 } else if (color.a >= 1.0) { 255u32 } else { (color.a * 255.0) as u32 };
                    (a << 24) | (r << 16) | (g << 8) | b
                }
            }
        )+
    }
}

color_argb_float!(f32 f64);

impl<T: PartialEq> PartialEq for Color<T> {
    fn eq(&self,other: &Self) -> bool {
        (self.r == other.r) && (self.g == other.g) && (self.b == other.b) && (self.a == other.a)
    }
}

impl<T: Display> Display for Color<T> {
    fn fmt(&self,f: &mut Formatter) -> Result {
        write!(f,"({},{},{},{})",self.r,self.g,self.b,self.a)
    }
}

impl<T: Zero> Zero for Color<T> {
    const ZERO: Self = Color { r: T::ZERO,g: T::ZERO,b: T::ZERO,a: T::ZERO, };
}

impl<T: Add<T,Output=T>> Add<Color<T>> for Color<T> {
    type Output = Self;
    fn add(self,other: Self) -> Self {
        Color { r: self.r + other.r,g: self.g + other.g,b: self.b + other.b,a: self.a + other.a, }
    }
}

impl<T: AddAssign<T>> AddAssign<Color<T>> for Color<T> {
    fn add_assign(&mut self,other: Self) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
        self.a += other.a;
    }
}

impl<T: Sub<T,Output=T>> Sub<Color<T>> for Color<T> {
    type Output = Self;
    fn sub(self,other: Self) -> Self {
        Color { r: self.r - other.r,g: self.g - other.g,b: self.b - other.b,a: self.a - other.a, }
    }
}

impl<T: SubAssign<T>> SubAssign<Color<T>> for Color<T> {
    fn sub_assign(&mut self,other: Self) {
        self.r -= other.r;
        self.g -= other.g;
        self.b -= other.b;
        self.a -= other.a;
    }
}

macro_rules! scalar_color_mul {
    ($($t:ty)+) => {
        $(
            impl Mul<Color<$t>> for $t {
                type Output = Color<$t>;
                fn mul(self,other: Color<$t>) -> Color<$t> {
                    Color { r: self * other.r,g: self * other.g,b: self * other.b,a: self * other.a, }
                }
            }
        )+
    }
}

scalar_color_mul!(u8 u16 f32 f64);

impl<T: Copy + Mul<T,Output=T>> Mul<T> for Color<T> {
    type Output = Self;
    fn mul(self,other: T) -> Self {
        Color { r: self.r * other,g: self.g * other,b: self.b * other,a: self.a * other, }
    }
}

impl<T: Copy + Mul<T,Output=T>> Mul<Color<T>> for Color<T> {
    type Output = Self;
    fn mul(self,other: Color<T>) -> Self {
        Color { r: self.r * other.r,g: self.g * other.g,b: self.b * other.b,a: self.a * other.a, }
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<T> for Color<T> {
    fn mul_assign(&mut self,other: T) {
        self.r *= other;
        self.g *= other;
        self.b *= other;
        self.a *= other;
    }
}

impl<T: Copy + MulAssign<T>> MulAssign<Color<T>> for Color<T> {
    fn mul_assign(&mut self,other: Color<T>) {
        self.r *= other.r;
        self.g *= other.g;
        self.b *= other.b;
        self.a *= other.a;
    }
}

impl<T: Copy + Div<T,Output=T>> Div<T> for Color<T> {
    type Output = Self;
    fn div(self,other: T) -> Self {
        Color { r: self.r / other,g: self.g / other,b: self.b / other,a: self.a / other, }
    }
}

impl<T: Copy + DivAssign<T>> DivAssign<T> for Color<T> {
    fn div_assign(&mut self,other: T) {
        self.r /= other;
        self.g /= other;
        self.b /= other;
        self.a /= other;
    }
}

#[allow(non_camel_case_types)]
pub type u8rgba = Color<u8>;
#[allow(non_camel_case_types)]
pub type u16rgba = Color<u16>;
#[allow(non_camel_case_types)]
pub type f32rgba = Color<f32>;
#[allow(non_camel_case_types)]
pub type f64rgba = Color<f64>;

macro_rules! color_consts {
    ($($t:ty)+) => {
        $(
            impl Color<$t> {
                pub const BLACK: Self = Color::from(0xFF000000);
                pub const BLUE: Self = Color::from(0xFF0000FF);
                pub const GREEN: Self = Color::from(0xFF00FF00);
                pub const CYAN: Self = Color::from(0xFF00FFFF);
                pub const RED: Self = Color::from(0xFFFF0000);
                pub const MAGENTA: Self = Color::from(0xFFFF00FF);
                pub const YELLOW: Self = Color::from(0xFFFFFF00);
                pub const WHITE: Self = Color::from(0xFFFFFFFF);
            }
        )+
    }
}

color_consts!(u8 u16 f32 f64);
