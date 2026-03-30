use std::fmt;

/// RGBA color with CSS string output.
///
/// Components are stored as `u8` RGB plus `f32` alpha. Construct from
/// individual channels with [`rgb`](Color::rgb)/[`rgba`](Color::rgba),
/// or from a `0xRRGGBB` integer via [`From<u32>`].
///
/// # Examples
///
/// ```
/// use webui::color::Color;
///
/// let c = Color::rgb(0x21, 0x96, 0xf3);
/// assert_eq!(c.to_css(), "#2196f3");
///
/// let t = Color::from(0xff8800).alpha(0.5);
/// assert_eq!(t.to_css(), "rgba(255,136,0,0.50)");
/// ```
#[derive(Copy, Clone, PartialEq)]
pub struct Color {
    /// Red channel (0..255).
    pub r: u8,
    /// Green channel (0..255).
    pub g: u8,
    /// Blue channel (0..255).
    pub b: u8,
    /// Alpha channel (0.0 = transparent, 1.0 = opaque).
    pub a: f32,
}

// -- constants --

impl Color {
    /// Opaque black.
    pub const BLACK: Color = Color::rgb(0, 0, 0);
    /// Opaque white.
    pub const WHITE: Color = Color::rgb(255, 255, 255);
    /// Fully transparent.
    pub const TRANSPARENT: Color = Color { r: 0, g: 0, b: 0, a: 0.0 };
}

// -- constructors --

impl Color {
    /// Create an opaque color from RGB channels.
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Color { r, g, b, a: 1.0 }
    }

    /// Create a color from RGBA channels.
    pub const fn rgba(r: u8, g: u8, b: u8, a: f32) -> Self {
        Color { r, g, b, a }
    }

    /// Return a copy with the given alpha.
    pub const fn alpha(self, a: f32) -> Self {
        Color { a, ..self }
    }

    /// Format as a CSS color string (`#rrggbb` or `rgba(r,g,b,a)`).
    pub fn to_css(&self) -> String {
        if self.a >= 1.0 {
            format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
        } else {
            format!("rgba({},{},{},{:.2})", self.r, self.g, self.b, self.a)
        }
    }
}

// -- trait impls --

impl fmt::Debug for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Color({})", self.to_css())
    }
}

/// Convert a `0xRRGGBB` integer to a [`Color`].
impl From<u32> for Color {
    fn from(hex: u32) -> Self {
        Color::rgb(
            ((hex >> 16) & 0xFF) as u8,
            ((hex >> 8) & 0xFF) as u8,
            (hex & 0xFF) as u8,
        )
    }
}
