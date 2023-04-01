mod ast;
pub use ast::*;

mod astdisplay;

mod stdlib;
pub use stdlib::*;

mod process;
pub use process::*;

pub enum ShaderStyle {
    Vertex,
    Fragment,
}

//mod tac;
//pub use tac::*;

//mod rendertac;
//pub use rendertac::*;

pub mod glsl;

pub mod spirv;
