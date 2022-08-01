mod session;
pub use session::*;

mod commandbuffer;
pub use commandbuffer::*;

mod renderpass;
pub use renderpass::*;

/*

System -> Gpu -> Screen
Screen -> Window
Screen -> CommandBuffer
screen.submit
window.present

every window has a swapchain with corresponding framebuffers

*/