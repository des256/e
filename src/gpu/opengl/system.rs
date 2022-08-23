use {
    crate::*,
    std::{
        rc::Rc,
        cell::Cell,
        os::raw::{
            c_int,
            c_void,
        },
        ptr::null_mut,
    },
};

const GLX_CONTEXT_MAJOR_VERSION_ARB: u32 = 0x2091;
const GLX_CONTEXT_MINOR_VERSION_ARB: u32 = 0x2092;

type GlXCreateContextAttribsARBProc = unsafe extern "C" fn(dpy: *mut sys::Display,fbc: sys::GLXFBConfig,share_context: sys::GLXContext,direct: c_int,attribs: *const c_int) -> sys::GLXContext;

// Supplemental fields for System
pub(crate) struct SystemGpu {
    pub glx_context: sys::GLXContext,
    pub xcb_hidden_window: sys::xcb_window_t,
    pub xcb_depth: u8,
    pub xcb_visual_id: u32,
}

pub(crate) fn open_system_gpu(xdisplay: *mut sys::Display,xcb_connection: *mut sys::xcb_connection_t,xcb_colormap: sys::xcb_colormap_t,xcb_root_window: sys::xcb_window_t) -> Option<SystemGpu> {

    // check if glX is useful
    let mut glxmaj: c_int = 0;
    let mut glxmin: c_int = 0;
    unsafe { if sys::glXQueryVersion(xdisplay,&mut glxmaj as *mut c_int,&mut glxmin as *mut c_int) == 0 { panic!("unable to get glX version"); } }
    if (glxmaj * 100 + glxmin) < 103 { panic!("glX version {}.{} needs to be at least 1.3",glxmaj,glxmin); }

    // choose appropriate framebuffer configuration
    let attribs = [
        sys::GLX_X_RENDERABLE,  1,
        sys::GLX_DRAWABLE_TYPE, sys::GLX_WINDOW_BIT,
        sys::GLX_RENDER_TYPE,   sys::GLX_RGBA_BIT,
        sys::GLX_X_VISUAL_TYPE, sys::GLX_TRUE_COLOR,
        sys::GLX_RED_SIZE,      8,
        sys::GLX_GREEN_SIZE,    8,
        sys::GLX_BLUE_SIZE,     8,
        sys::GLX_ALPHA_SIZE,    8,
        sys::GLX_DEPTH_SIZE,    24,
        sys::GLX_STENCIL_SIZE,  8,
        sys::GLX_DOUBLEBUFFER,  1,
        0,
    ];
    let mut fbcount: c_int = 0;
    let fbconfigs = unsafe { sys::glXChooseFBConfig(xdisplay,0,attribs.as_ptr() as *const i32,&mut fbcount as *mut c_int) };
    if fbcount == 0 { panic!("unable to find framebuffer config"); }
    let fbconfig = unsafe { *fbconfigs };
    unsafe { sys::XFree(fbconfigs as *mut c_void) };

    // adjust the window creation parameters accordingly
    let visual = unsafe { sys::glXGetVisualFromFBConfig(xdisplay,fbconfig) };
    let xcb_depth = unsafe { *visual }.depth as u8;
    let xcb_visual_id = unsafe { *visual }.visualid as u32;

    // get context creator
    let glx_create_context_attribs: GlXCreateContextAttribsARBProc = unsafe { std::mem::transmute(sys::glXGetProcAddress(b"glXCreateContextAttribARB" as *const u8)) };

    // create tiny window
    let xcb_hidden_window = unsafe { sys::xcb_generate_id(xcb_connection) };
    let values = [
        sys::XCB_EVENT_MASK_EXPOSURE
        | sys::XCB_EVENT_MASK_KEY_PRESS
        | sys::XCB_EVENT_MASK_KEY_RELEASE
        | sys::XCB_EVENT_MASK_BUTTON_PRESS
        | sys::XCB_EVENT_MASK_BUTTON_RELEASE
        | sys::XCB_EVENT_MASK_POINTER_MOTION
        | sys::XCB_EVENT_MASK_STRUCTURE_NOTIFY,
        xcb_colormap,
    ];
    unsafe {
        sys::xcb_create_window(
            xcb_connection,
            xcb_depth,
            xcb_hidden_window as u32,
            xcb_root_window,
            0,
            0,
            1,
            1,
            0,
            sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
            xcb_visual_id,
            sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
            &values as *const u32 as *const c_void
        );
        //sys::xcb_map_window(xcb_connection,xcb_hidden_window as u32);
        sys::xcb_flush(xcb_connection);
        sys::XSync(xdisplay,sys::False as c_int);
    }

    // create glX context
    let context_attribs: [c_int; 5] = [
        GLX_CONTEXT_MAJOR_VERSION_ARB as c_int, 4,
        GLX_CONTEXT_MINOR_VERSION_ARB as c_int, 5,
        0,
    ];
    let glx_context = unsafe { glx_create_context_attribs(xdisplay,fbconfig,std::ptr::null_mut(),sys::True as c_int,&context_attribs[0] as *const c_int) };
    unsafe {
        sys::xcb_flush(xcb_connection);
        sys::XSync(xdisplay,sys::False as c_int);
    }
    if glx_context.is_null() { panic!("unable to open OpenGL context"); }
    if unsafe { sys::glXIsDirect(xdisplay,glx_context) } == 0 { panic!("OpenGL context is not direct"); }
    unsafe { sys::glXMakeCurrent(xdisplay,xcb_hidden_window as u64,glx_context) };

    // load OpenGL symbols
    //gl::load_with(|symbol| load_function(&symbol));

    Some(SystemGpu {
        glx_context,
        xcb_hidden_window,
        xcb_depth,
        xcb_visual_id,
    })
}

impl System {

    /// Create GPU-specific Window part.
    pub fn create_window_gpu(&self) -> Option<WindowGpu> {
        Some(WindowGpu { visible_index: Cell::new(0), })
    }

    /// Create command buffer.
    pub fn create_command_buffer(self: &Rc<System>) -> Option<CommandBuffer> {

        Some(CommandBuffer {
            system: Rc::clone(self),
            commands: Vec::new(),
        })
    }

    /// Wait for wait_semaphore before submitting command buffer to the queue, and signal signal_semaphore when ready.
    pub fn submit(&mut self,command_buffer: &CommandBuffer,wait_semaphore: &Semaphore,signal_semaphore: &Semaphore) -> bool {

        // TODO: actually render the command buffer here

        false
    }

    pub fn drop_gpu(&self) {
        unsafe {
            sys::glXMakeCurrent(self.xdisplay,0,null_mut());
            sys::xcb_unmap_window(self.xcb_connection,self.gpu.xcb_hidden_window);
            sys::xcb_destroy_window(self.xcb_connection,self.gpu.xcb_hidden_window);
            sys::glXDestroyContext(self.xdisplay,self.gpu.glx_context);
        }
    }
}
