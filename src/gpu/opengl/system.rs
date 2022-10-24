use {
    crate::*,
    std::{
        rc::Rc,
        cell::{
            Cell,
            RefCell,
        },
        os::raw::{
            c_int,
            c_void,
        },
        ffi::CStr,
        ptr::null_mut,
        mem::size_of,
    },
};

// Supplemental fields for System
pub(crate) struct SystemGpu {
    pub xcb_depth: u8,
    pub xcb_visual_id: u32,
    pub glx_context: sys::GLXContext,
    pub xcb_hidden_window: sys::xcb_window_t,
}

pub(crate) fn open_system_gpu(xdisplay: *mut sys::Display,xcb_connection: *mut sys::xcb_connection_t,xcb_root_window: sys::xcb_window_t) -> Option<SystemGpu> {

    // TODO: load the OpenGL symbols here for systems that don't support header prototypes (Windows, mainly)

    // check if glX is useful
    let mut glxmaj: c_int = 0;
    let mut glxmin: c_int = 0;
    unsafe { if sys::glXQueryVersion(xdisplay,&mut glxmaj as *mut c_int,&mut glxmin as *mut c_int) == 0 { panic!("unable to get glX version"); } }
    if (glxmaj * 100 + glxmin) < 103 { panic!("glX version {}.{} needs to be at least 1.3",glxmaj,glxmin); }

    let vendor_cstr = unsafe { CStr::from_ptr(sys::glXGetClientString(xdisplay,sys::GLX_VENDOR as c_int)) };
    let version_cstr = unsafe { CStr::from_ptr(sys::glXGetClientString(xdisplay,sys::GLX_VERSION as c_int)) };
    println!("OpenGL glX vendor: {}",vendor_cstr.to_str().unwrap());
    println!("OpenGL glX version: {}",version_cstr.to_str().unwrap());

    // choose appropriate framebuffer configuration
    let attribs = [
        sys::GLX_X_RENDERABLE,   1,
        sys::GLX_DRAWABLE_TYPE,  sys::GLX_WINDOW_BIT,
        sys::GLX_RENDER_TYPE,    sys::GLX_RGBA_BIT,
        sys::GLX_X_VISUAL_TYPE,  sys::GLX_TRUE_COLOR,
        sys::GLX_RED_SIZE,       8,
        sys::GLX_GREEN_SIZE,     8,
        sys::GLX_BLUE_SIZE,      8,
        sys::GLX_ALPHA_SIZE,     8,
        sys::GLX_DEPTH_SIZE,     24,
        sys::GLX_STENCIL_SIZE,   8,
        sys::GLX_DOUBLEBUFFER,   1,
        sys::GLX_CONFIG_CAVEAT,  sys::GLX_NONE,
        sys::GLX_BUFFER_SIZE,    32,
        0,
    ];
    let mut fbcount: c_int = 0;
    let fbconfigs = unsafe { sys::glXChooseFBConfig(xdisplay,0,attribs.as_ptr() as *const i32,&mut fbcount as *mut c_int) };
    if fbcount == 0 { panic!("unable to find framebuffer config"); }
    let fbconfig = unsafe { *fbconfigs };

    // adjust the window creation parameters accordingly
    let visual = unsafe { sys::glXGetVisualFromFBConfig(xdisplay,fbconfig) };
    let xcb_depth = unsafe { *visual }.depth as u8;
    let xcb_visual_id = unsafe { *visual }.visualid as u32;

    unsafe { sys::XFree(fbconfigs as *mut c_void) };

    // create tiny window
    let xcb_hidden_window = unsafe { sys::xcb_generate_id(xcb_connection) };
    let xcb_colormap = unsafe { sys::xcb_generate_id(xcb_connection) };
    unsafe { sys::xcb_create_colormap(xcb_connection,sys::XCB_COLORMAP_ALLOC_NONE as u8,xcb_colormap,xcb_root_window,xcb_visual_id) };
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
        //sys::xcb_map_window(xcb_connection,xcb_hidden_window);
        sys::XSync(xdisplay,0);
    }

    // create glX context
    let context_attribs: [c_int; 5] = [
        sys::GLX_CONTEXT_MAJOR_VERSION_ARB as c_int, 4,
        sys::GLX_CONTEXT_MINOR_VERSION_ARB as c_int, 5,
        0,
    ];
    let glx_context = unsafe { sys::glXCreateContextAttribsARB(xdisplay,fbconfig,std::ptr::null_mut(),1,context_attribs.as_ptr()) };
    unsafe { sys::XSync(xdisplay,0) };

    if glx_context.is_null() { panic!("unable to open OpenGL context"); }
    if unsafe { sys::glXIsDirect(xdisplay,glx_context) } == 0 { panic!("OpenGL context is not direct"); }

    unsafe { sys::glXMakeCurrent(xdisplay,xcb_hidden_window as u64,glx_context) };

    Some(SystemGpu {
        xcb_depth,
        xcb_visual_id,
        glx_context,
        xcb_hidden_window,
    })
}

impl System {

    /// Create GPU-specific Window part.
    pub fn create_window_gpu(&self) -> Option<WindowGpu> {
        Some(WindowGpu { visible_index: Cell::new(0), })
    }

    /// Create a graphics pipeline.
    pub fn create_graphics_pipeline<T: Vertex>(
        self: &Rc<System>,
        _window: &Window,
        pipeline_layout: &Rc<PipelineLayout>,
        vertex_shader: &Rc<VertexShader>,
        fragment_shader: &Rc<FragmentShader>,
        topology: PrimitiveTopology,
        restart: PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: DepthClamp,
        primitive_discard: PrimitiveDiscard,
        polygon_mode: PolygonMode,
        cull_mode: CullMode,
        depth_bias: DepthBias,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: SampleShading,
        alpha_to_coverage: AlphaToCoverage,
        alpha_to_one: AlphaToOne,
        depth_test: DepthTest,
        depth_write_mask: bool,
        stencil_test: StencilTest,
        logic_op: LogicOp,
        blend: Blend,
        write_mask: (bool,bool,bool,bool),
        blend_constant: Color<f32>,
    ) -> Option<Rc<GraphicsPipeline>> {

        let shader_program = unsafe { sys::glCreateProgram() };
        unsafe {
            sys::glAttachShader(shader_program,vertex_shader.vs);
            sys::glAttachShader(shader_program,fragment_shader.fs);
            sys::glLinkProgram(shader_program);
        }
        let mut info_log = Vec::with_capacity(512);
        let mut success = sys::GL_FALSE as sys::GLint;
        unsafe {
            info_log.set_len(512 - 1);
            sys::glGetProgramiv(shader_program,sys::GL_LINK_STATUS,&mut success);
            sys::glGetProgramInfoLog(shader_program,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar);
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { println!("shader program errors:\n{}",str_slice); }
        if success != sys::GL_TRUE as sys::GLint { panic!("unable to link shader program"); }
        let gl_topology = match topology {
            PrimitiveTopology::Points => sys::GL_POINTS,
            PrimitiveTopology::Lines => sys::GL_LINES,
            PrimitiveTopology::LineStrip => sys::GL_LINE_STRIP,
            PrimitiveTopology::Triangles => sys::GL_TRIANGLES,
            PrimitiveTopology::TriangleStrip => sys::GL_TRIANGLE_STRIP,
            PrimitiveTopology::TriangleFan => sys::GL_TRIANGLE_FAN,
            PrimitiveTopology::LinesAdjacency => sys::GL_LINES_ADJACENCY,
            PrimitiveTopology::LineStripAdjacency => sys::GL_LINE_STRIP_ADJACENCY,
            PrimitiveTopology::TrianglesAdjacency => sys::GL_TRIANGLES_ADJACENCY,
            PrimitiveTopology::TriangleStripAdjacency => sys::GL_TRIANGLE_STRIP_ADJACENCY,
            PrimitiveTopology::Patches => sys::GL_PATCHES,
        };
        let gl_depth_clamp = if let DepthClamp::Enabled = depth_clamp { true } else { false };
        let gl_rasterizer_discard = if let PrimitiveDiscard::Enabled = primitive_discard { true } else { false };
        let gl_polygon_mode = match polygon_mode {
            PolygonMode::Point => sys::GL_POINT,
            PolygonMode::Line => sys::GL_LINE,
            PolygonMode::Fill => sys::GL_FILL,
        };
        let gl_polygon_offset = match depth_bias {
            DepthBias::Disabled => None,
            DepthBias::Enabled(c,_cl,s) => Some((s,c)),
        };
        let gl_cull_mode = match cull_mode {
            CullMode::None => None,
            CullMode::Front(front) => Some((sys::GL_FRONT,if let FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
            CullMode::Back(front) => Some((sys::GL_BACK,if let FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
            CullMode::FrontAndBack(front) => Some((sys::GL_FRONT_AND_BACK,if let FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
        };
        let gl_sample_alpha_to_coverage = if let AlphaToCoverage::Enabled = alpha_to_coverage { true } else { false };
        let gl_sample_alpha_to_one = if let AlphaToOne::Enabled = alpha_to_one { true } else { false };
        let gl_depth_test = match depth_test {
            DepthTest::Disabled => None,
            DepthTest::Enabled(op,_) => Some(match op {
                CompareOp::Never => sys::GL_NEVER,
                CompareOp::Equal => sys::GL_EQUAL,
                CompareOp::NotEqual => sys::GL_NOTEQUAL,
                CompareOp::Less => sys::GL_LESS,
                CompareOp::GreaterOrEqual => sys::GL_GEQUAL,
                CompareOp::Greater => sys::GL_GREATER,
                CompareOp::LessOrEqual => sys::GL_LEQUAL,
                CompareOp::Always => sys::GL_ALWAYS,
            }),
        };
        let gl_depth_mask = if depth_write_mask { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 };
        let gl_stencil_test = match stencil_test {
            StencilTest::Disabled => None,
            StencilTest::Enabled((fail_op,pass_op,dfail_op,comp_op,comp_mask,write_mask,fref),_) => Some((
                write_mask,
                match comp_op {
                    CompareOp::Never => sys::GL_NEVER,
                    CompareOp::Less => sys::GL_LESS,
                    CompareOp::Equal => sys::GL_EQUAL,
                    CompareOp::LessOrEqual => sys::GL_LEQUAL,
                    CompareOp::Greater => sys::GL_GREATER,
                    CompareOp::NotEqual => sys::GL_NOTEQUAL,
                    CompareOp::GreaterOrEqual => sys::GL_GEQUAL,
                    CompareOp::Always => sys::GL_ALWAYS,
                },
                fref as i32,
                comp_mask,
                match fail_op {
                    StencilOp::Keep => sys::GL_KEEP,
                    StencilOp::Zero => sys::GL_ZERO,
                    StencilOp::Replace => sys::GL_REPLACE,
                    StencilOp::IncClamp => sys::GL_INCR,
                    StencilOp::DecClamp => sys::GL_DECR,
                    StencilOp::Invert => sys::GL_INVERT,
                    StencilOp::IncWrap => sys::GL_INCR_WRAP,
                    StencilOp::DecWrap => sys::GL_DECR_WRAP,
                },
                match dfail_op {
                    StencilOp::Keep => sys::GL_KEEP,
                    StencilOp::Zero => sys::GL_ZERO,
                    StencilOp::Replace => sys::GL_REPLACE,
                    StencilOp::IncClamp => sys::GL_INCR,
                    StencilOp::DecClamp => sys::GL_DECR,
                    StencilOp::Invert => sys::GL_INVERT,
                    StencilOp::IncWrap => sys::GL_INCR_WRAP,
                    StencilOp::DecWrap => sys::GL_DECR_WRAP,
                },
                match pass_op {
                    StencilOp::Keep => sys::GL_KEEP,
                    StencilOp::Zero => sys::GL_ZERO,
                    StencilOp::Replace => sys::GL_REPLACE,
                    StencilOp::IncClamp => sys::GL_INCR,
                    StencilOp::DecClamp => sys::GL_DECR,
                    StencilOp::Invert => sys::GL_INVERT,
                    StencilOp::IncWrap => sys::GL_INCR_WRAP,
                    StencilOp::DecWrap => sys::GL_DECR_WRAP,
                },
            )),
        };
        let gl_logic_op = match logic_op {
            LogicOp::Disabled => None,
            LogicOp::Clear => Some(sys::GL_CLEAR),
            LogicOp::And => Some(sys::GL_AND),
            LogicOp::AndReverse => Some(sys::GL_AND_REVERSE),
            LogicOp::Copy => Some(sys::GL_COPY),
            LogicOp::AndInverted => Some(sys::GL_AND_INVERTED),
            LogicOp::NoOp => Some(sys::GL_NOOP),
            LogicOp::Xor => Some(sys::GL_XOR),
            LogicOp::Or => Some(sys::GL_OR),
            LogicOp::Nor => Some(sys::GL_NOR),
            LogicOp::Equivalent => Some(sys::GL_EQUIV),
            LogicOp::Invert => Some(sys::GL_INVERT),
            LogicOp::OrReverse => Some(sys::GL_OR_REVERSE),
            LogicOp::CopyInverted => Some(sys::GL_COPY_INVERTED),
            LogicOp::OrInverted => Some(sys::GL_OR_INVERTED),
            LogicOp::Nand => Some(sys::GL_NAND),
            LogicOp::Set => Some(sys::GL_SET),
        };

        let gl_blend = match blend {
            Blend::Disabled => None,
            Blend::Enabled((color_op,color_src,color_dst),(alpha_op,alpha_src,alpha_dst)) => Some((
                // TODO: what happens to alpha_op?
                match color_op {
                    BlendOp::Add => sys::GL_FUNC_ADD,
                    BlendOp::Subtract => sys::GL_FUNC_SUBTRACT,
                    BlendOp::ReverseSubtract => sys::GL_FUNC_REVERSE_SUBTRACT,
                    BlendOp::Min => sys::GL_MIN,
                    BlendOp::Max => sys::GL_MAX,
                },
                match color_src {
                    BlendFactor::Zero => sys::GL_ZERO,
                    BlendFactor::One => sys::GL_ONE,
                    BlendFactor::SrcColor => sys::GL_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::GL_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::GL_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::GL_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::GL_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::GL_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::GL_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::GL_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::GL_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::GL_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::GL_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::GL_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::GL_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::GL_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::GL_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::GL_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::GL_ONE_MINUS_SRC1_ALPHA,
                },
                match color_dst {
                    BlendFactor::Zero => sys::GL_ZERO,
                    BlendFactor::One => sys::GL_ONE,
                    BlendFactor::SrcColor => sys::GL_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::GL_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::GL_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::GL_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::GL_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::GL_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::GL_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::GL_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::GL_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::GL_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::GL_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::GL_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::GL_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::GL_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::GL_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::GL_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::GL_ONE_MINUS_SRC1_ALPHA,
                },
                match alpha_op {
                    BlendOp::Add => sys::GL_FUNC_ADD,
                    BlendOp::Subtract => sys::GL_FUNC_SUBTRACT,
                    BlendOp::ReverseSubtract => sys::GL_FUNC_REVERSE_SUBTRACT,
                    BlendOp::Min => sys::GL_MIN,
                    BlendOp::Max => sys::GL_MAX,
                },
                match alpha_src {
                    BlendFactor::Zero => sys::GL_ZERO,
                    BlendFactor::One => sys::GL_ONE,
                    BlendFactor::SrcColor => sys::GL_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::GL_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::GL_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::GL_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::GL_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::GL_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::GL_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::GL_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::GL_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::GL_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::GL_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::GL_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::GL_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::GL_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::GL_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::GL_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::GL_ONE_MINUS_SRC1_ALPHA,
                },
                match alpha_dst {
                    BlendFactor::Zero => sys::GL_ZERO,
                    BlendFactor::One => sys::GL_ONE,
                    BlendFactor::SrcColor => sys::GL_SRC_COLOR,
                    BlendFactor::OneMinusSrcColor => sys::GL_ONE_MINUS_SRC_COLOR,
                    BlendFactor::DstColor => sys::GL_DST_COLOR,
                    BlendFactor::OneMinusDstColor => sys::GL_ONE_MINUS_DST_COLOR,
                    BlendFactor::SrcAlpha => sys::GL_SRC_ALPHA,
                    BlendFactor::OneMinusSrcAlpha => sys::GL_ONE_MINUS_SRC_ALPHA,
                    BlendFactor::DstAlpha => sys::GL_DST_ALPHA,
                    BlendFactor::OneMinusDstAlpha => sys::GL_ONE_MINUS_DST_ALPHA,
                    BlendFactor::ConstantColor => sys::GL_CONSTANT_COLOR,
                    BlendFactor::OneMinusConstantColor => sys::GL_ONE_MINUS_CONSTANT_COLOR,
                    BlendFactor::ConstantAlpha => sys::GL_CONSTANT_ALPHA,
                    BlendFactor::OneMinusConstantAlpha => sys::GL_ONE_MINUS_CONSTANT_ALPHA,
                    BlendFactor::SrcAlphaSaturate => sys::GL_SRC_ALPHA_SATURATE,
                    BlendFactor::Src1Color => sys::GL_SRC1_COLOR,
                    BlendFactor::OneMinusSrc1Color => sys::GL_ONE_MINUS_SRC1_COLOR,
                    BlendFactor::Src1Alpha => sys::GL_SRC1_ALPHA,
                    BlendFactor::OneMinusSrc1Alpha => sys::GL_ONE_MINUS_SRC1_ALPHA,
                }
            )),
        };
        let gl_color_mask = (
            if write_mask.0 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.1 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.2 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.3 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
        );

        Some(Rc::new(GraphicsPipeline {
            _system: Rc::clone(self),
            _pipeline_layout: Rc::clone(pipeline_layout),
            shader_program: shader_program,
            topology: gl_topology,
            _restart: restart,
            _patch_control_points: patch_control_points,
            depth_clamp: gl_depth_clamp,
            rasterizer_discard: gl_rasterizer_discard,
            polygon_mode: gl_polygon_mode,
            cull_mode: gl_cull_mode,
            polygon_offset: gl_polygon_offset,
            line_width,
            _rasterization_samples: rasterization_samples,
            _sample_shading: sample_shading,
            sample_alpha_to_coverage: gl_sample_alpha_to_coverage,
            sample_alpha_to_one: gl_sample_alpha_to_one,
            depth_test: gl_depth_test,
            depth_mask: gl_depth_mask,
            stencil_test: gl_stencil_test,
            logic_op: gl_logic_op,
            blend: gl_blend,
            color_mask: gl_color_mask,
            blend_constant,
        }))
    }

    /// Create a pipeline layout.
    pub fn create_pipeline_layout(self: &Rc<System>) -> Option<Rc<PipelineLayout>> {

        // TODO

        Some(Rc::new(PipelineLayout {
            _system: Rc::clone(self),
        }))
    }

    /// Create command buffer.
    pub fn create_command_buffer(self: &Rc<System>) -> Option<Rc<CommandBuffer>> {

        Some(Rc::new(CommandBuffer {
            system: Rc::clone(self),
            commands: RefCell::new(Vec::new()),
        }))
    }

    /// Wait for wait_semaphore before submitting command buffer to the queue, and signal signal_semaphore when ready.
    pub fn submit(&self,command_buffer: &CommandBuffer,_wait_semaphore: &Semaphore,_signal_semaphore: &Semaphore) -> bool {

        command_buffer.execute();
        true
    }

    /// Create a vertex shader.
    pub fn create_vertex_shader(self: &Rc<System>,code: &[u8]) -> Option<Rc<VertexShader>> {
        let vs = unsafe { sys::glCreateShader(sys::GL_VERTEX_SHADER) };
        //let vcstr = CString::new(code.as_bytes()).unwrap();
        unsafe {
            //sys::glShaderSource(vs,1,&vcstr.as_ptr(),null_mut());
            sys::glShaderSource(vs,1,&code.as_ptr() as *const *const u8 as *const *const i8,null_mut());
            sys::glCompileShader(vs);
        }
        let mut success = sys::GL_FALSE as sys::GLint;
        let mut info_log = Vec::with_capacity(512);
        unsafe {
            info_log.set_len(512 - 1);
            sys::glGetShaderiv(vs,sys::GL_COMPILE_STATUS,&mut success);
            sys::glGetShaderInfoLog(vs,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar);
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { panic!("vertex shader errors:\n{}\nvertex shader source:\n{:?}",str_slice,code); }
        if success != sys::GL_TRUE as sys::GLint { panic!("unable to compile vertex shader"); }
        Some(Rc::new(VertexShader {
            system: Rc::clone(self),
            vs,
        }))
    }    

    /// Create a fragment shader.
    pub fn create_fragment_shader(self: &Rc<System>,code: &[u8]) -> Option<Rc<FragmentShader>> {
        let fs = unsafe { sys::glCreateShader(sys::GL_FRAGMENT_SHADER) };
        //let vcstr = CString::new(code.as_bytes()).unwrap();
        unsafe {
            //sys::glShaderSource(fs,1,&vcstr.as_ptr(),null_mut());
            sys::glShaderSource(fs,1,&code.as_ptr() as *const *const u8 as *const *const i8,null_mut());
            sys::glCompileShader(fs);
        }
        let mut success = sys::GL_FALSE as sys::GLint;
        let mut info_log = Vec::with_capacity(512);
        unsafe {
            info_log.set_len(512 - 1);
            sys::glGetShaderiv(fs,sys::GL_COMPILE_STATUS,&mut success);
            sys::glGetShaderInfoLog(fs,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar);
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { panic!("fragment shader errors:\n{}\nfragment shader source:\n{:?}",str_slice,code); }
        if success != sys::GL_TRUE as sys::GLint { panic!("unable to compile fragment shader"); }
        Some(Rc::new(FragmentShader {
            system: Rc::clone(self),
            fs,
        }))
    }

    /// create a vertex buffer.
    pub fn create_vertex_buffer<T: Vertex>(self: &Rc<System>,vertices: &Vec<T>) -> Option<Rc<VertexBuffer>> {

        let mut vao: sys::GLuint = 0;
        let mut vbo: sys::GLuint = 0;
        let vertex_ast = T::ast();
        //let mut i = 0usize;
        let mut size = 0i32;
        for field in vertex_ast.fields.iter() {
            match &field.type_ {
                ast::Type::Inferred => panic!("vertex field type needs to be specified"),
                ast::Type::Void => panic!("vertex field type cannot be ()"),
                ast::Type::Integer => panic!("{}","vertex field type cannot be {integer}"),
                ast::Type::Float => panic!("{}","vertex field type cannot be {float}"),
                ast::Type::Bool => panic!("vertex field type cannot be bool"),
                ast::Type::U8 |
                ast::Type::I8 => size += 1,
                ast::Type::U16 |
                ast::Type::I16 => size += 2,
                ast::Type::U32 |
                ast::Type::I32 => size += 4,
                ast::Type::U64 |
                ast::Type::I64 => size += 8,
                ast::Type::USize |
                ast::Type::ISize => size += 4,
                ast::Type::F16 => size += 2,
                ast::Type::F32 => size += 4,
                ast::Type::F64 => size += 4,
                ast::Type::AnonTuple(_) => panic!("vertex field cannot be anonymous tuple"),
                ast::Type::Array(_,_) => panic!("vertex field cannot be array"),
                ast::Type::UnknownIdent(ident) => panic!("vertex field cannot be unknown identifier {}",ident),
                ast::Type::Tuple(_) => panic!("vertex field cannot be tuple"),  // TODO
                ast::Type::Struct(_) => panic!("vertex field cannot be struct"),  // TODO
                ast::Type::Enum(_) => panic!("vertex field cannot be enum"),  // TODO
                ast::Type::Alias(_) => panic!("vertex field cannot be alias"),  // TODO
            }
        }
        println!("vertex size: {} bytes",size);
        //let mut offset = 0usize;
        unsafe {
            sys::glGenVertexArrays(1,&mut vao);
            sys::glBindVertexArray(vao);
            sys::glGenBuffers(1,&mut vbo);
            sys::glBindBuffer(sys::GL_ARRAY_BUFFER,vbo);
            //sys::glBufferData(sys::GL_ARRAY_BUFFER,1,null_mut() as *const c_void,sys::GL_STATIC_DRAW);
            for _field in vertex_ast.fields.iter() {
                /*
                if let sr::Type::Base(bt) = &field.type_ {
                    sys::glEnableVertexAttribArray(i as u32);
                    match bt {
                        sr::BaseType::U8 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::U16 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::U32 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::U64,
                        sr::BaseType::I8 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::I16 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::I32 => sys::glVertexAttribIPointer(i as u32,1,sys::GL_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::I64,
                        sr::BaseType::F16 => sys::glVertexAttribPointer(i as u32,1,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::F32 => sys::glVertexAttribPointer(i as u32,1,sys::GL_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::F64 => sys::glVertexAttribPointer(i as u32,1,sys::GL_DOUBLE,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2U8 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2U16 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2U32 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec2U64,
                        sr::BaseType::Vec2I8 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2I16 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2I32 => sys::glVertexAttribIPointer(i as u32,2,sys::GL_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec2I64,
                        sr::BaseType::Vec2F16 => sys::glVertexAttribPointer(i as u32,2,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2F32 => sys::glVertexAttribPointer(i as u32,2,sys::GL_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec2F64 => sys::glVertexAttribPointer(i as u32,2,sys::GL_DOUBLE,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3U8 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3U16 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3U32 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec3U64,
                        sr::BaseType::Vec3I8 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3I16 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3I32 => sys::glVertexAttribIPointer(i as u32,3,sys::GL_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec3I64,
                        sr::BaseType::Vec3F16 => sys::glVertexAttribPointer(i as u32,3,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3F32 => sys::glVertexAttribPointer(i as u32,3,sys::GL_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec3F64 => sys::glVertexAttribPointer(i as u32,3,sys::GL_DOUBLE,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4U8 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4U16 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4U32 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec4U64,
                        sr::BaseType::Vec4I8 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_BYTE,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4I16 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_SHORT,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4I32 => sys::glVertexAttribIPointer(i as u32,4,sys::GL_INT,size,offset as *const sys::GLvoid),
                        //sr::BaseType::Vec4I64,
                        sr::BaseType::Vec4F16 => sys::glVertexAttribPointer(i as u32,4,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4F32 => sys::glVertexAttribPointer(i as u32,4,sys::GL_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::Vec4F64 => sys::glVertexAttribPointer(i as u32,4,sys::GL_DOUBLE,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::ColorU8 => sys::glVertexAttribPointer(i as u32,4,sys::GL_UNSIGNED_BYTE,sys::GL_TRUE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::ColorU16 => sys::glVertexAttribPointer(i as u32,4,sys::GL_UNSIGNED_SHORT,sys::GL_TRUE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::ColorF16 => sys::glVertexAttribPointer(i as u32,4,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::ColorF32 => sys::glVertexAttribPointer(i as u32,4,sys::GL_FLOAT,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        sr::BaseType::ColorF64 => sys::glVertexAttribPointer(i as u32,4,sys::GL_DOUBLE,sys::GL_FALSE as u8,size,offset as *const sys::GLvoid),
                        _ => panic!("type not supported"),
                    }
                    offset += bt.size();
                    i += 1;
                }
                else {
                    panic!("vertex fields should be base type, not {}",field.type_);
                }
                */
            }
            sys::glBufferData(sys::GL_ARRAY_BUFFER,(size as usize * vertices.len()) as sys::GLsizeiptr,vertices.as_ptr() as *const c_void,sys::GL_STATIC_DRAW);
        }

        Some(Rc::new(VertexBuffer {
            system: Rc::clone(self),
            vao,
            vbo,
        }))
    }

    /// create an index buffer.
    pub fn create_index_buffer<T>(self: &Rc<System>,indices: &Vec<T>) -> Option<Rc<IndexBuffer>> {
        let mut ibo: sys::GLuint = 0;
        unsafe {
            sys::glGenBuffers(1,&mut ibo);
            sys::glBindBuffer(sys::GL_ELEMENT_ARRAY_BUFFER,ibo);
            sys::glBufferData(sys::GL_ELEMENT_ARRAY_BUFFER,(size_of::<T>() * indices.len()) as sys::GLsizeiptr,indices.as_ptr() as *const c_void,sys::GL_STATIC_DRAW);
        }
        Some(Rc::new(IndexBuffer {
            system: Rc::clone(self),
            ibo,
        }))
    }

    /// Create a semaphore.
    pub fn create_semaphore(self: &Rc<System>) -> Option<Rc<Semaphore>> {
        Some(Rc::new(Semaphore {
            system: Rc::clone(self),
        }))
    }
    
    /// Drop all GPU-specific resources.
    pub fn drop_gpu(&self) {
        unsafe {
            sys::glXMakeCurrent(self.xdisplay,0,null_mut());
            sys::xcb_unmap_window(self.xcb_connection,self.gpu.xcb_hidden_window);
            sys::xcb_destroy_window(self.xcb_connection,self.gpu.xcb_hidden_window);
            sys::glXDestroyContext(self.xdisplay,self.gpu.glx_context);
        }
    }
}
