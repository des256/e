use {
    super::*,
    crate::gpu,
    crate::checkgl,
    std::{
        result::Result,
        rc::Rc,
        ptr::null_mut,
        ffi::{
            c_void,
            c_int,
            CStr,
        },
        cell::{
            Cell,
            RefCell,
        },
        mem::size_of,
    },
};

#[derive(Debug)]
pub struct Gpu {
    pub system: Rc<System>,
#[cfg(system="linux")]
    pub xcb_depth: u8,
#[cfg(system="linux")]
    pub xcb_visual_id: u32,
#[cfg(system="linux")]
    pub glx_context: sys::GLXContext,
#[cfg(system="linux")]
    pub xcb_hidden_window: sys::xcb_window_t,
}

impl gpu::Gpu for Gpu {

    type CommandBuffer = CommandBuffer;
    type VertexShader = VertexShader;
    type FragmentShader = FragmentShader;
    type PipelineLayout = PipelineLayout;

    fn open(system: &Rc<System>) -> Result<Rc<Self>,String> {

#[cfg(system="linux")]
        {
            // check if glX is useful
            let mut glxmaj: c_int = 0;
            let mut glxmin: c_int = 0;
            unsafe { if sys::glXQueryVersion(system.xdisplay,&mut glxmaj as *mut c_int,&mut glxmin as *mut c_int) == 0 { return Err("unable to get glX version".to_string()); } }
            if (glxmaj * 100 + glxmin) < 103 {
                return Err(format!("glX version {}.{} needs to be at least 1.3",glxmaj,glxmin));
            }

#[cfg(build="debug")]
            {
                let vendor_cstr = unsafe { CStr::from_ptr(sys::glXGetClientString(system.xdisplay,sys::GLX_VENDOR as c_int)) };
                let version_cstr = unsafe { CStr::from_ptr(sys::glXGetClientString(system.xdisplay,sys::GLX_VERSION as c_int)) };
                dprintln!("OpenGL glX vendor: {}",vendor_cstr.to_str().unwrap());
                dprintln!("OpenGL glX version: {}",version_cstr.to_str().unwrap());
            }
            
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
            let fbconfigs = unsafe { sys::glXChooseFBConfig(system.xdisplay,0,attribs.as_ptr() as *const i32,&mut fbcount as *mut c_int) };
            if fbcount == 0 { return Err("unable to find framebuffer config".to_string()); }
            let fbconfig = unsafe { *fbconfigs };

            // adjust the window creation parameters accordingly
            let visual = unsafe { sys::glXGetVisualFromFBConfig(system.xdisplay,fbconfig) };
            let xcb_depth = unsafe { *visual }.depth as u8;
            let xcb_visual_id = unsafe { *visual }.visualid as u32;

            unsafe { sys::XFree(fbconfigs as *mut c_void) };

            // create tiny window
            let xcb_hidden_window = unsafe { sys::xcb_generate_id(system.xcb_connection) };
            let values = [
                sys::XCB_EVENT_MASK_EXPOSURE
                | sys::XCB_EVENT_MASK_KEY_PRESS
                | sys::XCB_EVENT_MASK_KEY_RELEASE
                | sys::XCB_EVENT_MASK_BUTTON_PRESS
                | sys::XCB_EVENT_MASK_BUTTON_RELEASE
                | sys::XCB_EVENT_MASK_POINTER_MOTION
                | sys::XCB_EVENT_MASK_STRUCTURE_NOTIFY,
                sys::XCB_COPY_FROM_PARENT,
            ];
            unsafe {
                sys::xcb_create_window(
                    system.xcb_connection,
                    xcb_depth,
                    xcb_hidden_window as u32,
                    (*system.xcb_screen).root,
                    0,
                    0,
                    1,
                    1,
                    0,
                    sys::XCB_WINDOW_CLASS_INPUT_OUTPUT as u16,
                    sys::XCB_COPY_FROM_PARENT,
                    sys::XCB_CW_EVENT_MASK | sys::XCB_CW_COLORMAP,
                    &values as *const u32 as *const c_void
                );
                //sys::xcb_map_window(xcb_connection,xcb_hidden_window);
                sys::XSync(system.xdisplay,0);
            }

            // create glX context
            let context_attribs: [c_int; 5] = [
                sys::GLX_CONTEXT_MAJOR_VERSION_ARB as c_int, 4,
                sys::GLX_CONTEXT_MINOR_VERSION_ARB as c_int, 5,
                0,
            ];
            let glx_context = unsafe { sys::glXCreateContextAttribsARB(system.xdisplay,fbconfig,std::ptr::null_mut(),1,context_attribs.as_ptr()) };
            unsafe { sys::XSync(system.xdisplay,0) };

            if glx_context.is_null() { return Err("unable to open OpenGL context".to_string()); }
            if unsafe { sys::glXIsDirect(system.xdisplay,glx_context) } == 0 { return Err("OpenGL context is not direct".to_string()); }

            unsafe {
                sys::glXMakeCurrent(system.xdisplay,xcb_hidden_window as u64,glx_context);
                sys::glEnable(sys::GL_FRAMEBUFFER_SRGB);
            }

            Ok(Rc::new(Gpu {
                system: Rc::clone(&system),
                xcb_depth,
                xcb_visual_id,
                glx_context,
                xcb_hidden_window,
            }))
        }
    }

    fn create_surface(self: &Rc<Self>,window: &Rc<Window>,_r: Rect<i32>) -> Result<Surface,String> {
        Ok(Surface {
            gpu: Rc::clone(&self),
            window: Rc::clone(&window),
            visible_index: Cell::new(0),
        })
    }

    fn create_command_buffer(self: &Rc<Self>) -> Result<CommandBuffer,String> {
        Ok(CommandBuffer {
            gpu: Rc::clone(&self),
            commands: RefCell::new(Vec::new()),
        })
    }

    fn submit_command_buffer(&self,command_buffer: &CommandBuffer) -> Result<(),String> {
        command_buffer.execute();
        Ok(())
    }

    fn create_vertex_shader(self: &Rc<Self>,ast: &gpu::sc::Module) -> Result<VertexShader,String> {

        dprintln!("OpenGL Vertex Shader AST:\n{}",ast);
        let ast = gpu::sc::process(ast)?;
        dprintln!("OpenGL Vertex Shader AST after processing:\n{}",ast);
        let glsl = gpu::sc::glsl::emit_module(&ast,gpu::sc::ShaderStyle::Vertex)?;
        dprintln!("GLSL Vertex Shader:\n{}",glsl);

        /*
        let vs = unsafe { checkgl!(sys::glCreateShader(sys::GL_VERTEX_SHADER)) };
        unsafe {
            checkgl!(sys::glShaderSource(vs,1,&code.as_ptr() as *const *const u8 as *const *const i8,null_mut()));
            checkgl!(sys::glCompileShader(vs));
        }
        let mut success = sys::GL_FALSE as sys::GLint;
        let mut info_log = Vec::with_capacity(512);
        unsafe {
            info_log.set_len(512 - 1);
            checkgl!(sys::glGetShaderiv(vs,sys::GL_COMPILE_STATUS,&mut success));
            checkgl!(sys::glGetShaderInfoLog(vs,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar));
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { return Err(format!("vertex shader errors:\n{}\nvertex shader source:\n{:?}",str_slice,code)); }
        if success != sys::GL_TRUE as sys::GLint { return Err("unable to compile vertex shader".to_string()); }
        Ok(VertexShader {
            gpu: Rc::clone(&self),
            vs,
        })
        */

        Err("TODO: GLSL compiler".to_string())
    }

    fn create_fragment_shader(self: &Rc<Self>,ast: &gpu::sc::Module) -> Result<FragmentShader,String> {

        dprintln!("OpenGL Fragment Shader AST:\n{}",ast);
        let ast = gpu::sc::process(ast)?;
        dprintln!("OpenGL Fragment Shader AST after preparing:\n{}",ast);
        let glsl = gpu::sc::glsl::emit_module(&ast,gpu::sc::ShaderStyle::Fragment)?;
        dprintln!("GLSL Fragment Shader:\n{}",glsl);

        /*
        let fs = unsafe { checkgl!(sys::glCreateShader(sys::GL_FRAGMENT_SHADER)) };
        unsafe {
            checkgl!(sys::glShaderSource(fs,1,&code.as_ptr() as *const *const u8 as *const *const i8,null_mut()));
            checkgl!(sys::glCompileShader(fs));
        }
        let mut success = sys::GL_FALSE as sys::GLint;
        let mut info_log = Vec::with_capacity(512);
        unsafe {
            info_log.set_len(512 - 1);
            checkgl!(sys::glGetShaderiv(fs,sys::GL_COMPILE_STATUS,&mut success));
            checkgl!(sys::glGetShaderInfoLog(fs,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar));
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { return Err(format!("fragment shader errors:\n{}\nfragment shader source:\n{:?}",str_slice,code)); }
        if success != sys::GL_TRUE as sys::GLint { return Err("unable to compile fragment shader".to_string()); }
        Ok(FragmentShader {
            gpu: Rc::clone(&self),
            fs,
        })
        */
        Err("TODO: GLSL compiler".to_string())
    }

    fn create_graphics_pipeline<T: gpu::Vertex>(self: &Rc<Self>,
        _surface: &Surface,
        pipeline_layout: &Rc<PipelineLayout>,
        vertex_shader: &Rc<VertexShader>,
        fragment_shader: &Rc<FragmentShader>,
        topology: gpu::PrimitiveTopology,
        restart: gpu::PrimitiveRestart,
        patch_control_points: usize,
        depth_clamp: gpu::DepthClamp,
        primitive_discard: gpu::PrimitiveDiscard,
        polygon_mode: gpu::PolygonMode,
        cull_mode: gpu::CullMode,
        depth_bias: gpu::DepthBias,
        line_width: f32,
        rasterization_samples: usize,
        sample_shading: gpu::SampleShading,
        alpha_to_coverage: gpu::AlphaToCoverage,
        alpha_to_one: gpu::AlphaToOne,
        depth_test: gpu::DepthTest,
        depth_write_mask: bool,
        stencil_test: gpu::StencilTest,
        logic_op: gpu::LogicOp,
        blend: gpu::Blend,
        write_mask: (bool,bool,bool,bool),
        blend_constant: Vec4<f32>,
    ) -> Result<GraphicsPipeline,String> {
        let shader_program = unsafe { checkgl!(sys::glCreateProgram()) };
        unsafe {
            checkgl!(sys::glAttachShader(shader_program,vertex_shader.vs));
            checkgl!(sys::glAttachShader(shader_program,fragment_shader.fs));
            checkgl!(sys::glLinkProgram(shader_program));
        }
        let mut info_log = Vec::with_capacity(512);
        let mut success = sys::GL_FALSE as sys::GLint;
        unsafe {
            info_log.set_len(512 - 1);
            checkgl!(sys::glGetProgramiv(shader_program,sys::GL_LINK_STATUS,&mut success));
            checkgl!(sys::glGetProgramInfoLog(shader_program,512,null_mut(),info_log.as_mut_ptr() as *mut sys::GLchar));
        }
        let c_str: &CStr = unsafe { CStr::from_ptr(info_log.as_ptr()) };
        let str_slice: &str = c_str.to_str().unwrap();
        if str_slice.len() > 0 { return Err(format!("shader program errors:\n{}",str_slice)); }
        if success != sys::GL_TRUE as sys::GLint { return Err("unable to link shader program".to_string()); }
        let gl_topology = primitive_topology_to_gl_topology(topology);
        let gl_depth_clamp = if let gpu::DepthClamp::Enabled = depth_clamp { true } else { false };
        let gl_rasterizer_discard = if let gpu::PrimitiveDiscard::Enabled = primitive_discard { true } else { false };
        let gl_polygon_mode = polygon_mode_to_gl_polygon_mode(polygon_mode);
        let gl_polygon_offset = depth_bias_to_gl_polygon_offset(depth_bias);
        let gl_cull_mode = cull_mode_to_gl_cull_mode(cull_mode);
        let gl_sample_alpha_to_coverage = if let gpu::AlphaToCoverage::Enabled = alpha_to_coverage { true } else { false };
        let gl_sample_alpha_to_one = if let gpu::AlphaToOne::Enabled = alpha_to_one { true } else { false };
        let gl_depth_test = depth_test_to_gl_depth_test(depth_test);
        let gl_depth_mask = if depth_write_mask { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 };
        let gl_stencil_test = match stencil_test {
            gpu::StencilTest::Disabled => None,
            gpu::StencilTest::Enabled((fail_op,pass_op,dfail_op,comp_op,comp_mask,write_mask,fref),_) => Some((
                write_mask,
                compare_op_to_gl_compare_op(comp_op),
                fref as i32,
                comp_mask,
                stencil_op_to_gl_stencil_op(fail_op),
                stencil_op_to_gl_stencil_op(dfail_op),
                stencil_op_to_gl_stencil_op(pass_op),
            )),
        };
        let gl_logic_op = logic_op_to_gl_logic_op(logic_op);
        let gl_blend = match blend {
            gpu::Blend::Disabled => None,
            gpu::Blend::Enabled((color_op,color_src,color_dst),(alpha_op,alpha_src,alpha_dst)) => Some((
                // TODO: what happens to alpha_op?
                blend_op_to_gl_blend_op(color_op),
                blend_factor_to_gl_blend_factor(color_src),
                blend_factor_to_gl_blend_factor(color_dst),
                blend_op_to_gl_blend_op(alpha_op),
                blend_factor_to_gl_blend_factor(alpha_src),
                blend_factor_to_gl_blend_factor(alpha_dst),
            )),
        };
        let gl_color_mask = (
            if write_mask.0 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.1 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.2 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
            if write_mask.3 { sys::GL_TRUE as u8 } else { sys::GL_FALSE as u8 },
        );

        Ok(GraphicsPipeline {
            gpu: Rc::clone(&self),
            pipeline_layout: Rc::clone(&pipeline_layout),
            shader_program: shader_program,
            topology: gl_topology,
            restart,
            patch_control_points,
            depth_clamp: gl_depth_clamp,
            rasterizer_discard: gl_rasterizer_discard,
            polygon_mode: gl_polygon_mode,
            cull_mode: gl_cull_mode,
            polygon_offset: gl_polygon_offset,
            line_width,
            rasterization_samples,
            sample_shading,
            sample_alpha_to_coverage: gl_sample_alpha_to_coverage,
            sample_alpha_to_one: gl_sample_alpha_to_one,
            depth_test: gl_depth_test,
            depth_mask: gl_depth_mask,
            stencil_test: gl_stencil_test,
            logic_op: gl_logic_op,
            blend: gl_blend,
            color_mask: gl_color_mask,
            blend_constant,        
        })
    }

    fn create_vertex_buffer<T: gpu::Vertex>(self: &Rc<Self>,vertices: &Vec<T>) -> Result<VertexBuffer,String> {
        use gpu::sc::*;

        let mut vao: sys::GLuint = 0;
        let mut vbo: sys::GLuint = 0;

        let vertex_struct = T::ast();
        let mut size = 0usize;
        for field in vertex_struct.fields.iter() {
            size += gpu::type_to_size(&field.1)?;
        }

        unsafe {
            checkgl!(sys::glGenVertexArrays(1,&mut vao));
            checkgl!(sys::glBindVertexArray(vao));
            checkgl!(sys::glGenBuffers(1,&mut vbo));
            checkgl!(sys::glBindBuffer(sys::GL_ARRAY_BUFFER,vbo));
            checkgl!(sys::glBufferData(sys::GL_ARRAY_BUFFER,(size * vertices.len()) as sys::GLsizeiptr,vertices.as_ptr() as *const c_void,sys::GL_STATIC_DRAW));
        }

        let mut offset: usize = 0;
        for i in 0..vertex_struct.fields.len() {
            unsafe { checkgl!(sys::glEnableVertexAttribArray(i as u32)) };
            let field = &vertex_struct.fields[i];
            match &field.1 {
                Type::Bool => { return Err("TODO: bool vertex field".to_string()); },
                Type::U8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::I8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::U16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::I16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::U32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_UNSIGNED_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::I32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,1,sys::GL_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::U64 => { return Err("Vertex field cannot be u64 for OpenGL".to_string()); },
                Type::I64 => { return Err("Vertex field cannot be i64 for OpenGL".to_string()); },
                Type::F16 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,1,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::F32 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,1,sys::GL_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::F64 => unsafe { checkgl!(sys::glVertexAttribLPointer(i as u32,1,sys::GL_DOUBLE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2Bool => { return Err("TODO: Vec2<bool> vertex field".to_string()); },
                Type::Vec2U8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2I8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2U16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2I16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2U32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_UNSIGNED_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2I32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,2,sys::GL_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2U64 => { return Err("Vertex field cannot be Vec2<u64> for OpenGL".to_string()); },
                Type::Vec2I64 => { return Err("Vertex field cannot be Vec2<i64> for OpenGL".to_string()); },
                Type::Vec2F16 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,2,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2F32 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,2,sys::GL_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec2F64 => unsafe { checkgl!(sys::glVertexAttribLPointer(i as u32,2,sys::GL_DOUBLE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3Bool => { return Err("TODO: Vec2<bool> vertex field".to_string()); },
                Type::Vec3U8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3I8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3U16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3I16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3U32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_UNSIGNED_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3I32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,3,sys::GL_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3U64 => { return Err("Vertex field cannot be Vec3<u64> for OpenGL".to_string()); },
                Type::Vec3I64 => { return Err("Vertex field cannot be Vec3<i64> for OpenGL".to_string()); },
                Type::Vec3F16 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,3,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3F32 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,3,sys::GL_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec3F64 => unsafe { checkgl!(sys::glVertexAttribLPointer(i as u32,3,sys::GL_DOUBLE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4Bool => { return Err("TODO: Vec2<bool> vertex field".to_string()); },
                Type::Vec4U8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4I8 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_BYTE,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4U16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4I16 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_SHORT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4U32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_UNSIGNED_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4I32 => unsafe { checkgl!(sys::glVertexAttribIPointer(i as u32,4,sys::GL_INT,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4U64 => { return Err("Vertex field cannot be Vec4<u64> for OpenGL".to_string()); },
                Type::Vec4I64 => { return Err("Vertex field cannot be Vec4<i64> for OpenGL".to_string()); },
                Type::Vec4F16 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,4,sys::GL_HALF_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4F32 => unsafe { checkgl!(sys::glVertexAttribPointer(i as u32,4,sys::GL_FLOAT,sys::GL_FALSE as u8,size as i32,offset as *const sys::GLvoid)); },
                Type::Vec4F64 => unsafe { checkgl!(sys::glVertexAttribLPointer(i as u32,4,sys::GL_DOUBLE,size as i32,offset as *const sys::GLvoid)); },
                _ => { return Err(format!("Vertex field cannot be {}",field.1)); },
            }
            offset += gpu::type_to_size(&field.1)?;
        }

        Ok(VertexBuffer {
            gpu: Rc::clone(&self),
            vao,
            vbo,
        })
    }

    fn create_index_buffer<T>(self: &Rc<Self>,indices: &Vec<T>) -> Result<IndexBuffer,String> {
        let mut ibo: sys::GLuint = 0;
        unsafe {
            checkgl!(sys::glGenBuffers(1,&mut ibo));
            checkgl!(sys::glBindBuffer(sys::GL_ELEMENT_ARRAY_BUFFER,ibo));
            checkgl!(sys::glBufferData(sys::GL_ELEMENT_ARRAY_BUFFER,(size_of::<T>() * indices.len()) as sys::GLsizeiptr,indices.as_ptr() as *const c_void,sys::GL_STATIC_DRAW));
        }
        Ok(IndexBuffer {
            gpu: Rc::clone(&self),
            ibo,
        })
    }

    fn create_pipeline_layout(self: &Rc<Self>) -> Result<Self::PipelineLayout,String> {
        Ok(PipelineLayout {
            gpu: Rc::clone(&self),
        })
    }
}

fn primitive_topology_to_gl_topology(topology: gpu::PrimitiveTopology) -> sys::GLenum {
    match topology {
        gpu::PrimitiveTopology::Points => sys::GL_POINTS,
        gpu::PrimitiveTopology::Lines => sys::GL_LINES,
        gpu::PrimitiveTopology::LineStrip => sys::GL_LINE_STRIP,
        gpu::PrimitiveTopology::Triangles => sys::GL_TRIANGLES,
        gpu::PrimitiveTopology::TriangleStrip => sys::GL_TRIANGLE_STRIP,
        gpu::PrimitiveTopology::TriangleFan => sys::GL_TRIANGLE_FAN,
        gpu::PrimitiveTopology::LinesAdjacency => sys::GL_LINES_ADJACENCY,
        gpu::PrimitiveTopology::LineStripAdjacency => sys::GL_LINE_STRIP_ADJACENCY,
        gpu::PrimitiveTopology::TrianglesAdjacency => sys::GL_TRIANGLES_ADJACENCY,
        gpu::PrimitiveTopology::TriangleStripAdjacency => sys::GL_TRIANGLE_STRIP_ADJACENCY,
        gpu::PrimitiveTopology::Patches => sys::GL_PATCHES,
    }
}

fn polygon_mode_to_gl_polygon_mode(polygon_mode: gpu::PolygonMode) -> sys::GLenum {
    match polygon_mode {
        gpu::PolygonMode::Point => sys::GL_POINT,
        gpu::PolygonMode::Line => sys::GL_LINE,
        gpu::PolygonMode::Fill => sys::GL_FILL,
    }
}

fn depth_bias_to_gl_polygon_offset(depth_bias: gpu::DepthBias) -> Option<(f32,f32)> {
    match depth_bias {
        gpu::DepthBias::Disabled => None,
        gpu::DepthBias::Enabled(c,_cl,s) => Some((s,c)),
    }
}

fn cull_mode_to_gl_cull_mode(cull_mode: gpu::CullMode) -> Option<(sys::GLenum,sys::GLenum)> {
    match cull_mode {
        gpu::CullMode::None => None,
        gpu::CullMode::Front(front) => Some((sys::GL_FRONT,if let gpu::FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
        gpu::CullMode::Back(front) => Some((sys::GL_BACK,if let gpu::FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
        gpu::CullMode::FrontAndBack(front) => Some((sys::GL_FRONT_AND_BACK,if let gpu::FrontFace::Clockwise = front { sys::GL_CW } else { sys::GL_CCW })),
    }
}

fn depth_test_to_gl_depth_test(depth_test: gpu::DepthTest) -> Option<sys::GLenum> {
    match depth_test {
        gpu::DepthTest::Disabled => None,
        gpu::DepthTest::Enabled(op,_) => Some(match op {
            gpu::CompareOp::Never => sys::GL_NEVER,
            gpu::CompareOp::Equal => sys::GL_EQUAL,
            gpu::CompareOp::NotEqual => sys::GL_NOTEQUAL,
            gpu::CompareOp::Less => sys::GL_LESS,
            gpu::CompareOp::GreaterOrEqual => sys::GL_GEQUAL,
            gpu::CompareOp::Greater => sys::GL_GREATER,
            gpu::CompareOp::LessOrEqual => sys::GL_LEQUAL,
            gpu::CompareOp::Always => sys::GL_ALWAYS,
        }),
    }
}

fn compare_op_to_gl_compare_op(compare_op: gpu::CompareOp) -> sys::GLenum {
    match compare_op {
        gpu::CompareOp::Never => sys::GL_NEVER,
        gpu::CompareOp::Less => sys::GL_LESS,
        gpu::CompareOp::Equal => sys::GL_EQUAL,
        gpu::CompareOp::LessOrEqual => sys::GL_LEQUAL,
        gpu::CompareOp::Greater => sys::GL_GREATER,
        gpu::CompareOp::NotEqual => sys::GL_NOTEQUAL,
        gpu::CompareOp::GreaterOrEqual => sys::GL_GEQUAL,
        gpu::CompareOp::Always => sys::GL_ALWAYS,
    }
}

fn stencil_op_to_gl_stencil_op(stencil_op: gpu::StencilOp) -> sys::GLenum {
    match stencil_op {
        gpu::StencilOp::Keep => sys::GL_KEEP,
        gpu::StencilOp::Zero => sys::GL_ZERO,
        gpu::StencilOp::Replace => sys::GL_REPLACE,
        gpu::StencilOp::IncClamp => sys::GL_INCR,
        gpu::StencilOp::DecClamp => sys::GL_DECR,
        gpu::StencilOp::Invert => sys::GL_INVERT,
        gpu::StencilOp::IncWrap => sys::GL_INCR_WRAP,
        gpu::StencilOp::DecWrap => sys::GL_DECR_WRAP,
    }
}

fn logic_op_to_gl_logic_op(logic_op: gpu::LogicOp) -> Option<sys::GLenum> {
    match logic_op {
        gpu::LogicOp::Disabled => None,
        gpu::LogicOp::Clear => Some(sys::GL_CLEAR),
        gpu::LogicOp::And => Some(sys::GL_AND),
        gpu::LogicOp::AndReverse => Some(sys::GL_AND_REVERSE),
        gpu::LogicOp::Copy => Some(sys::GL_COPY),
        gpu::LogicOp::AndInverted => Some(sys::GL_AND_INVERTED),
        gpu::LogicOp::NoOp => Some(sys::GL_NOOP),
        gpu::LogicOp::Xor => Some(sys::GL_XOR),
        gpu::LogicOp::Or => Some(sys::GL_OR),
        gpu::LogicOp::Nor => Some(sys::GL_NOR),
        gpu::LogicOp::Equivalent => Some(sys::GL_EQUIV),
        gpu::LogicOp::Invert => Some(sys::GL_INVERT),
        gpu::LogicOp::OrReverse => Some(sys::GL_OR_REVERSE),
        gpu::LogicOp::CopyInverted => Some(sys::GL_COPY_INVERTED),
        gpu::LogicOp::OrInverted => Some(sys::GL_OR_INVERTED),
        gpu::LogicOp::Nand => Some(sys::GL_NAND),
        gpu::LogicOp::Set => Some(sys::GL_SET),
    }
}

fn blend_op_to_gl_blend_op(blend_op: gpu::BlendOp) -> sys::GLenum {
    match blend_op {
        gpu::BlendOp::Add => sys::GL_FUNC_ADD,
        gpu::BlendOp::Subtract => sys::GL_FUNC_SUBTRACT,
        gpu::BlendOp::ReverseSubtract => sys::GL_FUNC_REVERSE_SUBTRACT,
        gpu::BlendOp::Min => sys::GL_MIN,
        gpu::BlendOp::Max => sys::GL_MAX,
    }
}

fn blend_factor_to_gl_blend_factor(blend_factor: gpu::BlendFactor) -> sys::GLenum {
    match blend_factor {
        gpu::BlendFactor::Zero => sys::GL_ZERO,
        gpu::BlendFactor::One => sys::GL_ONE,
        gpu::BlendFactor::SrcColor => sys::GL_SRC_COLOR,
        gpu::BlendFactor::OneMinusSrcColor => sys::GL_ONE_MINUS_SRC_COLOR,
        gpu::BlendFactor::DstColor => sys::GL_DST_COLOR,
        gpu::BlendFactor::OneMinusDstColor => sys::GL_ONE_MINUS_DST_COLOR,
        gpu::BlendFactor::SrcAlpha => sys::GL_SRC_ALPHA,
        gpu::BlendFactor::OneMinusSrcAlpha => sys::GL_ONE_MINUS_SRC_ALPHA,
        gpu::BlendFactor::DstAlpha => sys::GL_DST_ALPHA,
        gpu::BlendFactor::OneMinusDstAlpha => sys::GL_ONE_MINUS_DST_ALPHA,
        gpu::BlendFactor::ConstantColor => sys::GL_CONSTANT_COLOR,
        gpu::BlendFactor::OneMinusConstantColor => sys::GL_ONE_MINUS_CONSTANT_COLOR,
        gpu::BlendFactor::ConstantAlpha => sys::GL_CONSTANT_ALPHA,
        gpu::BlendFactor::OneMinusConstantAlpha => sys::GL_ONE_MINUS_CONSTANT_ALPHA,
        gpu::BlendFactor::SrcAlphaSaturate => sys::GL_SRC_ALPHA_SATURATE,
        gpu::BlendFactor::Src1Color => sys::GL_SRC1_COLOR,
        gpu::BlendFactor::OneMinusSrc1Color => sys::GL_ONE_MINUS_SRC1_COLOR,
        gpu::BlendFactor::Src1Alpha => sys::GL_SRC1_ALPHA,
        gpu::BlendFactor::OneMinusSrc1Alpha => sys::GL_ONE_MINUS_SRC1_ALPHA,
    }
}
