// E - GPU (OpenGL 4.5) - Graphics
// Desmond Germans, 2020

use crate::*;
use std::{
    rc::Rc,
    cell::Cell,
    ffi::CString,
    ptr::null_mut,
};
use gl::types::{
    GLuint,
    GLchar,
    GLenum,
};
#[cfg(target_os="linux")]
use {
    x11::glx::{
        glXMakeCurrent,
        glXSwapBuffers,
    },
    xcb::xproto::get_geometry,
};
#[cfg(target_os="windows")]
use {
    winapi::{
        um::wingdi::{
            wglMakeCurrent,
            SwapBuffers,
        },
    },
};

/// Graphics context.
pub struct Graphics {
    system: Rc<System>,
    pub(crate) sp: Cell<GLuint>,
    pub(crate) index_type: Cell<GLenum>,
}

/// Helper trait for Graphics::bind_texture (for various texture types).
pub trait GPUBindTexture {
    fn do_bind(&self,graphics: &Graphics,stage: usize);
}

impl GPUBindTexture for Framebuffer {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_2D,self.tex);
        }
    }
}

impl<T: GPUTextureFormat> GPUBindTexture for Texture1D<T> {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_1D,self.core.tex);
        }
    }
}

impl<T: GPUTextureFormat> GPUBindTexture for Texture2D<T> {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_2D,self.core.tex);
        }
    }
}

impl<T: GPUTextureFormat> GPUBindTexture for Texture3D<T> {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_3D,self.core.tex);
        }
    }
}

impl<T: GPUTextureFormat> GPUBindTexture for TextureCube<T> {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_CUBE_MAP,self.tex);
        }
    }
}

impl<T: GPUTextureFormat> GPUBindTexture for Texture2DArray<T> {
    fn do_bind(&self,_graphics: &Graphics,stage: usize) {
        unsafe {
            gl::ActiveTexture(gl::TEXTURE0 + stage as u32);
            gl::BindTexture(gl::TEXTURE_2D_ARRAY,self.tex);
        }
    }
}

/// Helper trait for Graphics::bind_target (for Texture2D and Framebuffer).
pub trait GPUBindTarget {
    fn do_bind(&self,graphics: &Graphics);
}

impl GPUBindTarget for Framebuffer {
    fn do_bind(&self,graphics: &Graphics) {
        unsafe {
#[cfg(target_os="linux")]
            glXMakeCurrent(graphics.system.connection.get_raw_dpy(),graphics.system.hidden_window,graphics.system.context);
#[cfg(target_os="windows")]
            wglMakeCurrent(graphics.system.hidden_hdc(),graphics.system.hglrc());
            gl::BindFramebuffer(gl::FRAMEBUFFER,self.fbo);
            gl::Viewport(0,0,self.size.x as i32,self.size.y as i32);
            gl::Scissor(0,0,self.size.x as i32,self.size.y as i32);
        }
    }
}

impl GPUBindTarget for Rc<Window> {
#[allow(unused_variables)]
    fn do_bind(&self,graphics: &Graphics) {
        unsafe {
#[cfg(target_os="linux")]
            glXMakeCurrent(graphics.system.connection.get_raw_dpy(),self.id,graphics.system.context);
#[cfg(target_os="windows")]
            wglMakeCurrent(self.hdc,self.anchor.hglrc);
            gl::BindFramebuffer(gl::FRAMEBUFFER,0);

#[cfg(target_os="linux")]
            let r = {
                let geometry_com = get_geometry(&graphics.system.connection,self.id as u32);
                let geometry = match geometry_com.get_reply() {
                    Ok(geometry) => *geometry.ptr,
                    Err(_) => { return; },
                };
                rect!(
                    geometry.x as i32,
                    geometry.y as i32,
                    geometry.width as i32,
                    geometry.height as i32
                )
            };
#[cfg(target_os="windows")]
            let r = rect!(0i32,0i32,0i32,0i32);
            
            gl::Viewport(0,0,r.s.x,r.s.y);
            gl::Scissor(0,0,r.s.x,r.s.y);
        }
    }
}

impl Graphics {
    /// Create new graphics context.
    /// 
    /// **Arguments**
    /// 
    /// * `system` - System to create the graphics context for.
    /// 
    /// **Returns**
    /// 
    /// * `Ok(GPU)` - The created graphics context.
    /// * `Err(SystemError)` - The graphics context could not be created.
    pub fn new(system: &Rc<System>) -> Result<Rc<Graphics>,SystemError> {
        Ok(Rc::new(Graphics {
            system: Rc::clone(system),
            sp: Cell::new(0),
            index_type: Cell::new(gl::UNSIGNED_INT),
        }))
    }

    /// (temporary) Bind current target.
    /// 
    /// **Arguments**
    /// 
    /// * `target` - Framebuffer or window to draw to.
    pub fn bind_target<T: GPUBindTarget>(&self,target: &T) {
        target.do_bind(&self);
    }

    /// (temporary) Flush currently pending graphics.
    pub fn flush(&self) {
        unsafe { gl::Flush(); }
    }

    /// (temporary) Clear current target.
    /// 
    /// **Arguments**
    /// 
    /// * `color` - Color to clear with.
    pub fn clear<T: ColorParameter>(&self,color: T) {
        let color = color.as_vec4();
        unsafe {
            gl::ClearColor(color.x,color.y,color.z,color.w);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }
    }

    /// (temporary) Draw points.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_points(&self,n: i32) {
        unsafe { gl::DrawArrays(gl::POINTS,0,n) };
    }

    /// (temporary) Draw triangle fan.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_triangle_fan(&self,n: i32) {
        unsafe { gl::DrawArrays(gl::TRIANGLE_FAN,0,n) };
    }

    /// (temporary) Draw triangles.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_triangles(&self,n: i32) {
        unsafe { gl::DrawArrays(gl::TRIANGLES,0,n) };
    }

    /// (temporary) Draw points.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_indexed_points(&self,n: i32) {
        unsafe { gl::DrawElements(gl::POINTS,n,self.index_type.get(),null_mut()) };
    }

    /// (temporary) Draw indexed triangle fan.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_indexed_triangle_fan(&self,n: i32) {
        unsafe { gl::DrawElements(gl::TRIANGLE_FAN,n,self.index_type.get(),null_mut()) };
    }

    /// (temporary) Draw triangles.
    /// 
    /// **Arguments**
    /// 
    /// * `n` - Number of vertices.
    pub fn draw_indexed_triangles(&self,n: i32) {
        unsafe { gl::DrawElements(gl::TRIANGLES,n,self.index_type.get(),null_mut()) };
    }

    /// (temporary) Draw instanced points.
    /// 
    /// **Arguments**
    /// 
    /// * `nv` - Number of vertices per instance.
    /// * `ni` - Number of instances.
    pub fn draw_instanced_points(&self,nv: i32, ni: i32) {
        unsafe { gl::DrawArraysInstanced(gl::POINTS,0,nv,ni) };
    }

    /// (temporary) Draw instanced triangles.
    /// 
    /// **Arguments**
    /// 
    /// * `nv` - Number of vertices per instance.
    /// * `ni` - Number of instances.
    pub fn draw_instanced_triangles(&self,nv: i32, ni: i32) {
        unsafe { gl::DrawArraysInstanced(gl::TRIANGLES,0,nv,ni) };
    }

    /// (temporary) Draw instanced triangle fan.
    /// 
    /// **Arguments**
    /// 
    /// * `nv` - Number of vertices per instance.
    /// * `ni` - Number of instances.
    pub fn draw_instanced_triangle_fan(&self,nv: i32, ni: i32) {
        unsafe { gl::DrawArraysInstanced(gl::TRIANGLE_FAN,0,nv,ni) };
    }

    /// (temporary) Set blending mode.
    /// 
    /// **Arguments**
    /// 
    /// * `mode` - Blending mode.
    pub fn set_blend(&self,mode: BlendMode) {
        match mode {
            BlendMode::Replace => unsafe { gl::Disable(gl::BLEND); },
            _ => unsafe { gl::Enable(gl::BLEND); },
        }
        match mode {
            BlendMode::Over => unsafe { gl::BlendFunc(gl::SRC_ALPHA,gl::ONE_MINUS_SRC_ALPHA); },
            _ => { },
        }
    }

    /// (temporary) Bind texture or framebuffer to a texture stage.
    /// 
    /// **Arguments**
    /// 
    /// * `stage` - Texture stage to bind to.
    /// * `texture` - Texture or framebuffer to bind.
    pub fn bind_texture<T: GPUBindTexture>(&self,stage: usize,texture: &T) {
        texture.do_bind(&self,stage);
    }

    /// (temporary) Bind current shader program.
    /// 
    /// **Arguments**
    /// 
    /// * `shader` - Shader program.
    pub fn bind_shader(&self,shader: &Shader) {
        unsafe { gl::UseProgram(shader.sp); }
        self.sp.set(shader.sp);
    }

    /// (temporary) Set uniform value for current shader program.
    /// 
    /// **Arguments**
    /// 
    /// * `name` - Variable name referenced in the shader program.
    /// * `value` - Value of the uniform.
    pub fn set_uniform<T: GPUUniformFormat>(&self,name: &str,value: T) {
        let cname = CString::new(name).unwrap();
        let res = unsafe { gl::GetUniformLocation(self.sp.get(),cname.as_ptr() as *const GLchar) };
        T::set(res,value);
    }

    /// (temporary) Bind current vertex buffer.
    /// 
    /// **Arguments**
    /// 
    /// * `vertexbuffer` - Vertexbuffer to bind.
    pub fn bind_vertexbuffer<T: GPUVertexFormat>(&self,vertexbuffer: &VertexBuffer<T>) {
        unsafe { gl::BindVertexArray(vertexbuffer.vao) };
    }

    /// (temporary) Bind current index buffer.
    /// 
    /// **Arguments**
    /// 
    /// * `indexbuffer` - Indexbuffer to bind.
    pub fn bind_indexbuffer<T: GPUIndexFormat>(&self,indexbuffer: &IndexBuffer<T>) {
        self.index_type.set(T::gl_type());
        unsafe { gl::BindBuffer(gl::ELEMENT_ARRAY_BUFFER,indexbuffer.ibo) };
    }

    /// (temporary) Bind uniform buffer.
    /// 
    /// **Arguments**
    /// 
    /// * `bp` - Uniform buffer binding point.
    /// * `name` - Name of the uniform block in the shader.
    /// * `uniformbuffer` - Uniform buffer.
    pub fn bind_uniformbuffer<T: GPUUniformFormat>(&self,bp: u32,name: &str,uniformbuffer: &UniformBuffer<T>) {
        let cname = CString::new(name).unwrap();
        unsafe {
            let res = gl::GetUniformBlockIndex(self.sp.get(),cname.as_ptr() as *const GLchar);
            gl::UniformBlockBinding(self.sp.get(),res,bp);
            gl::BindBufferBase(gl::UNIFORM_BUFFER,bp,uniformbuffer.ubo);
        }
    }

    /// (temporary) Enable/Disable VSync.
    /// 
    /// **Arguments**
    /// 
    /// * `window` - Window to set VSync for.
    /// * `state` - Whether or not VSync should be enabled.
    #[allow(unused_variables)]
    pub fn set_vsync(&self,window: &Window,state: bool) {
        unsafe {
    #[cfg(target_os="linux")]
            (self.system.glx_swap_interval)(self.system.connection.get_raw_dpy(),window.id,if state { 1 } else { 0 });
    #[cfg(target_os="windows")]
            (self.system.wgl_swap_interval)(if state { 1 } else { 0 });
        }
    }

    /// (temporary) Present target.
    /// 
    /// **Arguments**
    /// 
    /// * `id` - Unique window ID to present to.
    #[allow(unused_variables)]
    pub fn present(&self,id: u64) {
        unsafe {
    #[cfg(target_os="linux")]
            glXSwapBuffers(self.system.connection.get_raw_dpy(),id);
    #[cfg(target_os="windows")]
            SwapBuffers(window.hdc);
        }
    }
}