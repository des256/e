//! Minimal PulseAudio FFI bindings for the simple API.
//!
//! Covers only the subset of `libpulse-simple` and `libpulse` used by
//! [`audioin`](crate::audioin) and [`audioout`](crate::audioout).

use std::{
    ffi::{CStr, CString},
    fmt,
    os::raw::{c_char, c_void},
    ptr::null,
};

// -- ffi --

/// Opaque handle returned by `pa_simple_new`.
#[repr(C)]
struct pa_simple {
    _private: [u8; 0],
}

#[link(name = "pulse")]
unsafe extern "C" {
    fn pa_strerror(error: i32) -> *const c_char;
}

#[link(name = "pulse-simple")]
unsafe extern "C" {
    fn pa_simple_new(
        server: *const c_char,
        name: *const c_char,
        dir: StreamDirection,
        dev: *const c_char,
        stream_name: *const c_char,
        ss: *const SampleSpec,
        map: *const c_void,
        attr: *const BufferAttr,
        error: *mut i32,
    ) -> *mut pa_simple;

    fn pa_simple_free(s: *mut pa_simple);

    fn pa_simple_write(
        s: *mut pa_simple,
        data: *const c_void,
        bytes: usize,
        error: *mut i32,
    ) -> i32;

    fn pa_simple_read(
        s: *mut pa_simple,
        data: *mut c_void,
        bytes: usize,
        error: *mut i32,
    ) -> i32;
}

// -- types --

/// Sample format.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum SampleFormat {
    /// 32-bit IEEE float, little-endian.
    F32le = 5,
    /// Invalid / sentinel value.
    Invalid = -1,
}

/// Stream direction (playback or recording).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum StreamDirection {
    /// Playback stream.
    Playback = 1,
    /// Recording stream.
    Record = 2,
}

/// Sample format specification (`pa_sample_spec`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct SampleSpec {
    /// Sample format.
    pub format: SampleFormat,
    /// Sample rate in Hz.
    pub rate: u32,
    /// Number of channels.
    pub channels: u8,
}

/// Buffer metrics (`pa_buffer_attr`).
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub(crate) struct BufferAttr {
    /// Maximum buffer length in bytes.
    pub maxlength: u32,
    /// Target buffer length (playback).
    pub tlength: u32,
    /// Pre-buffering length.
    pub prebuf: u32,
    /// Minimum request size.
    pub minreq: u32,
    /// Fragment size (recording).
    pub fragsize: u32,
}

// -- errors --

/// PulseAudio error wrapping a raw error code.
pub(crate) struct PulseError(i32);

impl fmt::Display for PulseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = unsafe { CStr::from_ptr(pa_strerror(self.0)) };
        write!(f, "{}", msg.to_string_lossy())
    }
}

impl fmt::Debug for PulseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PulseError({}, \"{}\")", self.0, self)
    }
}

impl std::error::Error for PulseError {}

// -- simple api --

/// Safe wrapper around PulseAudio's simple API connection.
pub(crate) struct Simple {
    ptr: *mut pa_simple,
}

unsafe impl Send for Simple {}
unsafe impl Sync for Simple {}

impl Simple {
    /// Open a new connection to the PulseAudio server.
    ///
    /// * `server` — server name, or `None` for default.
    /// * `name` — application name.
    /// * `dir` — [`StreamDirection::Playback`] or [`StreamDirection::Record`].
    /// * `dev` — sink/source name, or `None` for default.
    /// * `stream_name` — stream description.
    /// * `ss` — sample format specification.
    /// * `attr` — buffer attributes, or `None` for defaults.
    pub fn new(
        server: Option<&str>,
        name: &str,
        dir: StreamDirection,
        dev: Option<&str>,
        stream_name: &str,
        ss: &SampleSpec,
        attr: Option<&BufferAttr>,
    ) -> Result<Self, PulseError> {
        let c_server = server.map(|s| CString::new(s).unwrap());
        let c_dev = dev.map(|d| CString::new(d).unwrap());
        let c_name = CString::new(name).unwrap();
        let c_stream_name = CString::new(stream_name).unwrap();

        let p_server = c_server
            .as_ref()
            .map_or(null(), |s| s.as_ptr());
        let p_dev = c_dev
            .as_ref()
            .map_or(null(), |d| d.as_ptr());
        let p_attr = attr.map_or(null(), |a| a as *const BufferAttr);

        let mut error: i32 = 0;
        let ptr = unsafe {
            pa_simple_new(
                p_server,
                c_name.as_ptr(),
                dir,
                p_dev,
                c_stream_name.as_ptr(),
                ss as *const SampleSpec,
                null(),
                p_attr,
                &mut error,
            )
        };

        if ptr.is_null() {
            Err(PulseError(error))
        } else {
            Ok(Self { ptr })
        }
    }

    /// Read audio data from a recording stream.
    ///
    /// Blocks until `data.len()` bytes have been received.
    pub fn read(&self, data: &mut [u8]) -> Result<(), PulseError> {
        let mut error: i32 = 0;
        let ret = unsafe {
            pa_simple_read(
                self.ptr,
                data.as_mut_ptr() as *mut c_void,
                data.len(),
                &mut error,
            )
        };
        if ret == 0 { Ok(()) } else { Err(PulseError(error)) }
    }

    /// Write audio data to a playback stream.
    ///
    /// Blocks until all data has been written.
    pub fn write(&self, data: &[u8]) -> Result<(), PulseError> {
        let mut error: i32 = 0;
        let ret = unsafe {
            pa_simple_write(
                self.ptr,
                data.as_ptr() as *const c_void,
                data.len(),
                &mut error,
            )
        };
        if ret == 0 { Ok(()) } else { Err(PulseError(error)) }
    }
}

impl Drop for Simple {
    fn drop(&mut self) {
        unsafe { pa_simple_free(self.ptr) };
    }
}
