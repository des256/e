use {
    std::{
        env,
        fs,
        process,
        path,
    },
};

pub enum System {
    Linux,
    Windows,
    Macos,
    Android,
    Ios,
    Web,
}

fn main() {

    // make current build available to source code
    println!("cargo:rustc-cfg=build={:?}",env::var("PROFILE").unwrap());

    // define system and gpu configurations
#[cfg(target_os="linux")]
    let (system,system_name) = (System::Linux,"linux");  // Vulkan, Opengl, Gles
#[cfg(target_os="windows")]
    let (system,system_name) = (System::Windows,"windows");  // Vulkan, Opengl, Directx
#[cfg(target_os="macos")]
    let (system,system_name) = (System::Macos,"macos");  // Vulkan, Metal
#[cfg(target_os="android")]
    let (system,system_name) = (System::Android,"android");  // Vulkan, Gles
#[cfg(target_os="ios")]
    let (system,system_name) = (System::Ios,"ios");  // Vulkan, Metal
#[cfg(target_family="wasm")]
    let (system,system_name) = (System::Web,"web");  // Webgl, Webgpu
    println!("cargo:rustc-cfg=system=\"{}\"",system_name);

    // create header files and system bindings, but not for web
    if let System::Web = system { } else {

        // sys.h: the needed includes
        let header_path = path::Path::new("src/sys/sys.h");
        let mut header = String::new();
        match system {
            System::Linux => {
                header.push_str("#include <sys/epoll.h>\n");
                header.push_str("#include <X11/Xlib.h>\n");
                header.push_str("#include <X11/Xlib-xcb.h>\n");
                header.push_str("#include <xcb/xcb.h>\n");    
                println!("cargo:rustc-link-lib=X11");
                println!("cargo:rustc-link-lib=X11-xcb");
                println!("cargo:rustc-link-lib=xcb");
            },
            _ => {
                panic!("missing include/lib for system=\"{}\"",system_name);
            },
        }

// Vulkan
#[cfg(any(target_os="linux",target_os="windows",target_os="macos",target_os="android",target_os="ios"))]
        {
            println!("cargo:rustc-cfg=vulkan");
            println!("cargo:rustc-link-lib=vulkan");
            header.push_str("#include <vulkan/vulkan.h>\n");
            if let System::Linux = system {
                header.push_str("#include <vulkan/vulkan_xcb.h>\n");
            }
        }

// OpenGL
#[cfg(any(target_os="linux",target_os="windows",target_os="macos"))]
        {
            println!("cargo:rustc-cfg=opengl");
            println!("cargo:rustc-link-lib=GL");
            if let System::Linux = system {
                header.push_str("#define GL_GLEXT_PROTOTYPES 1\n");
                header.push_str("#include <GL/glcorearb.h>\n");
                header.push_str("#define GLX_GLXEXT_PROTOTYPES 1\n");
                header.push_str("#include <GL/glx.h>\n");
                header.push_str("#include <GL/glxext.h>\n");
            }
            else {
                header.push_str("#include <GL/gl.h>\n");
            }
        }

// OpenGL ES
#[cfg(any(target_os="linux",target_os="windows",target_os="macos",target_os="android"))]
        {
        }

// Metal

        // sys.rs: the generated bindings
        fs::write(&header_path,header).expect("Unable to write header file");
        process::Command::new("bindgen")
            .args(&[
                &format!("src/sys/sys.h"),
                "-o",&format!("src/sys/sys.rs"),
                "--disable-nested-struct-naming",
                "--no-prepend-enum-name",
                "--no-layout-tests",
            ])
            .status()
            .expect("unable to generate system FFI bindings");

        // mod.rs: the sys module
        let sysmod_path = path::Path::new("src/sys/mod.rs");
        let mut sysmod = String::new();
        sysmod.push_str("#![allow(non_camel_case_types)]\n");
        sysmod.push_str("#![allow(non_upper_case_globals)]\n");
        sysmod.push_str("#![allow(non_snake_case)]\n");
        sysmod.push_str("#![allow(dead_code)]\n\n");
        sysmod.push_str("include!(\"sys.rs\");\n");
        fs::write(&sysmod_path,sysmod).expect("Unable to write module file");
    }

    // and indicate to rerun only if changed
    println!("cargo:rerun-if-changed=build.rs");
}
