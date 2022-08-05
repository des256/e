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

pub enum Gpu {
    Vulkan,
    Opengl,
    Gles,
    Directx,
    Metal,
    Webgl,
    Webgpu,
}

fn main() {

    // make current build available to source code
    println!("cargo:rustc-cfg=build={:?}",env::var("PROFILE").unwrap());

    // define system and gpu configurations
#[cfg(target_os="linux")]
    let (system,gpu) = (System::Linux,Gpu::Vulkan);  // Vulkan, Opengl, Gles
#[cfg(target_os="windows")]
    let (system,gpu) = (System::Windows,Gpu::Vulkan);  // Vulkan, Opengl, Directx
#[cfg(target_os="macos")]
    let (system,gpu) = (System::Macos,Gpu::Vulkan);  // Vulkan, Metal
#[cfg(target_os="android")]
    let (system,gpu) = (System::Android,Gpu::Vulkan);  // Vulkan, Gles
#[cfg(target_os="ios")]
    let (system,gpu) = (System::Ios,Gpu::Vulkan);  // Vulkan, Metal
#[cfg(target_family="wasm")]
    let (system,gpu) = (System::Web,Gpu::Webgl);  // Webgl, Webgpu
    let system_name = match system {
        System::Linux => "linux",
        System::Windows => "windows",
        System::Macos => "macos",
        System::Android => "android",
        System::Ios => "ios",
        System::Web => "web",
    };
    let gpu_name = match gpu {
        Gpu::Vulkan => "vulkan",
        Gpu::Opengl => "opengl",
        Gpu::Gles => "gles",
        Gpu::Directx => "directx",
        Gpu::Metal => "metal",
        Gpu::Webgl => "webgl",
        Gpu::Webgpu => "webgpu",    
    };
    println!("cargo:rustc-cfg=system=\"{}\"",system_name);
    println!("cargo:rustc-cfg=gpu=\"{}\"",gpu_name);

    // create header file, but not for web
    if let System::Web = system { } else {
        let out_dir_os = env::var_os("OUT_DIR").unwrap();
        let out_dir = out_dir_os.into_string().unwrap();
        let path = path::Path::new(&out_dir).join("sys.h");
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
        match gpu {
            Gpu::Vulkan => {
                header.push_str("#include <vulkan/vulkan.h>\n");
                println!("cargo:rustc-link-lib=vulkan");
                if let System::Linux = system {
                    header.push_str("#include <vulkan/vulkan_xcb.h>\n");
                }
            },
            Gpu::Opengl => {
                header.push_str("#include <GL/gl.h>\n");
                println!("cargo:rustc-link-lib=GL");
                if let System::Linux = system {
                    header.push_str("#include <GL/glx.h>\n");
                }
            },
            _ => {
                panic!("missing include/lib for gpu=\"{}\"",gpu_name);
            }
        }
        fs::write(&path,header).expect("Unable to write header file");
        process::Command::new("bindgen")
            .args(&[
                &format!("{}/sys.h",out_dir),
                "-o",&format!("{}/sys.rs",out_dir),
                "--disable-nested-struct-naming",
                "--no-prepend-enum-name",
                "--no-layout-tests",
            ])
            .status()
            .expect("unable to generate system FFI bindings");
    }
}
