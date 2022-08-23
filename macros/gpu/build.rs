// the only purpose of this build script is to make sure the macro tools have the same configuration
use std::env;

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

    // and indicate to rerun only if changed
    println!("cargo:rerun-if-changed=build.rs");
}
