use {
    std::{
        env,
        fs,
        process,
        path,
    },
};

fn main() {

    let out_dir_os = env::var_os("OUT_DIR").unwrap();
    let out_dir = out_dir_os.into_string().unwrap();

    // create header file
    let path = path::Path::new(&out_dir).join("sys.h");
    let mut header = String::new();
#[cfg(target_os="linux")]
    header.push_str("#include <X11/Xlib.h>\n");
    header.push_str("#include <X11/Xlib-xcb.h>\n");
    header.push_str("#include <xcb/xcb.h>\n");
#[cfg(feature="gpu_vulkan")]
    {
        header.push_str("#include <vulkan/vulkan.h>\n");
#[cfg(target_os="linux")]
        header.push_str("#include <vulkan/vulkan_xcb.h>\n");
    }
#[cfg(feature="gpu_opengl")]
    {
        header.push_str("#include <GL/gl.h>\n");
#[cfg(target_os="linux")]
        header.push_str("#include <GL/glx.h>\n");
    }
    fs::write(&path,header).expect("Unable to write header file");

    // generate bindings
    process::Command::new("bindgen")
        .args(&[
            &format!("{}/sys.h",out_dir),
            "-o",&format!("{}/sys.rs",out_dir),
            "--disable-nested-struct-naming",
            "--no-prepend-enum-name",
            "--no-layout-tests",
        ])
        .status()
        .expect("Unable to generate system FFI bindings");

    // instruct cargo to also include the relevant libraries
#[cfg(target_os="linux")]
    {
        println!("cargo:rustc-link-lib=X11");
        println!("cargo:rustc-link-lib=X11-xcb");
        println!("cargo:rustc-link-lib=xcb");
    }    
#[cfg(feature="gpu_vulkan")]
    println!("cargo:rustc-link-lib=vulkan");
#[cfg(feature="gpu_opengl")]
    println!("cargo:rustc-link-lib=GL");
}
