fn main() {    
#[cfg(feature="system_linux")]
    println!("cargo:rustc-link-lib=xcb");
    
#[cfg(feature="gpu_vulkan")]
    println!("cargo:rustc-link-lib=vulkan");
}
