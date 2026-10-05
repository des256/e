fn main() {
    // --- Always: plain TensorRT runtime ---
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("src/ffi.cpp")
        .include("/usr/local/cuda/include")
        .compile("trt_ffi_stub");

    // TensorRT
    println!("cargo:rustc-link-search=native=/usr/local/tensorrt/lib");
    println!("cargo:rustc-link-search=native=/usr/lib/aarch64-linux-gnu");
    println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
    println!("cargo:rustc-link-lib=dylib=nvinfer");

    // CUDA runtime
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/aarch64-linux/lib");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/targets/x86_64-linux/lib");
    println!("cargo:rustc-link-lib=dylib=cudart");

    // C++ standard library
    println!("cargo:rustc-link-lib=dylib=stdc++");

    println!("cargo:rerun-if-changed=src/ffi/ffi.cpp");
    println!("cargo:rerun-if-changed=src/ffi/ffi.h");
}
