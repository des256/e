#!/bin/sh
# Run this every time the features change, before running cargo build
bindgen --disable-nested-struct-naming --no-prepend-enum-name --no-layout-tests wrapper.h -o src/bindings.rs -- -D_SYSTEM_LINUX_ -D_GPU_VULKAN_
