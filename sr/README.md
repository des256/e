# SR - Shader Representation

This is a common understanding between gpu_macros and the main library. gpu_macros replaces the rust shader code with SR literals at compile time. The main library then compiles/renders these literals as GLSL/SPIR-V/etc. on the target platform.
