# VERTEX SHADER

mod my_vertex_shader {

    fn main(vertex: MyVertex,uniforms: MyUniforms,...) -> (Vec3<f32>,MyVaryings) {
        ...
    }
}

# FRAGMENT SHADER

mod my_fragment_shader {

    fn main(varyings: MyVaryings,uniforms: MyUniforms,...) -> (Color<f32>,...) {
        ...
    }
}