# GPU Macros

These macros make it possible for (reduced) Rust to be used to specify various GPU things, like shaders and vertex formats.

## Vertex Format

Add ```#[derive(Vertex)]``` before any structure to use it as vertex format. The macro will automatically create the Vertex trait, which is needed by vertex buffers and vertex shaders.

## Vertex Shader

Specify the shader as a module, and add ```#[vertex_shader(MyVertexFormat)]``` before to connect it to the GPU. `MyVertexFormat` is a struct that has `Vertex` derived from it. A quick example:

```
#[derive(Vertex)]
struct MyVertex {
    position: Vec2<f32>,
}

#[vertex_shader(MyVertex)]
mod my_vertex_shader {
    fn main(vertex: MyVertex) -> Vec4<f32> {
        Vec4<f32> {
            x: vertex.position.x,
            y: vertex.position.y,
            z: 0.0,
            w: 1.0,
        }
    }
}
```

Later, when the shader is needed, call ```my_vertex_shader::create()``` to generate the GPU-specific ```VertexShader``` object.

## Fragment Shader

Specify the shader as a module, and add ```#[fragment_shader]``` before to connect it to the GPU. A quick example:

```
#[fragment_shader]
mod my_fragment_shader {
    fn main() -> Color<f32> {
        Color<f32> {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }
}
```

Later, when the shader is needed, call ```my_fragment_shader::create()``` to generate the GPU-specific ```FragmentShader``` object.
