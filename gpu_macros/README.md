# Limited Rust Shader Compiler

## Wants and Needs

- We want to be able to program shaders in Rust directly in the Rust source code
- This should be compiled automatically into GLSL, HLSL, MSL, WebSL, and SPIR-V
- We should be able to use the same vertex and uniform structs in Rust and in the shader

## Artist Impression

In the Rust source code, we define a vertex format:

```
#[derive(Vertex)]
struct MyVertex {
    Vec2<f32> pos;
}
```

And the vertex shader:

```
#[vertex_shader(MyVertex)]
mod my_vertex_shader {
    fn main(vertex: MyVertex) -> Vec4<f32> {
        Vec4<f32> { x: vertex.pos.x,y: vertex.pos.y,z: 0.0,w: 1.0, }
    }
}
```

or smaller if there are no functions other than the main shader:

```
#[vertex_shader(MyVertex)]
fn my_vertex_shader(vertex: MyVertex) -> Vec4<f32> {
    Vec4<f32> { x: vertex.pos.x,y: vertex.pos.y,z: vertex.pos.z, w: 1.0, }
}
```

And similarly for the fragment shader:

```
[fragment_shader]
mod my_fragment_shader {
    fn main(color: Vec3<f32>) -> Vec4<f32> {
        Vec4<f32> { x: color.x,y: color.y,z: color.z,w: 1.0, }
    }
}
```

or:

```
fn my_fragment_shader(color: Vec3<f32>) -> Vec4<f32> {
    Vec4<f32> { x: color.x,y: color.y,z: color.z,w: 1.0, }
}
```

To specify the vertex buffer, we can use `MyVertex` (since it supports the `Vertex` trait), and to create the shader objects:

```
vertex_shader = gpu.create_vertex_shader(my_vertex_shader::module());
fragment_shader = gpu.create_fragment_shader(my_fragment_shader::module());
```

## The Process

The compiler calls the `vertex_shader` macro processor. This reads the following token stream and parses it into an AST. It then renders the AST as Rust constant that replaces the vertex shader function, like so:

```
my_vertex_shader {
    pub fn ast() -> ast::Module {
        {
            use {
                super::*,
                std::collections::HashMap,
            };
            let tuples: HashMap<String,ast::Tuple> = HashMap::new();
            let mut extern_structs: HashMap<String,ast::Struct> = HashMap::new();
            extern_structs.insert("MyVertex".to_string(),super::MyVertex::ast());
            let mut structs: HashMap<String,ast::Struct> = HashMap::new();
            let enums: HashMap<String,ast::Enum> = HashMap::new();
            let aliases: HashMap<String,ast::Alias> = HashMap::new();
            let consts: HashMap<String,ast::Const> = HashMap::new();
            let mut functions: HashMap<String,ast::Function> = HashMap::new();
            functions.insert("main".to_string(),ast::Function {
                ident: "main".to_string(),
                params: vec![
                    ast::Symbol {
                        ident: "vertex".to_string(),
                        type_: ast::Type::UnknownIdent("MyVertex".to_string()),
                    },
                ],
                type_: ast::Type::Generic("Vec4".to_string(),vec![ast::Type::F32,]),
                block: ast::Block {
                    stats: vec![],
                    expr: Some(Box::new(ast::Expr::Field(Box::new(ast::Expr::UnknownIdent("vertex".to_string())),"_pos".to_string()))),
                },
            });
            ast::Module {
                ident: "triangle_vs".to_string(),
                tuples,
                structs,
                extern_structs,
                enums,
                aliases,
                consts,
                functions,
            }
        }
    }
}
```

Then, when the shader is needed, the implementation of `Gpu::create_vertex_shader` or `Gpu::create_fragment_shader` gets called with the shader AST. These methods first optimize the AST generically (this could also be done during compilation), and then specifically for their respective shader languages, after which they render the code to pass down to the API.

## Development Details

Generally, the compiler should convert Rust into one of the shading languages. These are all C-like languages, so the compiler most likely only has to do:

1. parse Rust into syntax tree
2. convert Rust features into C-like equivalents
3. render into target language

So there is no intermediate representation or assembly, and we're not considering optimizations at this time.

### Structs Inside and Outside Shaders

To use Vertex and Uniform structs both inside and outside the shaders, the compiler will need access to the description of these structs. This is not possible during compiletime because attribute macros have no view outside the shader code. To fix this, we'll need to split the compiler into a compiletime part and a runtime part. In the runtime part we have access to the structure descriptions. Parsing the shader into the syntax tree can be done at compiletime, the syntax tree can then be rendered into Rust literals, so they become available during runtime. At runtime, the tree is then rendered in the target language.

## Defining Limited Rust

`RUST.md` shows the grammar of Rust full of features we don't need or that make no sense in shaders:

- Generics: Other than Vec2<f32> and such, let's not support generics.
- Function pointers and closures: Let's not support this for shaders.
- Nested modules: Let's not support this. A shader should be expressed as one module, containing the relevant functions and constants and such.
- Macros: Let's not support this.
- Conditional compilation: Let's not support this.
- Referencing external crates and use declarations: Let's not support this.
- Type aliasing: Let's not support this.
- Static/global variables: This makes very little sense in shaders. - OR - These translate to uniforms...
- Traits: Let's not support this.
- Lifetimes and mutability: Even though this is important in Rust, it doesn't make much sense in shaders.
- Inherent implementations: Maybe...
- Visibility: Shaders don't export anything.
- Strings and characters: This has no meaning for shaders.
- Unsafe: This has no meaning in shaders, and we don't support mutability.
- Borrowing, references and pointers: Also a big concept in Rust, but we don't need this in shaders.

`GRAMMAR.md` shows the simplified grammar without these features. This grammar is halfway between C and Rust, and has been taylored to get rid of parsing ambiguities and also automatically solve operator precedence like older C compilers do. Parsing ambiguities can occur when attempting to parse, for example `Foo::Bar(12)`. It can be:

- a function call `Bar` from the module `Foo`, with parameter 12
- a tuple struct literal `Bar(12)` from the module `Foo`
- an enum variant `Bar(12)` from the enum `Foo`

and even:

- a tuple struct pattern matching `Bar(12)` from the module `Foo`
- an enum pattern matching `Bar(12)` from the enum `Foo`

## Representing Limited Rust Features

Some features that still exist cannot be represented directly in any of the target shading languages, but we like to be able to use them anyway.

- Tuples
- Rust-style Enums
- Pattern matching (`let`, `if let`, `while let`, `match`, `for`)
- Everything is an Expression

### Tuples

Tuples are actually just structs without field names, and structs are supported in most target languages. This means there is a one-to-one mapping possible, probably with some name mangling.

### Enums

Enums (or unions) are not supported, so we need to come up with a way to store enums in structs. A very straightforward way to do this is keep an unsigned int to denote the variant, and then economically add any struct or tuple elements such that you reuse elements of the same type where needed. For example:

```
enum Material {
    Direct(Color<u8>),
    Cartoon(Color<u8>),
    Phong { color: Color<u8>,shininess: f32, },
    BRDF(u8),
}
```

becomes:

```
struct Material {
    variant: u8,  // 0..3
    _0: Color<u8>,
    _1: f32,
    _2: u8,
}
```

### Pattern Matching

Pattern matching can be split into two distinct tasks: the boolean expression to find out if a pattern matches or not, and destructuring of local variables from a successful match. The boolean expression can be used in traditional if and while statements, and destructuring will form the first statements of the code inside the if- or while-block. For example:

```
if let MyStruct { a: 4,b: x,.. } = data {
    ...
}
```

The boolean expression is `data.a == 4` and the destructuring statement is `x = data.b;`. So this `if let` can be transformed into an `if` (which is available in C-like languages) like so:

```
if data.a == 4 {
    let x = data.b;
    ...
}
```

Another example:

```
let color = match material {
    Material::Direct(color) => color,
    Material::Cartoon(color) => do_cartoon(color),
    Material::Phong { color,shininess, } => do_phong(color,shininess),
    Material::BRDF(foobar) => do_brdf(foobar),
}
```

Becomes:

```
if material.variant == 0 {
    let color = material._0;
    color
}
else if material.variant == 1 {
    let color = material._0;
    do_gouraud(color)
}
else if material.variant == 2 {
    let color = material._0;
    let shininess = material._1;
    do_phong(color,shininess);
}
else if material.variant == 3 {
    let foobar = material._2;
    do_brdf(foobar);
}
```

#### Boolean Expression

The boolean expression can be created by recursively rendering pattern checks for specific patterns (so, anything but identifiers, wildcards or rest indicators) and connecting them with logical or operators.

#### Destructuring Statements

Destructuring statements can be created by recursively rendering the points where identifiers appear as individual let-statements. For example:

```
let Vec4<f32> { x,y,.. } = foo;
```

becomes:

```
let x = foo.x;
let y = foo.y;
```
