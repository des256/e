# TODO

- clean up math code, do something smart with SIMD, vectors and colors
- write tests for merged math code
- GPU code only implemented for Vulkan, using GLSL shaders
- async and callback executor architecture like Android, designed around smol

## Base Types

trait Unsigned describes unsigned-specific methods

Unsigned is implemented for:
    u..
    i..
    f..
    Rational<u..,u..>
    Rational<i..,u..>
    Fixed<u..,B>
    Fixed<i..,B>
    Complex<f..>

    question: should Complex<Rational<>> and Complex<Fixed<>> be possible?
    --> possible via templates, but not macros

trait Signed extends Unsigned and describes signed-specific methods

Signed is implemented for:
    i..
    f..
    Rational<i..,u..>
    Fixed<i..,B>
    Complex<f..>

struct Rational<N,D> combines two integers and describes rational-specific methods

Rational<N,D> implements operators:
    N + Rational<N,D> = Rational<N,D>
    Rational<N,D> + N = Rational<N,D>
    Rational<N,D> + Rational<N,D> = Rational<N,D>
    Rational<N,D> += N
    Rational<N,D> += Rational<N,D>
    N - Rational<N,D> = Rational<N,D>
    Rational<N,D> - N = Rational<N,D>
    Rational<N,D> - Rational<N,D> = Rational<N,D>
    Rational<N,D> -= N
    Rational<N,D> -= Rational<N,D>
    N * Rational<N,D>
    Rational<N,D> * N = Rational<N,D>
    Rational<N,D> * Rational<N,D> = Rational<N,D>
    Rational<N,D> *= N
    Rational<N,D> *= Rational<N,D>
    N / Rational<N,D> = Rational<N,D>
    Rational<N,D> / N = Rational<N,D>
    Rational<N,D> / Rational<N,D> = Rational<N,D>
    Rational<N,D> /= N
    Rational<N,D> /= Rational<N,D>
    -Rational<N,D> = Rational<N,D>

signed Rationals have type shortcuts:
    rsize
    r8
    r16
    r32
    r64
    r128

trait Real describes real-specific methods

Real is implemented for:
    f..
    Rational<i..,u..>
    Fixed<i..,..>

trait Float describes float-specific methods

Float is implemented for:
    f..

struct Fixed<T,B> combines an integer and a shift and describes fixed-specific methods

Fixed<T,B> implements operators:
    Fixed<T,B> + Fixed<T,B> = Fixed<T,B>
    Fixed<T,B> += Fixed<T,B>
    Fixed<T,B> - Fixed<T,B> = Fixed<T,B>
    Fixed<T,B> -= Fixed<T,B>
    Fixed<T,B> * Fixed<T,B> = Fixed<T,B>
    Fixed<T,B> *= Fixed<T,B>
    Fixed<T,B> / Fixed<T,B> = Fixed<T,B>
    Fixed<T,B> /= Fixed<T,B>
    -Fixed<T,B> = Fixed<T,B>

    question: implement T + Fixed<T,B>, etc. as well?
    --> not likely

Fixeds have type shortcuts:
    u88 - this is 'unorm' for GPUs (we wipe the 1/256 error under the carpet)
    u168
    u3216
    u6432
    u12864
    i88 - this is 'snorm' for GPUs
    i168
    i3216
    i6432
    i12864

struct Complex<T> combines two reals and describes complex-specific methods

Complex<T> implements operators:
    T + Complex<T> = Complex<T>
    Complex<T> + T = Complex<T>
    Complex<T> + Complex<T> = Complex<T>
    Complex<T> += T
    Complex<T> += Complex<T>
    T - Complex<T> = Complex<T>
    Complex<T> - T = Complex<T>
    Complex<T> - Complex<T> = Complex<T>
    Complex<T> -= T
    Complex<T> -= Complex<T>
    T * Complex<T> = Complex<T>
    Complex<T> * T = Complex<T>
    Complex<T> * Complex<T> = Complex<T>
    Complex<T> *= T
    Complex<T> *= Complex<T>
    T / Complex<T> = Complex<T>
    Complex<T> / T = Complex<T>
    Complex<T> / Complex<T> = Complex<T>
    Complex<T> /= T
    Complex<T> /= Complex<T>
    -Complex<T> = Complex<T>

Complex<T> have type shortcuts:
    c16
    c32
    c64

struct Quaternion<T> combines four reals and describes
quaternion-specific methods

Quaternion<T> implements operators:
    T + Quaternion<T> = Quaternion<T>
    Quaternion<T> + T = Quaternion<T>
    Quaternion<T> + Quaternion<T> = Quaternion<T>
    Quaternion<T> += T
    Quaternion<T> += Quaternion<T>
    T - Quaternion<T> = Quaternion<T>
    Quaternion<T> - T = Quaternion<T>
    Quaternion<T> - Quaternion<T> = Quaternion<T>
    Quaternion<T> -= T
    Quaternion<T> -= Quaternion<T>
    T * Quaternion<T> = Quaternion<T>
    Quaternion<T> * T = Quaternion<T>
    Quaternion<T> * Quaternion<T> = Quaternion<T>
    Quaternion<T> *= T
    Quaternion<T> *= Quaternion<T>
    T / Quaternion<T> = Quaternion<T>
    Quaternion<T> / T = Quaternion<T>
    Quaternion<T> / Quaternion<T> = Quaternion<T>
    Quaternion<T> /= T
    Quaternion<T> /= Quaternion<T>
    -Quaternion<T> = Quaternion<T>

Quaternion<T> have type shortcuts:
    q16
    q32
    q64

struct VecC<T,C> is a fixed length generic vector and describes vector-specific methods

VecC<T,C> implements operators:
    VecC<T,C> + VecC<T,C> = VecC<T,C>
    VecC<T,C> += VecC<T,C>
    VecC<T,C> - VecC<T,C> = VecC<T,C>
    VecC<T,C> -= VecC<T,C>
    T * VecC<T,C> = VecC<T,C>
    VecC<T,C> * T = VecC<T,C>
    VecC<T,C> *= T
    VecC<T,C> / T = VecC<T,C>
    VecC<T,C> /= T

    question: implement VecC<T,C> * VecC<T,C> etc. as well?
    --> no, but the operation should be available as method

struct Vec2<T>, Vec3<T> and Vec4<T> are 2D, 3D and 4D specializations with similar characteristics as VecC

    question: implement interactions with VecC?
    --> no, doesn't seem to make much sense

struct Vec2<T>, Vec3<T> and Vec4<T> implement operators

struct MatRxC<T,R,C> is a fixed size generic matrix and describes matrix-specific methods

MatRxC<T,R,C> implements operators:
    MatRxC<T,R,C> + MatRxC<T,R,C> = MatRxC<T,R,C>
    MatRxC<T,R,C> += MatRxC<T,R,C>
    MatRxC<T,R,C> - MatRxC<T,R,C> = MatRxC<T,R,C>
    MatRxC<T,R,C> -= MatRxC<T,R,C>
    T * MatRxC<T,R,C> = MatRxC<T,R,C>
    VecC<T,C> * MatRxC<T,R,C> = MatRxC<T,R,C>
    MatRxC<T,R,C> * T = MatRxC<T,R,C>
    MatRxC<T,R,C> * VecC<R> = VecC<R>
    MatRxC<T,R,C> * MatRxC<T,R,C> = MatRxC<T,R,C>
    MatRxC<T,R,C> *= T
    MatRxC<T,R,C> / T = MatRxC<T,R,C>
    MatRxC<T,R,C> /= T

struct Mat2x2<T>, Mat2x3<T>, ..., Mat4x3<T>, Mat4x4<T> are specializations with similar characteristics as MatRxC

struct Mat2x2<T>, Mat2x3<T>, ..., Mat4x3<T>, Mat4x4<T> implement operators

struct MultiVec2<T>, MultiVec3<T> and MultiVec4<T> are geometric algebra multivectors

struct Rect<T> describes a rectangle

Pose<T> describes a space transformation in 3D by Vec3 and Quaternion

## Shader Compiler

- use Result in all resolve_*
- use naming new_ for the newly created items everywhere

## NOTES

it does seem to make sense to first get rid of tuples and aliases in a separate pass, in fact, to resurrect the original distinct passes idea:

    1. destructure all pattern nodes into regular boolean and field expressions
    2. convert all enums to structs
    3. convert all tuples to structs
    4. convert aliases to their types
    5. convert anonymous tuples to structs
    6. resolve Descriptor and Destructure nodes from step 1 and 2

what is left is a tree that can easily be translated into one of the target shader languages

- these steps require full knowledge about external types for vertex and uniform descriptions, so the public AST from the ast crate should only represent Rust feature as they go into step 1

- it is theoretically possible to have a different AST between every step, although it's probably more practical to only use a single utility AST between steps 1 and 6, and perhaps only translate to a C-like TAC at step 6

- if the utility AST is always a superset of the public AST, the rendering can be done directly in utility AST, and the public AST is only interesting for gpu_macros, so the ast crate can be removed

- logging all changes made to AST might still be interesting for later optimization, although this is mostly the TAC domain

- expression evaluator? (eval_* in the old code is never referenced...)

- expression type estimator? might not be necessary if we use expectations with should_type (get_expr_type in the old code is never referenced...)

so TODO:

* restore AST from ast crate to minimal Rust AST that matches the language being parsed in gpu_macros, everything is unknown, tuples and enums exist, patterns exist, etc.

* move AST definition into gpu_macros and destroy ast crate, gpu_macros is now self-contained and parses Rust shaders into utility AST constants at compile time

* define utility AST, a superset of gpu_macros AST, in /gpu/sc, with Display implementations

- define TAC in /gpu/sc, with Display implementations, this TAC should be ideal for optimizations and translation into target shader languages

- write step 1 which destructures all pattern nodes into regular boolean and field expressions in utility AST, using Destructure nodes

- write step 2 which converts all enums to structs using Descriptor nodes

- write step 3 which converts all tuples to structs

- write step 4 which converts all aliases to their final types

- write step 5 which converts all anonymous tuple literals to structs

- write step 6 which resolves Descriptor and Destructure nodes, does type checking and renders out TAC buffers

- write renderer from TAC to GLSL for opengl, gles and webgl targets

- write renderer from TAC to SPIR-V

- write renderer from TAC to HLSL

- write renderer from TAC To MSL

- write renderer from TAC to WGSL

- write optimizations for TAC (have fun!)
