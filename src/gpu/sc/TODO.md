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

- expression type estimator? might not be necessary if we use expectations with expected_type (get_expr_type in the old code is never referenced...)

so TODO:

* restore AST from ast crate to minimal Rust AST that matches the language being parsed in gpu_macros, everything is unknown, tuples and enums exist, patterns exist, etc.

* move AST definition into gpu_macros and destroy ast crate, gpu_macros is now self-contained and parses Rust shaders into utility AST constants at compile time

* define utility AST, a superset of gpu_macros AST, in /gpu/sc, with Display implementations

+ define TAC in /gpu/sc, with Display implementations, this TAC should be ideal for optimizations and translation into target shader languages

* make sure the system uses Result<> instead of panic! for errors

* write step 1 which destructures all pattern nodes into regular boolean and field expressions in utility AST, using Destructure nodes

* make sure standardlib works with methods defined on any type, not just structs

- write step 2 which converts all enums to structs using Descriptor nodes

X steps 3, 4 and 5 no longer needed

- write step 6 which resolves Descriptor and Destructure nodes, does type checking and renders out TAC buffers

- write renderer from TAC to GLSL for opengl, gles and webgl targets

- write renderer from TAC to SPIR-V

- write renderer from TAC to HLSL

- write renderer from TAC To MSL

- write renderer from TAC to WGSL

- write optimizations for TAC (have fun!)
