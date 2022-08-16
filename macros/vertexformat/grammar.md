# Rust Grammar Subset for Valid Vertex Definition

The input can be assumed to be valid Rust already, since it is also compiled by rustc. This simplifies the parser considerably.

We're assuming only one level of generic parameter. This covers `Vec2<u8>`, `Vec4<f32>`, etc.

Attrs = { `#` `[` ... `]` } .

Visibility = [ `pub` [ `(` ... `)` ] ] .

Struct = Attrs Visibility `struct` IDENTIFIER `{` { StructField [ `,` ] } `;` .

StructField = Attrs Visibility IDENTIFIER `:` IDENTIFIER [ `<` IDENTIFIER `>` ] .

