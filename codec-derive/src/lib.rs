//! Proc-macro crate for `#[derive(Codec)]`.
//!
//! Generates [`codec::Codec`] implementations for structs and enums.
//! Uses raw `proc_macro` token parsing — no external dependencies.

extern crate proc_macro;

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};

// -- public derive entry point --

/// Derive macro for [`codec::Codec`].
///
/// Supports structs with named fields and enums with unit, tuple, and
/// struct variants.
///
/// # Examples
///
/// ```ignore
/// use codec::Codec;
///
/// #[derive(Codec)]
/// struct Point {
///     x: f32,
///     y: f32,
/// }
/// ```
#[proc_macro_derive(Codec)]
pub fn derive_codec(input: TokenStream) -> TokenStream {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut pos = 0;

    // skip attributes
    while pos < tokens.len() {
        if matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == '#') {
            pos += 1; // skip '#'
            if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket) {
                pos += 1; // skip [...]
            }
            continue;
        }
        break;
    }

    // skip visibility (pub, pub(crate), etc.)
    pos = skip_visibility(&tokens, pos);

    // read 'struct' or 'enum'
    let kind = match &tokens[pos] {
        TokenTree::Ident(id) => id.to_string(),
        _ => panic!("expected 'struct' or 'enum'"),
    };
    pos += 1;

    // read name
    let name = match &tokens[pos] {
        TokenTree::Ident(id) => id.to_string(),
        _ => panic!("expected type name"),
    };
    pos += 1;

    // read optional generics
    let (generics, pos) = parse_generics(&tokens, pos);

    // read optional where clause (before the body)
    let (where_clause, pos) = parse_where_clause(&tokens, pos);

    match kind.as_str() {
        "struct" => derive_struct(&name, &generics, &where_clause, &tokens, pos),
        "enum" => derive_enum(&name, &generics, &where_clause, &tokens, pos),
        _ => panic!("expected 'struct' or 'enum', got '{}'", kind),
    }
}

// -- generics parsing --

#[derive(Clone)]
struct GenericParam {
    name: String,
}

fn parse_generics(tokens: &[TokenTree], mut pos: usize) -> (Vec<GenericParam>, usize) {
    let mut params = Vec::new();

    // check for '<'
    if pos >= tokens.len() {
        return (params, pos);
    }
    let is_lt = matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == '<');
    if !is_lt {
        return (params, pos);
    }
    pos += 1;

    // collect type parameter names (skip bounds)
    let mut depth = 1;
    let mut expect_name = true;
    while pos < tokens.len() && depth > 0 {
        match &tokens[pos] {
            TokenTree::Punct(p) if p.as_char() == '<' => depth += 1,
            TokenTree::Punct(p) if p.as_char() == '>' => {
                depth -= 1;
                if depth == 0 {
                    pos += 1;
                    break;
                }
            }
            TokenTree::Punct(p) if p.as_char() == ',' && depth == 1 => {
                expect_name = true;
            }
            TokenTree::Ident(id) if expect_name && depth == 1 => {
                let name = id.to_string();
                // skip lifetime params
                if name != "const" {
                    params.push(GenericParam { name });
                }
                expect_name = false;
            }
            TokenTree::Punct(p) if p.as_char() == '\'' && depth == 1 => {
                // lifetime — skip the name too
                expect_name = false;
                pos += 1; // skip the lifetime name
            }
            _ => {}
        }
        pos += 1;
    }

    (params, pos)
}

// -- where clause parsing --

fn parse_where_clause(tokens: &[TokenTree], mut pos: usize) -> (String, usize) {
    if pos >= tokens.len() {
        return (String::new(), pos);
    }
    let is_where = matches!(&tokens[pos], TokenTree::Ident(id) if id.to_string() == "where");
    if !is_where {
        return (String::new(), pos);
    }

    let mut clause = String::from("where ");
    pos += 1;

    // collect tokens until we hit a { or ;
    while pos < tokens.len() {
        match &tokens[pos] {
            TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => break,
            TokenTree::Punct(p) if p.as_char() == ';' => break,
            tt => {
                clause.push_str(&tt.to_string());
                clause.push(' ');
            }
        }
        pos += 1;
    }

    (clause, pos)
}

// -- visibility skipping --

fn skip_visibility(tokens: &[TokenTree], mut pos: usize) -> usize {
    if pos >= tokens.len() {
        return pos;
    }
    let is_pub = matches!(&tokens[pos], TokenTree::Ident(id) if id.to_string() == "pub");
    if !is_pub {
        return pos;
    }
    pos += 1;
    // skip optional (crate) / (super) / (in path)
    if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis) {
        pos += 1;
    }
    pos
}

// -- struct field parsing --

struct Field {
    name: String,
    ty: String,
}

fn parse_struct_fields(body: &Group) -> Vec<Field> {
    let tokens: Vec<TokenTree> = body.stream().into_iter().collect();
    let mut fields = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        // skip attributes
        if matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == '#') {
            pos += 1;
            if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket) {
                pos += 1;
            }
            continue;
        }

        // skip visibility
        pos = skip_visibility(&tokens, pos);

        // read field name
        let field_name = match tokens.get(pos) {
            Some(TokenTree::Ident(id)) => id.to_string(),
            _ => break,
        };
        pos += 1;

        // skip ':'
        if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == ':') {
            pos += 1;
        }

        // collect type tokens until ',' or end
        let mut ty = String::new();
        let mut depth = 0i32;
        while pos < tokens.len() {
            match &tokens[pos] {
                TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                    pos += 1;
                    break;
                }
                TokenTree::Punct(p) if p.as_char() == '<' => {
                    depth += 1;
                    ty.push('<');
                }
                TokenTree::Punct(p) if p.as_char() == '>' => {
                    depth -= 1;
                    ty.push('>');
                }
                tt => {
                    ty.push_str(&tt.to_string());
                }
            }
            pos += 1;
        }

        fields.push(Field { name: field_name, ty: ty.trim().to_string() });
    }

    fields
}

// -- struct derive --

fn derive_struct(name: &str, generics: &[GenericParam], where_clause: &str, tokens: &[TokenTree], pos: usize) -> TokenStream {
    let body = match tokens.get(pos) {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g,
        _ => panic!("expected struct body"),
    };

    let fields = parse_struct_fields(body);
    let gen_params = generic_params_str(generics);
    let gen_args = generic_args_str(generics);
    let where_str = build_where_clause(generics, where_clause);

    let mut encode_body = String::new();
    let mut decode_body = String::new();
    let mut field_names = Vec::new();

    for f in &fields {
        encode_body.push_str(&format!("self.{}.encode(buf);\n", f.name));
        decode_body.push_str(&format!(
            "let ({name}, _n) = <{ty} as codec::Codec>::decode(&buf[_off..])?; _off += _n;\n",
            name = f.name,
            ty = f.ty,
        ));
        field_names.push(f.name.clone());
    }

    let constructor: String = field_names.join(", ");

    let code = format!(
        r#"impl{gen_params} codec::Codec for {name}{gen_args} {where_str} {{
            fn encode(&self, buf: &mut Vec<u8>) {{
                {encode_body}
            }}

            fn decode(buf: &[u8]) -> std::result::Result<(Self, usize), codec::CodecError> {{
                let mut _off = 0usize;
                {decode_body}
                Ok(({name} {{ {constructor} }}, _off))
            }}
        }}"#,
    );

    code.parse().expect("generated code must parse")
}

// -- enum variant parsing --

enum VariantKind {
    Unit,
    Tuple(Vec<String>),
    Struct(Vec<Field>),
}

struct Variant {
    name: String,
    kind: VariantKind,
}

fn parse_enum_variants(body: &Group) -> Vec<Variant> {
    let tokens: Vec<TokenTree> = body.stream().into_iter().collect();
    let mut variants = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        // skip attributes
        if matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == '#') {
            pos += 1;
            if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket) {
                pos += 1;
            }
            continue;
        }

        // read variant name
        let variant_name = match tokens.get(pos) {
            Some(TokenTree::Ident(id)) => id.to_string(),
            _ => break,
        };
        pos += 1;

        // check what follows: '(' for tuple, '{' for struct, ',' or end for unit
        let kind = if pos < tokens.len() {
            match &tokens[pos] {
                TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => {
                    let types = parse_tuple_fields(g);
                    pos += 1;
                    VariantKind::Tuple(types)
                }
                TokenTree::Group(g) if g.delimiter() == Delimiter::Brace => {
                    let fields = parse_struct_fields(g);
                    pos += 1;
                    VariantKind::Struct(fields)
                }
                _ => VariantKind::Unit,
            }
        } else {
            VariantKind::Unit
        };

        variants.push(Variant { name: variant_name, kind });

        // skip ',' if present
        if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == ',') {
            pos += 1;
        }
    }

    variants
}

fn parse_tuple_fields(group: &Group) -> Vec<String> {
    let tokens: Vec<TokenTree> = group.stream().into_iter().collect();
    let mut types = Vec::new();
    let mut pos = 0;

    while pos < tokens.len() {
        // skip attributes
        if matches!(&tokens[pos], TokenTree::Punct(p) if p.as_char() == '#') {
            pos += 1;
            if pos < tokens.len() && matches!(&tokens[pos], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket) {
                pos += 1;
            }
            continue;
        }

        let mut ty = String::new();
        let mut depth = 0i32;
        while pos < tokens.len() {
            match &tokens[pos] {
                TokenTree::Punct(p) if p.as_char() == ',' && depth == 0 => {
                    pos += 1;
                    break;
                }
                TokenTree::Punct(p) if p.as_char() == '<' => {
                    depth += 1;
                    ty.push('<');
                }
                TokenTree::Punct(p) if p.as_char() == '>' => {
                    depth -= 1;
                    ty.push('>');
                }
                tt => {
                    ty.push_str(&tt.to_string());
                }
            }
            pos += 1;
        }

        let ty = ty.trim().to_string();
        if !ty.is_empty() {
            types.push(ty);
        }
    }

    types
}

// -- enum derive --

fn derive_enum(name: &str, generics: &[GenericParam], where_clause: &str, tokens: &[TokenTree], pos: usize) -> TokenStream {
    let body = match tokens.get(pos) {
        Some(TokenTree::Group(g)) if g.delimiter() == Delimiter::Brace => g,
        _ => panic!("expected enum body"),
    };

    let variants = parse_enum_variants(body);
    let gen_params = generic_params_str(generics);
    let gen_args = generic_args_str(generics);
    let where_str = build_where_clause(generics, where_clause);

    // build encode match arms
    let mut encode_arms = String::new();
    for (idx, v) in variants.iter().enumerate() {
        let idx_u32 = idx as u32;
        match &v.kind {
            VariantKind::Unit => {
                encode_arms.push_str(&format!(
                    "{name}::{vname} => {{ ({idx_u32}u32).encode(buf); }}\n",
                    vname = v.name,
                ));
            }
            VariantKind::Tuple(types) => {
                let bindings: Vec<String> = (0..types.len()).map(|i| format!("_f{}", i)).collect();
                let pattern = bindings.join(", ");
                let mut body = format!("({idx_u32}u32).encode(buf);\n");
                for b in &bindings {
                    body.push_str(&format!("{b}.encode(buf);\n"));
                }
                encode_arms.push_str(&format!(
                    "{name}::{vname}({pattern}) => {{ {body} }}\n",
                    vname = v.name,
                ));
            }
            VariantKind::Struct(fields) => {
                let bindings: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();
                let pattern = bindings.join(", ");
                let mut body = format!("({idx_u32}u32).encode(buf);\n");
                for b in &bindings {
                    body.push_str(&format!("{b}.encode(buf);\n"));
                }
                encode_arms.push_str(&format!(
                    "{name}::{vname} {{ {pattern} }} => {{ {body} }}\n",
                    vname = v.name,
                ));
            }
        }
    }

    // build decode match arms
    let mut decode_arms = String::new();
    for (idx, v) in variants.iter().enumerate() {
        let idx_u32 = idx as u32;
        match &v.kind {
            VariantKind::Unit => {
                decode_arms.push_str(&format!(
                    "{idx_u32} => Ok(({name}::{vname}, _off)),\n",
                    vname = v.name,
                ));
            }
            VariantKind::Tuple(types) => {
                let mut body = String::new();
                let mut bindings = Vec::new();
                for (i, ty) in types.iter().enumerate() {
                    let bname = format!("_f{}", i);
                    body.push_str(&format!(
                        "let ({bname}, _n) = <{ty} as codec::Codec>::decode(&buf[_off..])?; _off += _n;\n",
                    ));
                    bindings.push(bname);
                }
                let constructor = bindings.join(", ");
                decode_arms.push_str(&format!(
                    "{idx_u32} => {{ {body} Ok(({name}::{vname}({constructor}), _off)) }}\n",
                    vname = v.name,
                ));
            }
            VariantKind::Struct(fields) => {
                let mut body = String::new();
                let mut field_names = Vec::new();
                for f in fields {
                    body.push_str(&format!(
                        "let ({fname}, _n) = <{ty} as codec::Codec>::decode(&buf[_off..])?; _off += _n;\n",
                        fname = f.name,
                        ty = f.ty,
                    ));
                    field_names.push(f.name.clone());
                }
                let constructor = field_names.join(", ");
                decode_arms.push_str(&format!(
                    "{idx_u32} => {{ {body} Ok(({name}::{vname} {{ {constructor} }}, _off)) }}\n",
                    vname = v.name,
                ));
            }
        }
    }

    let code = format!(
        r#"impl{gen_params} codec::Codec for {name}{gen_args} {where_str} {{
            fn encode(&self, buf: &mut Vec<u8>) {{
                match self {{
                    {encode_arms}
                }}
            }}

            fn decode(buf: &[u8]) -> std::result::Result<(Self, usize), codec::CodecError> {{
                let mut _off = 0usize;
                let (_variant_idx, _n) = <u32 as codec::Codec>::decode(buf)?;
                _off += _n;
                match _variant_idx {{
                    {decode_arms}
                    _ => Err(codec::CodecError::InvalidVariant(_variant_idx)),
                }}
            }}
        }}"#,
    );

    code.parse().expect("generated enum code must parse")
}

// -- helpers --

fn generic_params_str(generics: &[GenericParam]) -> String {
    if generics.is_empty() {
        String::new()
    } else {
        let params: Vec<&str> = generics.iter().map(|g| g.name.as_str()).collect();
        format!("<{}>", params.join(", "))
    }
}

fn generic_args_str(generics: &[GenericParam]) -> String {
    generic_params_str(generics) // same for simple type params
}

fn build_where_clause(generics: &[GenericParam], existing: &str) -> String {
    let codec_bounds: Vec<String> = generics
        .iter()
        .map(|g| format!("{}: codec::Codec", g.name))
        .collect();

    if existing.is_empty() && codec_bounds.is_empty() {
        String::new()
    } else if existing.is_empty() {
        format!("where {}", codec_bounds.join(", "))
    } else {
        // existing already starts with "where "
        if codec_bounds.is_empty() {
            existing.to_string()
        } else {
            format!("{}, {}", existing.trim_end(), codec_bounds.join(", "))
        }
    }
}
