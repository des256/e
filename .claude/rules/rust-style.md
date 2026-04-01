## Rust Code Style

### Imports

Consolidate all imports into a single `use { ... };` block with nested braces. Group by origin: `crate::` first, then external crates, then `std::`.

```rust
// ✅ Correct — single consolidated block
use {
    crate::{Foo, Bar},
    shared::Message,
    std::{
        cell::Cell,
        sync::{Arc, Mutex},
    },
};

// ❌ Wrong — flat separate statements
use crate::Foo;
use shared::Message;
use std::cell::Cell;
use std::sync::{Arc, Mutex};
```

Exception: a file with only one or two simple imports (e.g. `use crate::F16;`) doesn't need the outer braces.

### Section Comments

Use `// -- section name --` (lowercase, double-dashed) to separate logical sections within a file.

```rust
// -- constructors --

// -- layout builders --

// -- events --
```

### Documentation

- Every `lib.rs` starts with `//!` module-level doc comments.
- All public items (`pub fn`, `pub struct`, `pub enum`, `pub trait`, `pub const`) get `///` doc comments.
- Reference related types with rustdoc links: `` [`TypeName`] ``.
- Public struct fields get `///` comments.
- Doc examples use `use <crate>::*;` to mirror how downstream code imports.

### Derives

Use consistent ordering: `Copy, Clone, Debug, PartialEq, Eq` (drop traits that don't apply).

### General

- `const fn` constructors where possible.
- Test modules use `use super::*;`.
- No trailing comments on `use` lines or struct fields — use `///` above instead.
- Inline comments are rare — only for section breaks and brief non-obvious clarifications.
