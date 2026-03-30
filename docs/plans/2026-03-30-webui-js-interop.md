# WebUI JS Interop Layer Implementation Plan

Created: 2026-03-30
Status: VERIFIED
Approved: Yes
Iterations: 0
Worktree: No
Type: Feature

## Summary

**Goal:** Replace `wasm-bindgen`, `web-sys`, and `js-sys` dependencies in the `webui` crate with a hand-written JS interop layer, eliminating all 14 external dependencies.

**Architecture:** A static JS glue file exports ~17 DOM functions that Rust calls via `extern "C"`. JS objects are tracked by integer handle in a JS-side table. Rust exposes `alloc`/`dealloc` for string passing through linear memory. Event callbacks use a global `Vec<Box<dyn FnMut(u32)>>` registry indexed by integer ID.

**Tech Stack:** Raw WASM FFI (`extern "C"`), hand-written JS glue, `std::alloc` for memory exports.

## Scope

### In Scope

- JS glue file (`webui.js`) with DOM operations and handle management
- Rust FFI module (`ffi.rs`) with `extern "C"` imports and memory exports
- Typed handle newtypes (`JsHandle`, `Element`, `Event`, `MouseEvent`, `KeyboardEvent`)
- Callback registry replacing `Closure::wrap` + `.forget()`
- Update `patcher.rs`, `style.rs`, `builder.rs`, `node.rs` to use new FFI
- Remove `wasm-bindgen`, `web-sys`, `js-sys` from `Cargo.toml`

### Out of Scope

- Build tooling (wasm-pack, trunk) — the JS file is static
- Cleanup/deallocation of handles (current design leaks closures intentionally)
- WebSocket integration (Layer 6 from architecture doc — not yet implemented)
- Widget implementations (slider, dropdown, etc. — not yet implemented)

## Approach

**Chosen:** Hand-written JS glue + `extern "C"` FFI

**Why:** The crate uses only ~17 DOM operations. A static JS file exporting these as WASM imports is straightforward, eliminates 14 dependencies (5 of which are proc-macro crates adding compile time), and matches the project's from-scratch philosophy.

**Alternatives considered:**

- **Keep wasm-bindgen, strip web-sys only** — Still requires the proc-macro chain (syn, quote, etc.) for `Closure::wrap`. Doesn't achieve zero-dependency goal.
- **Use wasm-bindgen only for Closure, hand-bind DOM** — Partial solution. The closure machinery is the most complex part of wasm-bindgen; keeping it defeats most of the purpose.

## Context for Implementer

**Patterns to follow:**
- `base` crate style: grouped imports, `// -- section --` markers, doc comments on every public item
- Existing `runtime.rs` uses `thread_local!` for global state — callback registry should follow the same pattern

**Conventions:**
- Types are `Copy + Clone` where possible (handles are just `u32`)
- Full words for type names (`Context`, `Element`, not `Cx`, `El`)

**Key files:**
- `webui/src/patcher.rs` — main consumer of DOM APIs (mount, events, reactive patching)
- `webui/src/style.rs:317-357` — stylesheet injection (create_element, head, appendChild, append text)
- `webui/src/builder.rs` — event handler types in `on_click`, `on_input`, etc.
- `webui/src/node.rs` — `EventBinding` handler type signature

**Gotchas:**
- `dyn_into`/`dyn_ref` are used for type casting (MouseEvent, KeyboardEvent, HtmlInputElement, HtmlStyleElement) — these become unnecessary with typed newtypes and typed JS imports
- `Closure::wrap` + `.forget()` + `.unchecked_ref()` is the current callback mechanism — replaced entirely by the callback registry
- `class_list().add_1(cls)` is a convenience for `el.classList.add(cls)` — JS glue can use `setAttribute("class", ...)` or `classList.add()` directly
- `el.append_with_str_1(&rules)` appends text content to the `<style>` element — JS glue does `el.textContent += rules` or `el.appendChild(document.createTextNode(rules))`
- The `on_document` method currently uses `JsCast` generics — replaced with typed variants (`on_document_mouse`, `on_document_keyboard`)

**Domain context:**
WASM only has i32/i64/f32/f64 primitives. All JS objects (DOM elements, events) are represented as integer handles. The JS side maintains a `handles[]` array mapping ID → JS object. Strings are passed through WASM linear memory: Rust writes UTF-8 bytes, passes (pointer, length) to JS, JS reads via `new Uint8Array(memory.buffer, ptr, len)` and decodes with `TextDecoder`.

## Assumptions

- WASM is single-threaded — global mutable state (callback registry, handle counter) is safe without synchronization. All files depend on this.
- The JS glue file is loaded before the WASM module (standard WASM instantiation pattern). Tasks 1-5 depend on this.
- Event handlers never need to be deregistered — current design leaks them intentionally, new design follows the same model. Task 3 depends on this.
- Most string arguments flow Rust→JS (tag names, attribute values, class names, CSS rules). The one JS→Rust string case is `keyboard_event_key`, which writes into a caller-provided buffer. Task 1 implements this buffer-write pattern in the JS glue.

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Missing a DOM API call during migration | Low | High — runtime crash | Feature inventory below maps every call site; verification compiles for wasm32 target |
| String encoding mismatch (UTF-8 vs UTF-16) | Low | Medium — garbled text | JS glue uses TextDecoder("utf-8") explicitly |
| Handle table memory growth | Low | Low — handles are never freed currently | Same as current behavior (Closure::forget); acceptable for UI-scale handle counts |
| Event handle growth per dispatch | Medium | Low — per-fire handle push grows table for high-frequency events (mousemove, scroll) | JS-side callback wrapper reuses a fixed event handle slot per cb_id, overwriting on each dispatch rather than pushing |

## Feature Inventory

### wasm-bindgen usage (to be removed)

| Current Usage | File:Line | Replacement | Task |
|---|---|---|---|
| `JsCast` trait (`dyn_into`, `dyn_ref`) | builder.rs:8, patcher.rs:6, style.rs:12 | Typed newtypes — no casting needed | Task 4, 5 |
| `Closure::wrap` + `.forget()` | patcher.rs:63-68, patcher.rs:152-158 | Callback registry (ffi.rs) | Task 3, 5 |
| `Closure::as_ref().unchecked_ref()` | patcher.rs:66, patcher.rs:156 | JS-side callback dispatch by ID | Task 3, 5 |

### web-sys type usage (to be replaced with newtypes)

| web_sys Type | Used As | Replacement Newtype | Task |
|---|---|---|---|
| `web_sys::Window` | Global accessor | Removed (implicit) | Task 2 |
| `web_sys::Document` | Global accessor | Removed (implicit) | Task 2 |
| `web_sys::Element` | DOM element | `ffi::Element` | Task 2 |
| `web_sys::Comment` | Reactive marker | `ffi::JsHandle` | Task 2 |
| `web_sys::Event` | Base event | `ffi::Event` | Task 2 |
| `web_sys::MouseEvent` | Click/mousedown | `ffi::MouseEvent` | Task 2 |
| `web_sys::KeyboardEvent` | Keydown | `ffi::KeyboardEvent` | Task 2 |
| `web_sys::HtmlInputElement` | Input value setter | Folded into `ffi::Element::set_value` | Task 2 |
| `web_sys::HtmlStyleElement` | Style element | `ffi::Element` (no special type needed) | Task 2 |

### DOM API calls (to be implemented in JS glue)

| JS API | Current Rust Call | JS Glue Function | Task |
|---|---|---|---|
| `document.createElement(tag)` | `doc.create_element(tag)` | `create_element(tag_ptr, tag_len) -> handle` | Task 1 |
| `document.createTextNode(text)` | `doc.create_text_node(&t)` | `create_text_node(ptr, len) -> handle` | Task 1 |
| `document.createComment(text)` | `doc.create_comment("reactive")` | `create_comment(ptr, len) -> handle` | Task 1 |
| `document.head` | `doc.head()` | `document_head() -> handle` | Task 1 |
| `el.setAttribute(name, val)` | `el.set_attribute(name, val)` | `set_attribute(handle, name_ptr, name_len, val_ptr, val_len)` | Task 1 |
| `el.classList.add(cls)` | `el.class_list().add_1(cls)` | `class_list_add(handle, cls_ptr, cls_len)` | Task 1 |
| `el.addEventListener(name, fn)` | `el.add_event_listener_with_callback(...)` | `add_event_listener(handle, name_ptr, name_len, cb_id)` | Task 1 |
| `parent.appendChild(child)` | `parent.append_child(&child)` | `append_child(parent_handle, child_handle)` | Task 1 |
| `parent.insertBefore(node, ref)` | `parent.insert_before(&child, ref)` | `insert_before(parent, node, ref_handle)` | Task 1 |
| `parent.removeChild(child)` | `parent.remove_child(&sibling)` | `remove_child(parent, child)` | Task 1 |
| `node.firstChild` | `tmp.first_child()` | `first_child(handle) -> handle_or_0` | Task 1 |
| `node.nextSibling` | `marker.next_sibling()` | `next_sibling(handle) -> handle_or_0` | Task 1 |
| `node.parentNode` | `marker.parent_node()` | `parent_node(handle) -> handle` | Task 1 |
| `input.value = val` | `input.set_value(&val)` | `set_value(handle, ptr, len)` | Task 1 |
| `style.textContent += text` | `style_el.append_with_str_1(&rules)` | `append_text_content(handle, ptr, len)` | Task 1 |
| Event property: `event.target` | Required by widget patterns in architecture doc (dropdown click-outside, text field input.value) | `event_target(handle) -> handle` | Task 1 |
| Event property: `event.clientX` | Required by slider widget in architecture doc (pointer drag) | `mouse_event_client_x(handle) -> i32` | Task 1 |
| Event property: `event.clientY` | Paired with clientX for pointer handling | `mouse_event_client_y(handle) -> i32` | Task 1 |
| Event property: `event.currentTarget` | Required by popup widget in architecture doc (backdrop click detection) | `event_current_target(handle) -> handle` | Task 1 |
| Event property: `event.key` | Required by keyboard handling in architecture doc | `keyboard_event_key(handle, buf_ptr, buf_len) -> actual_len` | Task 1 |
| `document.addEventListener` | `on_document` | `document_add_event_listener(name_ptr, name_len, cb_id)` | Task 1 |

### Event property access — JS→Rust string passing

The `keyboard_event_key` function is the one case where JS needs to write a string into Rust memory. The JS glue writes UTF-8 bytes into a caller-provided buffer and returns the actual length. Rust reads from the buffer.

## Goal Verification

### Truths

1. `Cargo.toml` has zero `[dependencies]` entries — no wasm-bindgen, web-sys, js-sys
2. `cargo build -p webui --target wasm32-unknown-unknown` succeeds with no warnings
3. `cargo test -p webui --lib` passes all existing signal/effect tests
4. `cargo test -p webui --doc` passes all existing doc tests
5. No occurrence of `wasm_bindgen`, `web_sys`, or `js_sys` in any `.rs` file under `webui/src/`
6. A `webui.js` file exists that exports all required DOM operations
7. The public API (`Element`, `Context`, `Signal`, `Node`, `mount`, etc.) is preserved with equivalent functionality

### Artifacts

1. `webui/Cargo.toml` — zero dependencies
2. `webui/src/ffi.rs` — extern imports, handle types, callback registry, memory exports
3. `webui/webui.js` — JS glue file
4. `webui/src/patcher.rs` — updated to use ffi types
5. `webui/src/style.rs` — updated to use ffi types
6. `webui/src/builder.rs` — updated to use ffi types
7. `webui/src/node.rs` — updated to use ffi types

## Progress Tracking

- [x] Task 1: JS glue file
- [x] Task 2: Rust FFI module with handle types
- [x] Task 3: Callback registry
- [x] Task 4: Update node.rs and builder.rs
- [x] Task 5: Update patcher.rs and style.rs
- [x] Task 6: Remove dependencies, verify build

**Total Tasks:** 6 | **Completed:** 6 | **Remaining:** 0

## Implementation Tasks

### Task 1: JS Glue File

**Objective:** Create `webui/webui.js` with all DOM operations, handle table, callback dispatch, and string reading helpers.

**Dependencies:** None

**Files:**

- Create: `webui/webui.js`

**Key Decisions / Notes:**

- Handle table: `const handles = [null];` — index 0 is null/invalid, handles start at 1
- String reading: `function readStr(ptr, len) { return decoder.decode(new Uint8Array(memory.buffer, ptr, len)); }` where `memory` is the WASM memory export
- Callback dispatch: each `cb_id` gets a pre-allocated handle slot; on event fire JS overwrites that slot with the event object and calls `wasm.exports.callback_dispatch(cb_id, slot_handle)` — avoids unbounded handle growth for high-frequency events (mousemove, scroll)
- Keyboard key: write UTF-8 into caller buffer, return actual length
- Module pattern: export a function `webui_imports(memory)` that returns the import object for `WebAssembly.instantiate`
- Handle 0 represents null/none (for firstChild, nextSibling returning null)

**Definition of Done:**

- [ ] File exists at `webui/webui.js`
- [ ] Exports a function that produces a WASM import object
- [ ] All 21 DOM operations from the feature inventory are implemented
- [ ] Callback dispatch calls back into WASM via `callback_dispatch` export

**Verify:**

- Manual review — JS file is not testable with cargo

---

### Task 2: Rust FFI Module — Extern Imports and Handle Types

**Objective:** Create `webui/src/ffi.rs` with `extern "C"` declarations matching the JS glue, typed handle newtypes (`JsHandle`, `Element`, `Event`, `MouseEvent`, `KeyboardEvent`), and memory export functions (`alloc`/`dealloc`).

**Dependencies:** Task 1 (needs to agree on function signatures)

**Files:**

- Create: `webui/src/ffi.rs`

**Key Decisions / Notes:**

- All handle types are `#[derive(Copy, Clone, Debug, PartialEq)]` newtypes around `u32`
- `JsHandle(u32)` is the base type (comments, generic nodes); `Element(u32)`, `Event(u32)`, `MouseEvent(u32)`, `KeyboardEvent(u32)` are specific
- Conversion: `Element` → `JsHandle` is free (From impl); no need for the reverse (type safety)
- String helper: `fn with_str(s: &str, f: impl FnOnce(*const u8, usize) -> R) -> R` to pass Rust strings to extern functions
- Memory exports: `#[no_mangle] pub extern "C" fn alloc(len: usize) -> *mut u8` and `dealloc`
- `#[no_mangle] pub extern "C" fn callback_dispatch(cb_id: u32, event_handle: u32)` — the entry point JS calls for event callbacks
- Event property methods on newtypes: `MouseEvent::client_x()`, `MouseEvent::client_y()`, `Event::target()`, `Event::current_target()`, `KeyboardEvent::key()` (returns String)

**Definition of Done:**

- [ ] All 21 extern functions declared
- [ ] Handle newtypes with doc comments
- [ ] Event property accessor methods
- [ ] `alloc`/`dealloc` exports
- [ ] `callback_dispatch` export
- [ ] `cargo build -p webui --target wasm32-unknown-unknown` succeeds

**Verify:**

- `cargo build -p webui --target wasm32-unknown-unknown 2>&1`

---

### Task 3: Callback Registry

**Objective:** Implement the callback registry that maps integer IDs to Rust closures, replacing `Closure::wrap` + `.forget()`.

**Dependencies:** Task 2 (uses handle types)

**Files:**

- Modify: `webui/src/ffi.rs` (add registry to the same module)

**Key Decisions / Notes:**

- Thread-local `RefCell<Vec<Option<Box<dyn FnMut(u32)>>>>` — index is the callback ID
- `pub fn register_callback(f: impl FnMut(u32) + 'static) -> u32` — pushes to vec, returns index
- `callback_dispatch(cb_id, event_handle)` — borrows registry, calls `registry[cb_id](event_handle)`
- Callbacks receive a raw `u32` handle — callers in patcher.rs wrap this into typed `Event`/`MouseEvent` etc.
- Follow `runtime.rs` pattern: thread-local RefCell with careful borrow management

**Definition of Done:**

- [ ] `register_callback` returns a `u32` ID
- [ ] `callback_dispatch` correctly invokes the registered closure
- [ ] Unit test: register a callback, call dispatch, verify it ran

**Verify:**

- `cargo test -p webui --lib 2>&1`

---

### Task 4: Update node.rs and builder.rs

**Objective:** Replace all `web_sys` and `wasm_bindgen` types with the new FFI types. Update `EventBinding`, event handler signatures, and builder methods.

**Dependencies:** Task 2, Task 3

**Files:**

- Modify: `webui/src/node.rs`
- Modify: `webui/src/builder.rs`

**Key Decisions / Notes:**

- `EventBinding.handler`: changes from `Rc<dyn Fn(&Context, web_sys::Event)>` to `Rc<dyn Fn(&Context, Event)>` (using `ffi::Event`)
- `on_click` handler: changes from `Fn(&Context, web_sys::MouseEvent)` to `Fn(&Context, MouseEvent)`
- `on_mousedown`: same pattern as `on_click`
- `on_keydown`: changes to `Fn(&Context, KeyboardEvent)`
- `on_input`: changes to `Fn(&Context, Event)` (was already generic event)
- `on()` raw handler: `Fn(&Context, Event)`
- Remove all `wasm_bindgen::JsCast` imports
- Remove all `dyn_into` calls — typed dispatch happens in the callback registry layer

**Definition of Done:**

- [ ] No `web_sys` or `wasm_bindgen` imports in `node.rs` or `builder.rs`
- [ ] All event handler signatures use FFI types
- [ ] `cargo build -p webui --target wasm32-unknown-unknown` succeeds

**Verify:**

- `cargo build -p webui --target wasm32-unknown-unknown 2>&1`

---

### Task 5: Update patcher.rs and style.rs

**Objective:** Replace all DOM manipulation code in `patcher.rs` and `style.rs` to use the new FFI functions instead of `web-sys` methods. This is the largest task — every DOM call site changes.

**Dependencies:** Task 2, Task 3, Task 4

**Files:**

- Modify: `webui/src/patcher.rs`
- Modify: `webui/src/style.rs`

**Key Decisions / Notes:**

- `patcher.rs` changes:
  - `window()` / `document()` functions: removed — JS glue operates on the implicit global document
  - `mount(node, parent)`: takes `ffi::Element` instead of `&web_sys::Element`
  - `mount_element`: all `el.set_attribute(...)` → `ffi::set_attribute(handle, ...)`; all `parent.append_child(...)` → `ffi::append_child(parent, child)`; etc.
  - Event binding: `register_callback(move |event_handle| { handler(&Context::new(), Event(event_handle)); })` then `ffi::add_event_listener(el, name, cb_id)`
  - Reactive mounting: marker is a `JsHandle` (comment node), uses `ffi::next_sibling`, `ffi::parent_node`, `ffi::remove_child`
  - `on_document`: split into `on_document_mouse(event, handler)` and `on_document_keyboard(event, handler)` — each wraps the raw callback handle into the typed newtype before calling the handler
  - `dyn_ref::<web_sys::HtmlInputElement>` for value setting → use `ffi::set_value(handle, ...)` unconditionally on the `"value"` attribute path (JS-side `handle.value = ...` works on any element, harmless on non-inputs)
- `style.rs` changes:
  - `ensure_style_element`: `ffi::create_element("style")` → `ffi::document_head()` → `ffi::append_child(head, style_el)`
  - `STYLE_EL` thread-local stores `Option<ffi::Element>` instead of `Option<web_sys::HtmlStyleElement>`
  - `append_with_str_1(&rules)` → `ffi::append_text_content(style_handle, &rules)`
  - Remove `dyn_into::<HtmlStyleElement>` — unnecessary with our typed handle (it's just an Element)
  - Remove `wasm_bindgen::JsCast` import
- Export `mount` and helper functions from `patcher.rs` — `document()` and `window()` are removed from the public API (they returned web_sys types)

**Definition of Done:**

- [ ] No `web_sys`, `wasm_bindgen`, `Closure`, or `JsCast` in `patcher.rs` or `style.rs`
- [ ] All DOM operations use `ffi::` functions
- [ ] `mount` takes `ffi::Element` as parent
- [ ] `on_document_mouse` and `on_document_keyboard` replace the generic `on_document`
- [ ] `cargo build -p webui --target wasm32-unknown-unknown` succeeds

**Verify:**

- `cargo build -p webui --target wasm32-unknown-unknown 2>&1`

---

### Task 6: Remove Dependencies and Final Verification

**Objective:** Remove all three dependencies from `Cargo.toml`, update `lib.rs` re-exports, verify the entire crate compiles and all tests pass.

**Dependencies:** Task 4, Task 5

**Files:**

- Modify: `webui/Cargo.toml`
- Modify: `webui/src/lib.rs`

**Key Decisions / Notes:**

- `Cargo.toml`: remove `wasm-bindgen`, `js-sys`, and entire `[dependencies.web-sys]` section. The `[dependencies]` section should be empty.
- `lib.rs`: update re-exports — add `pub mod ffi`, rename `builder::Element` to `builder::ElementBuilder` throughout, re-export as `ElementBuilder`
- **Naming resolution:** `builder::Element` → `ElementBuilder`. `ffi::Element` stays as `Element`. Re-export `ffi::Element` at crate root. The builder is typically accessed via `div()`, `span()`, `element()` constructors — the `ElementBuilder` type name is rarely written explicitly.
- Remove `document()` and `window()` from public re-exports (they returned web_sys types)
- Verify: `cargo test -p webui --lib`, `cargo test -p webui --doc`, `cargo build -p webui --target wasm32-unknown-unknown`, full workspace `cargo build`

**Definition of Done:**

- [ ] `Cargo.toml` has zero `[dependencies]` entries
- [ ] No occurrence of `wasm_bindgen`, `web_sys`, `js_sys` in any `.rs` file
- [ ] `cargo build -p webui --target wasm32-unknown-unknown` succeeds with no warnings
- [ ] `cargo test -p webui --lib` — all tests pass
- [ ] `cargo test -p webui --doc` — all doc tests pass
- [ ] `builder::Element` renamed to `ElementBuilder` throughout `builder.rs` and all references
- [ ] `lib.rs` re-exports `ElementBuilder` from builder and `Element` from ffi
- [ ] `cargo build` (full workspace) succeeds

**Verify:**

- `cargo test -p webui --lib 2>&1 && cargo test -p webui --doc 2>&1 && cargo build -p webui --target wasm32-unknown-unknown 2>&1 && cargo build 2>&1`

## Deferred Ideas

- Handle deallocation / free list for element handles (when unmount is implemented)
- JS→Rust string passing for more event properties (e.g., input event data)
- `wasm-opt` integration for binary size optimization
