# Rust WASM Frontend Architecture

A minimal, from-scratch frontend framework in Rust targeting the browser DOM via WebSocket-connected backends.

## Design Principles

- **CSS does layout** — no reimplementation of flexbox/grid/box model in Rust
- **Signals solve ownership** — `Copy`-able handles into an arena, no borrow checker conflicts
- **Fine-grained reactivity** — no virtual DOM, only affected DOM nodes update
- **Views are functions** — no OO widget tree, no trait objects, no lifecycle ceremony
- **Two state sources** — local UI signals (tabs, input text) + backend signals (WebSocket data)
- **Hybrid styling** — layout via Rust builder methods (inline styles), interactive states via generated CSS classes, theme via CSS custom properties

## Project Structure

```
e/
  crates/
    shared/           # types, Codec derives — used by both backend and frontend
      src/lib.rs
    backend/
      src/main.rs     # HTTP server + WebSocket + app logic
    frontend/
      src/lib.rs      # WASM entry point, app(), widget functions
      src/ui/         # framework: signals, nodes, patcher, style builders
      src/widgets/    # reusable: slider, tab_bar, dropdown, etc.
```

`shared` crate has `#[derive(Codec)]` types. Both `backend` and `frontend` depend on it.
The frontend compiles to WASM via `wasm-pack` or `trunk`. The backend serves the `.wasm` + `.js` glue + `index.html`.

## Architecture

```
  Backend (Rust)                     Frontend (Rust -> WASM)
 +----------------+                 +-----------------------------------+
 |  App State     |<-- WebSocket -->|  Signal Arena (reactive core)     |
 |  Codec types   |   (your codec)  |    backend_data: Signal<T>        |
 +----------------+                 |    active_tab: Signal<usize>      |
                                    |    input_text: Signal<String>     |
                                    +-----------------------------------+
                                    |  View Tree (declarative)          |
                                    |    reads signals -> emits Nodes   |
                                    +-----------------------------------+
                                    |  Patcher (fine-grained)           |
                                    |    Node -> DOM via web-sys        |
                                    +-----------------------------------+
                                    |  Event Router                     |
                                    |    DOM events -> signal writes    |
                                    +-----------------------------------+
                                              |
                                        DOM + CSS (browser does layout)
```

---

## Layer 1: Reactive Core (Signal Arena)

Signals are `Copy`-able handles into a slab. Effects auto-track which signals they read.

```rust
struct Runtime {
    signals: Vec<SlotData>,       // arena of signal values (type-erased)
    effects: Vec<EffectData>,     // registered effects
    tracking: Option<EffectId>,   // which effect is currently executing
    dirty: Vec<EffectId>,         // effects that need re-run
}

#[derive(Copy, Clone)]
struct Signal<T> {
    id: u32,
    _phantom: PhantomData<T>,
}

struct Cx<'a> {
    runtime: &'a mut Runtime,
}

impl<T: 'static> Signal<T> {
    fn get(&self, cx: &Cx) -> &T {
        // 1. If an effect is currently running, record this signal as a dependency
        if let Some(effect_id) = cx.runtime.tracking {
            cx.runtime.effects[effect_id].deps.insert(self.id);
        }
        // 2. Return the value
        cx.runtime.signals[self.id].value.downcast_ref::<T>()
    }

    fn set(&self, cx: &mut Cx, value: T) {
        // 1. Store the new value
        cx.runtime.signals[self.id].value = Box::new(value);
        // 2. Mark all dependent effects as dirty
        for effect in &cx.runtime.effects {
            if effect.deps.contains(&self.id) {
                cx.runtime.dirty.push(effect.id);
            }
        }
        // 3. Flush dirty effects (microtask-style batching)
        cx.runtime.flush();
    }
}
```

Key property: `Signal<T>` is `Copy`. The `Runtime` owns all data, signals are just indices.
Same pattern as ECS entity handles or arena-allocated AST nodes.

Derived signals (computed values) are effects that write to another signal:

```rust
let full_name = cx.signal(String::new());
cx.effect(move |cx| {
    let f = first.get(cx);
    let l = last.get(cx);
    full_name.set(cx, format!("{f} {l}"));
});
```

## Layer 2: View Description

Views produce `Node` values — lightweight descriptions of what DOM elements should exist.

```rust
enum Node {
    Element {
        tag: Tag,                        // Div, Span, Input, Button, ...
        styles: Vec<(&'static str, String)>,  // inline style properties
        classes: Vec<String>,            // CSS class names
        attrs: SmallVec<Attr, 4>,        // id, href, etc.
        events: SmallVec<EventBinding, 2>,
        children: Vec<Node>,
    },
    Text(String),
    Reactive(Box<dyn Fn(&Cx) -> Node>),  // re-evaluates when signals change
    Empty,
}
```

The `Reactive` variant wraps a closure that reads signals and produces a `Node`.
When those signals change, only that closure re-runs and only that DOM subtree gets patched.

## Layer 3: Styling — The Hybrid Approach

Three mechanisms, each handling what it's good at:

### Layout builders (inline styles)

Shorthand methods on `NodeBuilder` that emit inline style properties:

```rust
impl NodeBuilder {
    fn row(self) -> Self       { self.style("display", "flex").style("flex-direction", "row") }
    fn col(self) -> Self       { self.style("display", "flex").style("flex-direction", "column") }
    fn gap(self, px: u32) -> Self     { self.style("gap", &format!("{px}px")) }
    fn width(self, px: u32) -> Self   { self.style("width", &format!("{px}px")) }
    fn width_pct(self, p: u32) -> Self { self.style("width", &format!("{p}%")) }
    fn height(self, px: u32) -> Self  { self.style("height", &format!("{px}px")) }
    fn padding(self, px: u32) -> Self { self.style("padding", &format!("{px}px")) }
    fn padding_xy(self, x: u32, y: u32) -> Self {
        self.style("padding", &format!("{y}px {x}px"))
    }
    fn margin_auto(self) -> Self      { self.style("margin", "0 auto") }
    fn grow(self) -> Self      { self.style("flex-grow", "1") }
    fn shrink(self, n: u32) -> Self   { self.style("flex-shrink", &format!("{n}")) }
    fn align_center(self) -> Self     { self.style("align-items", "center") }
    fn justify_between(self) -> Self  { self.style("justify-content", "space-between") }
    fn align_right(self) -> Self      { self.style("text-align", "right") }
    fn bold(self) -> Self      { self.style("font-weight", "500") }
    fn tabular_nums(self) -> Self     { self.style("font-variant-numeric", "tabular-nums") }
    fn cursor_pointer(self) -> Self   { self.style("cursor", "pointer") }
    fn overflow_hidden(self) -> Self  { self.style("overflow", "hidden") }
    fn relative(self) -> Self  { self.style("position", "relative") }
    fn absolute(self) -> Self  { self.style("position", "absolute") }
    fn border_radius(self, px: u32) -> Self { self.style("border-radius", &format!("{px}px")) }
    fn bg(self, c: Color) -> Self     { self.style("background", &c.to_css()) }
    fn color(self, c: Color) -> Self  { self.style("color", &c.to_css()) }
    // ... add as needed, each is one line
}
```

Usage reads like a layout DSL:

```rust
div().row().gap(8).align_center().padding(16).children([...])
```

### Interactive states (generated CSS classes)

Pseudo-classes (`:hover`, `:active`, `:focus`) and transitions cannot be inline styles.
Generate real CSS rules from Rust and inject them into a `<style>` element:

```rust
fn register_styles(cx: &mut Cx) {
    cx.css_class("arrow-btn", css()
        .size(28, 28).center().border_radius(4).cursor_pointer().select_none()
        .transition("background", 0.1, Ease)
        .hover(|s| s.background(0xe0e0e0))
        .active(|s| s.background(0xd0d0d0))
    );

    cx.css_class("slider-thumb", css()
        .size(16, 16).round().background(theme::ACCENT)
        .transition("box-shadow", 0.15, Ease)
        .hover(|s| s.box_shadow(0, 0, 0, 4, theme::ACCENT.alpha(0.2)))
    );

    cx.css_class("tab", css()
        .padding_xy(16, 8).cursor_pointer()
        .border_bottom(2, Color::TRANSPARENT).margin_bottom(-2)
        .transition("border-color", 0.2, Ease).transition("color", 0.2, Ease)
        .hover(|s| s.border_bottom_color(0xcccccc))
    );

    cx.css_class("tab-active", css()
        .border_bottom_color(theme::ACCENT).color(theme::ACCENT)
    );
}
```

Under the hood:

```rust
impl Cx {
    fn css_class(&mut self, name: &str, def: CssDef) {
        if self.registered_classes.contains(name) { return; }
        self.registered_classes.insert(name.to_string());

        let mut rules = String::new();
        write!(rules, ".{name} {{ {} }}", def.base_properties());
        if let Some(hover) = &def.hover {
            write!(rules, ".{name}:hover {{ {} }}", hover.properties());
        }
        if let Some(active) = &def.active {
            write!(rules, ".{name}:active {{ {} }}", active.properties());
        }
        if let Some(focus) = &def.focus {
            write!(rules, ".{name}:focus {{ {} }}", focus.properties());
        }
        self.stylesheet.append(&rules);
    }
}
```

Widgets reference by name:

```rust
div().class("arrow-btn").on_click(...).child(text("\u{25C0}"))
```

### Theme (CSS custom properties)

Design tokens as Rust constants that inject as CSS custom properties:

```rust
mod theme {
    pub const ACCENT: Color = Color::rgb(0x21, 0x96, 0xf3);
    pub const BG: Color = Color::rgb(0x1a, 0x1a, 0x2e);
    pub const SURFACE: Color = Color::rgb(0x25, 0x25, 0x3a);
    pub const TEXT: Color = Color::rgb(0xe0, 0xe0, 0xe0);
    pub const TEXT_MUTED: Color = Color::rgb(0x88, 0x88, 0x99);
    pub const RADIUS: u32 = 4;
    pub const SPACING: u32 = 8;
}

// Inject once at startup — both Rust code and generated CSS can use these
fn inject_theme(theme: &Theme) {
    let root = document().document_element().unwrap();
    let style = root.unchecked_ref::<web_sys::HtmlElement>().style();
    style.set_property("--accent", &theme.accent.to_css()).unwrap();
    style.set_property("--bg", &theme.bg.to_css()).unwrap();
    // ... etc
}
```

### Dynamic values (reactive inline styles)

Signal-driven styles use `reactive_style`:

```rust
div().class("slider-fill").reactive_style(move |cx| {
    format!("width:{}%", value.get(cx) * 100.0)
})
```

### Why this split works

| What                  | Mechanism              | Why not the others                              |
|-----------------------|------------------------|-------------------------------------------------|
| Layout (flex/pad/gap) | Builder -> inline      | Per-instance, structural, no pseudo needed       |
| Hover/active/focus    | Generated CSS classes  | Inline styles can't express pseudo-classes       |
| Transitions/keyframes | Generated CSS classes  | Inline styles can't express `@keyframes`         |
| Theme colors/spacing  | CSS custom properties  | One source of truth for both CSS and Rust        |
| Signal-driven values  | Reactive inline styles | Must update at runtime per signal change         |

Inline styles can't express: `:hover`, `:active`, `:focus`, `::placeholder`, `@media`, `@keyframes`.
So anything interactive or responsive must be a CSS class.

---

## Layer 4: DOM Patcher

Fine-grained patching, no virtual DOM diffing:

```rust
struct Patcher {
    runtime: Rc<RefCell<Runtime>>,
}

impl Patcher {
    fn mount(&self, node: &Node, parent: &web_sys::Element) {
        match node {
            Node::Element { tag, styles, classes, attrs, events, children } => {
                let el = document().create_element(tag.as_str());
                el.set_attribute("style", &styles_to_string(styles));
                for cls in classes { el.class_list().add_1(cls).unwrap(); }
                for attr in attrs { el.set_attribute(&attr.name, &attr.value).unwrap(); }
                for event in events { self.bind_event(&el, event); }
                for child in children { self.mount(child, &el); }
                parent.append_child(&el);
            }
            Node::Text(t) => {
                parent.append_child(&document().create_text_node(t));
            }
            Node::Reactive(f) => {
                let marker = document().create_comment("reactive");
                parent.append_child(&marker);
                let marker = marker.clone();
                self.runtime.borrow_mut().create_effect(move |cx| {
                    let node = f(cx);
                    clear_after(&marker);
                    self.mount(&node, &marker.parent_node().unwrap());
                });
            }
            Node::Empty => {}
        }
    }
}
```

Targeted reactive bindings avoid full subtree rebuilds:

```rust
// reactive_class: updates only the class attribute
fn reactive_class(el: &web_sys::Element, f: impl Fn(&Cx) -> &str + 'static) {
    let el = el.clone();
    runtime.create_effect(move |cx| {
        el.set_attribute("class", f(cx)).unwrap();
    });
}

// reactive_style: updates only the style attribute
fn reactive_style(el: &web_sys::Element, f: impl Fn(&Cx) -> String + 'static) {
    let el = el.clone();
    runtime.create_effect(move |cx| {
        el.set_attribute("style", &f(cx)).unwrap();
    });
}

// reactive_attr: updates a single attribute (special-case "value" for input elements)
fn reactive_attr(el: &web_sys::Element, name: &str, f: impl Fn(&Cx) -> String + 'static) {
    let el = el.clone();
    let name = name.to_string();
    runtime.create_effect(move |cx| {
        let val = f(cx);
        if name == "value" {
            // Set property, not attribute — preserves cursor position
            el.dyn_ref::<web_sys::HtmlInputElement>().unwrap().set_value(&val);
        } else {
            el.set_attribute(&name, &val).unwrap();
        }
    });
}
```

## Layer 5: Event Router

DOM events bound at mount time, routed to signal writes:

```rust
struct EventBinding {
    event: &'static str,       // "click", "input", "keydown", ...
    handler: Rc<dyn Fn(&mut Cx, web_sys::Event)>,
}

fn bind_event(el: &web_sys::Element, binding: &EventBinding) {
    let handler = binding.handler.clone();
    let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
        let mut cx = Cx::new(&RUNTIME);
        handler(&mut cx, event);
        // runtime flushes dirty effects -> DOM updates
    }) as Box<dyn FnMut(web_sys::Event)>);
    el.add_event_listener_with_callback(
        binding.event, closure.as_ref().unchecked_ref()
    ).unwrap();
    closure.forget(); // lives as long as the element
}
```

Flow: DOM event -> Rust closure -> signal write -> effect re-run -> DOM patch.

Document-level listeners for patterns like drag and click-outside:

```rust
impl Cx {
    // Register a document-level event listener (for drag, click-outside, etc.)
    fn on_document<E: JsCast + 'static>(
        &mut self,
        event: &str,
        handler: impl Fn(&mut Cx, E) + 'static,
    ) {
        let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
            let mut cx = Cx::new(&RUNTIME);
            handler(&mut cx, event.dyn_into::<E>().unwrap());
        }) as Box<dyn FnMut(web_sys::Event)>);
        document().add_event_listener_with_callback(event, closure.as_ref().unchecked_ref()).unwrap();
        closure.forget();
    }
}

// Click-outside detection
fn register_click_outside(el: &web_sys::Element, handler: impl Fn(&mut Cx) + 'static) {
    let el = el.clone();
    let closure = Closure::wrap(Box::new(move |event: web_sys::Event| {
        let target = event.target().unwrap().dyn_into::<web_sys::Node>().unwrap();
        if !el.contains(Some(&target)) {
            let mut cx = Cx::new(&RUNTIME);
            handler(&mut cx);
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    document().add_event_listener_with_callback("click", closure.as_ref().unchecked_ref()).unwrap();
    closure.forget();
}

// Element ref capture (for getBoundingClientRect, focus, etc.)
fn element_ref() -> Rc<RefCell<Option<web_sys::Element>>> {
    Rc::new(RefCell::new(None))
}
```

## Layer 6: WebSocket Integration

The WebSocket is another source of signal writes, using the existing Codec trait:

```rust
fn connect_backend(cx: &mut Cx, state: Signal<AppData>) {
    let ws = WebSocket::new("ws://localhost:8080").unwrap();

    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data = event.data().dyn_into::<ArrayBuffer>().unwrap();
        let bytes = js_sys::Uint8Array::new(&data).to_vec();
        let app_data = AppData::decode(&bytes); // Codec trait
        let mut cx = Cx::new(&RUNTIME);
        state.set(&mut cx, app_data);
    }) as Box<dyn FnMut(MessageEvent)>);

    ws.set_onmessage(Some(on_message.as_ref().unchecked_ref()));
    on_message.forget();
}
```

Sending actions back: event handlers serialize an action enum and send over WebSocket.
`#[derive(Codec)]` works both directions.

## Global Runtime

The `Runtime` needs global access from event callbacks and WASM entry points.
A thread-local `RefCell<Runtime>` or `static RefCell<Runtime>` is safe in WASM (single-threaded).
One global mutable runtime is fine — signals eliminate per-widget mutability.

## What This Skips vs Flutter

| Flutter Has              | Skipped          | Why                           |
|--------------------------|------------------|-------------------------------|
| Layout engine (RenderBox)| CSS flexbox/grid | Browser does it better        |
| Paint engine (Skia)      | CSS rendering    | Browser does it               |
| Text shaping (libTxt)    | Browser text     | Browser does it               |
| Hit testing              | DOM event dispatch | Browser does it             |
| Widget lifecycle         | Signal create/drop | Arena handles cleanup       |
| InheritedWidget/Provider | Signal handles   | `Copy` handles solve this     |
| BuildContext tree walking| Direct signal access | No tree to walk             |

## Implementation Order

1. **Signal arena** — `Runtime`, `Signal<T>`, `create_effect`, dependency tracking, flush. ~200-300 lines.
2. **Node + builder API** — `Node` enum, `div()`, `text()`, `reactive()`, layout builders. ~200 lines.
3. **Patcher** — mount `Node` to real DOM via `web-sys`. ~150 lines.
4. **Events** — `on_click`, `on_input`, bind to DOM, route to signal writes. ~100 lines.
5. **CSS generation** — `css_class()`, `CssDef` builder, stylesheet injection. ~150 lines.
6. **WebSocket** — connect, decode into signals, encode actions back.

Steps 1-5: ~800 lines of Rust for a working reactive frontend with styling.

---

## Widgets

Widgets are plain functions that return `Node` values, capturing `Signal` handles for state.
Every widget follows the same pattern:

1. **Create local signals** for UI-only state (open/closed, hover, drag, focus)
2. **Accept shared signals** as parameters for data that flows between widgets
3. **Return a `Node`** built with the builder API
4. **Use `reactive()`** for parts that change
5. **Use `.class()`** for interactive states (hover, active, focus)
6. **Use layout builders** (`.row()`, `.gap()`, `.padding()`) for structure

### Button

CSS handles push animation and color change. Rust only handles the click.

```rust
// Registered once:
cx.css_class("btn", css()
    .padding_xy(16, 8).border(1, 0xcccccc).border_radius(4)
    .background(0xf0f0f0).cursor_pointer().select_none()
    .transition("transform", 0.1, Ease).transition("background", 0.15, Ease)
    .hover(|s| s.background(0xe0e0e0))
    .active(|s| s.scale(0.96).background(0xd0d0d0))
);

fn button_widget(label: &str, on_click: impl Fn(&mut Cx) + 'static) -> Node {
    element("button").class("btn")
        .on_click(move |cx, _| on_click(cx))
        .child(text(label))
}

// Reactive label variant
fn button_reactive(label: Signal<String>, on_click: impl Fn(&mut Cx) + 'static) -> Node {
    element("button").class("btn")
        .on_click(move |cx, _| on_click(cx))
        .child(reactive(move |cx| text(label.get(cx))))
}

// Disabled state via signal-driven class
fn button_stateful(
    label: &str,
    disabled: Signal<bool>,
    on_click: impl Fn(&mut Cx) + 'static,
) -> Node {
    element("button")
        .reactive_class(move |cx| if *disabled.get(cx) { "btn btn-disabled" } else { "btn" })
        .on_click(move |cx, _| { if !*disabled.get(cx) { on_click(cx); } })
        .child(text(label))
}
```

### Dropdown

One signal for open/closed. CSS animates the menu.

```rust
// Registered once:
cx.css_class("dropdown-menu", css()
    .absolute().top_full().left(0).min_width(160)
    .background(0xffffff).border(1, 0xcccccc).border_radius(4)
    .shadow(0, 2, 8, Color::BLACK.alpha(0.12)).z_index(100)
    .transform_origin("top")
    .transition("opacity", 0.15, Ease).transition("transform", 0.15, Ease)
);
cx.css_class("dropdown-closed", css().opacity(0).scale_y(0).pointer_events_none());
cx.css_class("dropdown-open", css().opacity(1).scale_y(1));
cx.css_class("dropdown-item", css()
    .padding_xy(12, 8).cursor_pointer()
    .hover(|s| s.background(0xf0f0f0))
);

fn dropdown(
    cx: &mut Cx,
    items: &[(&str, u32)],
    selected: Signal<u32>,
) -> Node {
    let open = cx.signal(false);
    let items: Rc<Vec<(String, u32)>> = Rc::new(
        items.iter().map(|(l, v)| (l.to_string(), *v)).collect()
    );

    div().relative().child(
        // Trigger button
        element("button").class("btn")
            .on_click(move |cx, _| open.set(cx, !*open.get(cx)))
            .child(reactive({
                let items = items.clone();
                move |cx| {
                    let val = *selected.get(cx);
                    let label = items.iter()
                        .find(|(_, v)| *v == val)
                        .map(|(l, _)| l.as_str())
                        .unwrap_or("Select...");
                    text(label)
                }
            }))
    ).child(
        // Menu
        reactive({
            let items = items.clone();
            move |cx| {
                let is_open = *open.get(cx);
                let cls = if is_open { "dropdown-menu dropdown-open" }
                          else { "dropdown-menu dropdown-closed" };
                div().class(cls).children(
                    items.iter().map(|(label, value)| {
                        let value = *value;
                        let label = label.clone();
                        div().class("dropdown-item")
                            .on_click(move |cx, _| {
                                selected.set(cx, value);
                                open.set(cx, false);
                            })
                            .child(text(&label))
                    })
                )
            }
        })
    )
    .on_click_outside(move |cx| open.set(cx, false))
}
```

### Tab Bar

One signal for active index. CSS transitions the active indicator.

```rust
// Registered once (see css_class examples above for "tab" and "tab-active")

fn tab_bar(cx: &mut Cx, active: Signal<usize>, labels: &[&str]) -> Node {
    div().row().style("border-bottom", "2px solid #e0e0e0").children(
        labels.iter().enumerate().map(|(i, label)| {
            div()
                .class("tab")
                .css_if(active, move |a| *a == i, "tab-active")
                .on_click(move |cx, _| active.set(cx, i))
                .child(text(label))
        })
    )
}

fn tab_page(
    active: Signal<usize>,
    index: usize,
    content: impl Fn(&mut Cx) -> Node + 'static,
) -> Node {
    reactive(move |cx| {
        if *active.get(cx) == index { content(cx) } else { empty() }
    })
}
```

`css_if` is a convenience — it watches a signal and toggles a class:

```rust
fn css_if<T: 'static>(
    self,
    signal: Signal<T>,
    predicate: impl Fn(&T) -> bool + 'static,
    class: &'static str,
) -> Self {
    // Creates a reactive_class binding that adds/removes the class
}
```

### Text Field

Two-way binding: DOM input events write to signal, signal flows back to DOM.

```rust
// Registered once:
cx.css_class("text-field", css()
    .padding_xy(12, 8).border(1, 0xcccccc).border_radius(4).outline_none()
    .transition("border-color", 0.2, Ease).transition("box-shadow", 0.2, Ease)
    .focus(|s| s.border_color(theme::ACCENT).box_shadow(0, 0, 0, 2, theme::ACCENT.alpha(0.2)))
);

fn text_field(value: Signal<String>, placeholder: &str) -> Node {
    element("input").class("text-field")
        .attr("type", "text")
        .attr("placeholder", placeholder)
        .reactive_attr("value", move |cx| value.get(cx).clone())
        .on_input(move |cx, event| {
            let input: web_sys::HtmlInputElement = event.target().unwrap().dyn_into().unwrap();
            value.set(cx, input.value());
        })
}
```

Important: `reactive_attr("value", ...)` sets the DOM property (not HTML attribute) to preserve cursor position.

### Slider

Pointer drag handling — the one widget with raw pointer events.

```rust
// Registered once:
cx.css_class("slider-track", css()
    .relative().height(4).background(0xe0e0e0).border_radius(2).cursor_pointer()
);
cx.css_class("slider-fill", css()
    .absolute().height_pct(100).background(theme::ACCENT).border_radius(2)
);
cx.css_class("slider-thumb", css()
    .absolute().top_pct(50).size(16, 16).round().background(theme::ACCENT)
    .transform("translate(-50%, -50%)")
    .transition("box-shadow", 0.15, Ease)
    .hover(|s| s.box_shadow(0, 0, 0, 4, theme::ACCENT.alpha(0.2)))
);

fn slider(cx: &mut Cx, value: Signal<f32>, min: f32, max: f32) -> Node {
    let dragging = cx.signal(false);
    let track = cx.element_ref();
    let range = max - min;

    let to_pct = move |cx: &Cx| -> f32 { (value.get(cx) - min) / range * 100.0 };

    let from_pointer = {
        let track = track.clone();
        move |client_x: f64| -> f32 {
            let rect = track.get().get_bounding_client_rect();
            let frac = ((client_x - rect.left()) / rect.width()).clamp(0.0, 1.0) as f32;
            min + frac * range
        }
    };

    cx.on_document("mousemove", move |cx, e: MouseEvent| {
        if *dragging.get(cx) {
            value.set(cx, from_pointer(e.client_x() as f64));
        }
    });
    cx.on_document("mouseup", move |cx, _: MouseEvent| {
        dragging.set(cx, false);
    });

    div().padding_xy(0, 8).child(
        div().class("slider-track").ref_el(&track)
            .on_mousedown(move |cx, e: MouseEvent| {
                value.set(cx, from_pointer(e.client_x() as f64));
                dragging.set(cx, true);
            })
            .children([
                div().class("slider-fill").reactive_style(move |cx| {
                    format!("width:{}%", to_pct(cx))
                }),
                div().class("slider-thumb").reactive_style(move |cx| {
                    format!("left:{}%", to_pct(cx))
                }),
            ])
    )
}
```

### Popup / Dialog

#### Overlay popup

Signal controls visibility. Backdrop click closes.

```rust
// Registered once:
cx.css_class("popup-overlay", css()
    .fixed().inset(0).background(Color::BLACK.alpha(0.4))
    .row().align_center().justify_center().z_index(1000)
);
cx.css_class("popup-panel", css()
    .background(0xffffff).border_radius(8)
    .shadow(0, 8, 32, Color::BLACK.alpha(0.2))
    .min_width(300).max_width_pct(80).max_height_pct(80).overflow_auto()
);

fn popup(visible: Signal<bool>, title: &str, content: Node) -> Node {
    let title = title.to_string();
    reactive(move |cx| {
        if !*visible.get(cx) { return empty(); }
        div().class("popup-overlay")
            .on_click(move |cx, e| {
                if e.target() == e.current_target() { visible.set(cx, false); }
            })
            .child(div().class("popup-panel").children([
                div().row().justify_between().padding(16)
                    .child(element("h3").child(text(&title)))
                    .child(button_widget("\u{2715}", move |cx| visible.set(cx, false))),
                div().padding(16).child(content.clone()),
            ]))
    })
}
```

#### Navigation-based popup (separate URL)

For bookmarkable/shareable popups:

```rust
#[derive(Clone, PartialEq)]
enum Route {
    Home,
    Settings,
    Item(u32),
    NotFound,
}

fn parse_route(path: &str) -> Route {
    match path {
        "/" => Route::Home,
        "/settings" => Route::Settings,
        p if p.starts_with("/item/") => {
            p[6..].parse().map(Route::Item).unwrap_or(Route::NotFound)
        }
        _ => Route::NotFound,
    }
}

fn router(route: Signal<Route>) -> Node {
    // Listen to popstate (browser back/forward)
    let closure = Closure::wrap(Box::new(move |_: web_sys::Event| {
        let path = window().location().pathname().unwrap();
        let mut cx = Cx::new(&RUNTIME);
        route.set(&mut cx, parse_route(&path));
    }) as Box<dyn FnMut(web_sys::Event)>);
    window().add_event_listener_with_callback("popstate", closure.as_ref().unchecked_ref()).unwrap();
    closure.forget();

    reactive(move |cx| {
        match route.get(cx) {
            Route::Home => home_view(),
            Route::Settings => popup_page("Settings", settings_panel()),
            Route::Item(id) => popup_page(&format!("Item {id}"), item_view(*id)),
            Route::NotFound => text("404"),
        }
    })
}

fn navigate(cx: &mut Cx, route: Signal<Route>, new_route: Route, path: &str) {
    window().history().unwrap()
        .push_state_with_url(&JsValue::NULL, "", Some(path)).unwrap();
    route.set(cx, new_route);
}
```

---

## Realistic Example: Channel Sliders + Video Feed

### Shared types (in `shared` crate)

```rust
#[derive(Clone, Codec)]
struct ChannelState {
    channels: Vec<ChannelInfo>,
}

#[derive(Clone, Codec)]
struct ChannelInfo {
    name: String,
    value: f32,
    min: f32,
    max: f32,
    step: f32,
}

#[derive(Clone, Codec)]
enum BackendMsg {
    State(ChannelState),
    Frame(Vec<u8>),            // raw jpeg bytes
}

#[derive(Clone, Codec)]
enum FrontendMsg {
    SetChannel { index: usize, value: f32 },
}
```

### Frontend app

```rust
fn app(cx: &mut Cx, ws: &Ws) -> Node {
    theme::DARK.inject();
    register_styles(cx);

    let tab = cx.signal(0usize);
    let channels = cx.signal(ChannelState::default());
    let frame_url = cx.signal(String::new());

    ws.on_message(move |cx, msg: BackendMsg| {
        match msg {
            BackendMsg::State(s) => channels.set(cx, s),
            BackendMsg::Frame(jpeg) => {
                revoke_if_exists(frame_url.get(cx));
                frame_url.set(cx, blob_url(&jpeg, "image/jpeg"));
            }
        }
    });

    div().col().gap(0).children([
        tab_bar(cx, tab, &["Channels", "Video"]),
        div().grow().children([
            tab_page(tab, 0, move |cx| channel_page(cx, ws, channels)),
            tab_page(tab, 1, move |cx| video_page(cx, frame_url)),
        ]),
    ])
}

fn channel_page(cx: &mut Cx, ws: &Ws, channels: Signal<ChannelState>) -> Node {
    div().col().gap(12).padding(16).child(
        reactive(move |cx| {
            let state = channels.get(cx);
            div().col().gap(12).children(
                state.channels.iter().enumerate().map(|(i, ch)| {
                    channel_row(cx, ws, i, ch)
                })
            )
        })
    )
}

fn channel_row(cx: &mut Cx, ws: &Ws, index: usize, info: &ChannelInfo) -> Node {
    let val = cx.signal(info.value);
    let (min, max, step) = (info.min, info.max, info.step);

    cx.effect(move |cx| {
        ws.send(cx, FrontendMsg::SetChannel { index, value: *val.get(cx) });
    });

    let nudge = move |cx: &mut Cx, d: f32| {
        val.set(cx, (*val.get(cx) + d).clamp(min, max));
    };

    div().row().gap(8).align_center().children([
        div().width(120).bold().child(text(&info.name)),
        div().class("arrow-btn").on_click(move |cx, _| nudge(cx, -step)).child(text("\u{25C0}")),
        slider(cx, val, min, max),
        div().class("arrow-btn").on_click(move |cx, _| nudge(cx, step)).child(text("\u{25B6}")),
        div().width(60).align_right().tabular_nums()
            .child(reactive(move |cx| text(&format!("{:.1}", val.get(cx))))),
    ])
}

fn video_page(cx: &mut Cx, frame_url: Signal<String>) -> Node {
    div().col().padding(16).gap(8).children([
        div().bold().child(text("Live Feed")),
        reactive(move |cx| {
            let url = frame_url.get(cx);
            if url.is_empty() {
                div().padding(64).align_center().color(theme::TEXT_MUTED)
                    .child(text("Waiting for frames..."))
            } else {
                element("img").attr("src", url).width_pct(100).border_radius(4)
            }
        }),
    ])
}
```

### Ergonomics assessment

**What feels good:**

- Composition is function calls. The tree structure is visible in the code:
  `div().row().gap(8).children([label, arrow, slider, arrow, readout])`
- Signals are invisible plumbing. `nudge` captures `val`, `min`, `max` — all Copy.
  No Rc, no RefCell, no context.read<T>().
- Refactoring is fearless. Extract subtree -> function, pass signals as args.
  Compiler tells you if you forgot one.
- Layout reads like a DSL: `.row().gap(8).align_center()` is nearly as terse as CSS.

**What's clunky:**

- `move` on every closure. Unavoidable in Rust, visual noise.
- `reactive()` wrapping. Every signal-dependent piece needs it.
  Forgetting it = UI doesn't update. #1 new-user mistake.
  A proc macro (rsx!) could auto-detect signal reads, but that's a big investment.
- `cx` threading. Every function that creates signals needs `&mut Cx`.
  Alternative: thread-local implicit (Leptos does this). Tradeoff: ergonomics vs clarity.
- List rebuilding. When `channels` signal updates, the entire list rebuilds.
  Fine for 10 items, need keyed reconciliation for 1000+:
  `keyed_each(channels, |ch| ch.id, |cx, ch| channel_row(cx, ws, ch))`

**Caveats:**

- Blob URL lifecycle: frames arriving while video tab is hidden still create/revoke blobs.
  Need mount/unmount hooks or gate on active tab.
- Signal-per-channel vs monolithic: local `val` signal is disconnected from backend pushes.
  Use `cx.derived()` for read-only signals computed from a parent:
  `let ch_value = cx.derived(move |cx| channels.get(cx).channels[index].value);`
- JPEG frame rate: 30fps = 30 signal sets/sec. Fine (O(1) per frame), but keep
  frame-dependent reactives minimal.
- No CSS devtools iteration: inline styles from builders appear as `style="..."` in inspector.
  You can read them but editing doesn't feed back to Rust. Solo Rust dev iterating in code
  anyway — probably fine.

### Concept count comparison

| Flutter                         | This framework    |
|---------------------------------|-------------------|
| StatelessWidget                 | fn(...) -> Node   |
| StatefulWidget + State<T>       | fn + cx.signal()  |
| build()                         | (the function body)|
| setState()                      | signal.set()      |
| initState() / dispose()         | (just code before the return) |
| didUpdateWidget()               | (reactive effects)|
| InheritedWidget / Provider      | Signal parameters |
| Consumer / Selector             | reactive()        |
| BuildContext                    | &mut Cx           |
| Navigator / Routes              | Signal<Route>     |
| AnimationController             | CSS transitions   |
| CustomPainter                   | (not needed — CSS)|

Total: ~12 Flutter concepts -> ~4 concepts (signal, effect, reactive, Node builders).

---

## Backend: HTTP Server + WebSocket

Thread-per-connection with blocking I/O. No async needed on the server side —
the executor and channels handle app logic, the HTTP/WS layer is pure blocking I/O
on dedicated threads.

### Architecture

```
  Main Thread                    HTTP Thread               Client Threads (one per connection)
 +-------------------+         +-------------------+      +----------------------------+
 |  App logic        |         |  TcpListener      |      |  Connection handler        |
 |  Executor         |<--Arc-->|  accept() loop    |----->|  HTTP parse + route        |
 |  Channels         |         |  thread::spawn    |      |  Static files / WS upgrade |
 +-------------------+         +-------------------+      +----------------------------+
```

The HTTP server runs entirely on its own threads. It communicates with the app
through `Arc<App>` and channels. Clean separation: I/O plumbing on threads,
app logic wherever you want it.

### Server Entry Point

```rust
fn main() {
    let app = Arc::new(App::new());

    // HTTP server on its own thread
    let app_clone = app.clone();
    std::thread::spawn(move || {
        let listener = TcpListener::bind("0.0.0.0:8080").unwrap();
        for stream in listener.incoming() {
            let stream = stream.unwrap();
            let app = app_clone.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_connection(stream, &app) {
                    eprintln!("connection error: {e}");
                }
            });
        }
    });

    // Main thread runs app logic
    app.run();
}
```

### Connection Handler

```rust
fn handle_connection(mut stream: TcpStream, app: &App) -> io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut filled = 0;

    loop {
        let n = stream.read(&mut buf[filled..])?;  // blocks — that's fine
        if n == 0 { return Ok(()); }
        filled += n;

        match HttpRequest::parse(&buf[..filled]) {
            Ok((request, consumed)) => {
                buf.drain(..consumed);
                filled -= consumed;

                let keep_alive = request.header("connection")
                    .map(|v| v.eq_ignore_ascii_case("keep-alive"))
                    .unwrap_or(true);

                // WebSocket upgrade
                if request.is_websocket_upgrade() {
                    return handle_websocket(stream, &request, app);
                }

                // Static file / API
                let response = route(&request, app);
                stream.write_all(&response.to_bytes())?;

                if !keep_alive { return Ok(()); }
            }
            Err(ParseState::NeedMoreData) => {
                if filled == buf.len() { buf.resize(filled + 8192, 0); }
            }
            Err(ParseState::BadRequest) => {
                stream.write_all(&HttpResponse::new(400).to_bytes())?;
                return Ok(());
            }
        }
    }
}
```

### HTTP Request / Response

```rust
struct HttpRequest {
    method: Method,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    headers: Vec<(&'static str, String)>,
    body: Vec<u8>,
}

enum Method { Get, Post, Put, Delete, Head, Options }

impl HttpRequest {
    fn parse(buf: &[u8]) -> Result<(Self, usize), ParseState> {
        // Find \r\n\r\n (end of headers)
        let header_end = find_double_crlf(buf)
            .ok_or(ParseState::NeedMoreData)?;

        let header_str = std::str::from_utf8(&buf[..header_end])
            .map_err(|_| ParseState::BadRequest)?;

        let mut lines = header_str.split("\r\n");

        // Request line: "GET /path HTTP/1.1"
        let request_line = lines.next().ok_or(ParseState::BadRequest)?;
        let mut parts = request_line.split(' ');
        let method = Method::parse(parts.next().ok_or(ParseState::BadRequest)?)?;
        let path = parts.next().ok_or(ParseState::BadRequest)?.to_string();

        // Headers
        let mut headers = Vec::new();
        let mut content_length = 0usize;
        for line in lines {
            if line.is_empty() { break; }
            if let Some((key, value)) = line.split_once(": ") {
                if key.eq_ignore_ascii_case("content-length") {
                    content_length = value.parse().unwrap_or(0);
                }
                headers.push((key.to_lowercase(), value.to_string()));
            }
        }

        // Body
        let body_start = header_end + 4;
        let total = body_start + content_length;
        if buf.len() < total { return Err(ParseState::NeedMoreData); }
        let body = buf[body_start..total].to_vec();

        Ok((HttpRequest { method, path, headers, body }, total))
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn is_websocket_upgrade(&self) -> bool {
        self.header("upgrade")
            .map(|v| v.eq_ignore_ascii_case("websocket"))
            .unwrap_or(false)
    }
}

impl HttpResponse {
    fn new(status: u16) -> Self {
        HttpResponse { status, headers: Vec::new(), body: Vec::new() }
    }

    fn header(mut self, key: &'static str, value: impl Into<String>) -> Self {
        self.headers.push((key, value.into()));
        self
    }

    fn body(mut self, data: Vec<u8>) -> Self {
        self.body = data;
        self
    }

    fn to_bytes(&self) -> Vec<u8> {
        let reason = match self.status {
            101 => "Switching Protocols", 200 => "OK", 304 => "Not Modified",
            400 => "Bad Request", 404 => "Not Found", 500 => "Internal Server Error",
            _ => "Unknown",
        };
        let mut buf = format!("HTTP/1.1 {} {}\r\n", self.status, reason);
        write!(buf, "Content-Length: {}\r\n", self.body.len()).unwrap();
        for (key, value) in &self.headers {
            write!(buf, "{key}: {value}\r\n").unwrap();
        }
        buf.push_str("\r\n");
        let mut bytes = buf.into_bytes();
        bytes.extend_from_slice(&self.body);
        bytes
    }
}
```

### Routing and Static File Serving

```rust
fn route(request: &HttpRequest, app: &App) -> HttpResponse {
    match (&request.method, request.path.as_str()) {
        (Method::Get, "/api/status") => {
            HttpResponse::new(200)
                .header("Content-Type", "application/json")
                .body(b"{\"status\":\"ok\"}".to_vec())
        }
        (Method::Get, path) => serve_static("dist", path),
        _ => HttpResponse::new(404).body(b"Not Found".to_vec()),
    }
}

fn serve_static(dir: &str, path: &str) -> HttpResponse {
    let req_path = path.trim_start_matches('/');
    let req_path = if req_path.is_empty() { "index.html" } else { req_path };

    // Reject path traversal
    if req_path.contains("..") {
        return HttpResponse::new(400).body(b"Bad Request".to_vec());
    }

    let file_path = Path::new(dir).join(req_path);
    let data = match std::fs::read(&file_path) {
        Ok(data) => data,
        Err(_) => return HttpResponse::new(404).body(b"Not Found".to_vec()),
    };

    let mime = match file_path.extension().and_then(|e| e.to_str()) {
        Some("html") => "text/html; charset=utf-8",
        Some("css")  => "text/css; charset=utf-8",
        Some("js")   => "application/javascript; charset=utf-8",
        Some("wasm") => "application/wasm",
        Some("json") => "application/json",
        Some("png")  => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("svg")  => "image/svg+xml",
        Some("ico")  => "image/x-icon",
        _ => "application/octet-stream",
    };

    let cache = match file_path.extension().and_then(|e| e.to_str()) {
        Some("wasm") | Some("js") => "public, max-age=31536000, immutable",
        Some("html") => "no-cache",
        _ => "public, max-age=3600",
    };

    HttpResponse::new(200)
        .header("Content-Type", mime)
        .header("Cache-Control", cache)
        .body(data)
}
```

### WebSocket (Blocking)

```rust
fn handle_websocket(stream: TcpStream, request: &HttpRequest, app: &App) -> io::Result<()> {
    // Upgrade handshake
    let key = request.header("sec-websocket-key")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing key"))?;
    let accept = websocket_accept_key(key);

    let response = HttpResponse::new(101)
        .header("Upgrade", "websocket")
        .header("Connection", "Upgrade")
        .header("Sec-WebSocket-Accept", accept);

    let mut writer_stream = stream.try_clone()?;
    writer_stream.write_all(&response.to_bytes())?;

    // Split into reader (this thread) + writer (spawned thread)
    let mut reader = WebSocket::new(stream);
    let mut writer = WebSocket::new(writer_stream);

    // Send initial state
    writer.send(&BackendMsg::State(app.current_state()))?;

    // Writer thread — receives messages from app via channel
    let (tx, rx) = std::sync::mpsc::channel();
    app.register_client(tx.clone());

    let writer_handle = std::thread::spawn(move || {
        while let Ok(msg) = rx.recv() {
            if writer.send_raw(&msg).is_err() { break; }
        }
    });

    // Read loop (this thread) — blocks on socket read
    loop {
        match reader.recv::<FrontendMsg>() {
            Ok(msg) => app.handle_message(msg),
            Err(_) => break,
        }
    }

    app.unregister_client(&tx);
    writer_handle.join().ok();
    Ok(())
}

struct WebSocket {
    stream: TcpStream,
}

impl WebSocket {
    fn new(stream: TcpStream) -> Self {
        WebSocket { stream }
    }

    fn read_frame(&mut self) -> io::Result<Frame> {
        let mut header = [0u8; 2];
        self.stream.read_exact(&mut header)?;

        let opcode = header[0] & 0x0F;
        let masked = header[1] & 0x80 != 0;
        let mut len = (header[1] & 0x7F) as u64;

        if len == 126 {
            let mut buf = [0u8; 2];
            self.stream.read_exact(&mut buf)?;
            len = u16::from_be_bytes(buf) as u64;
        } else if len == 127 {
            let mut buf = [0u8; 8];
            self.stream.read_exact(&mut buf)?;
            len = u64::from_be_bytes(buf);
        }

        let mask = if masked {
            let mut buf = [0u8; 4];
            self.stream.read_exact(&mut buf)?;
            Some(buf)
        } else { None };

        let mut payload = vec![0u8; len as usize];
        self.stream.read_exact(&mut payload)?;

        if let Some(mask) = mask {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }

        Ok(Frame { opcode, payload })
    }

    fn write_frame(&mut self, opcode: u8, data: &[u8]) -> io::Result<()> {
        let mut header = vec![0x80 | opcode];
        if data.len() < 126 {
            header.push(data.len() as u8);
        } else if data.len() < 65536 {
            header.push(126);
            header.extend_from_slice(&(data.len() as u16).to_be_bytes());
        } else {
            header.push(127);
            header.extend_from_slice(&(data.len() as u64).to_be_bytes());
        }
        self.stream.write_all(&header)?;
        self.stream.write_all(data)
    }

    fn send<T: Codec>(&mut self, msg: &T) -> io::Result<()> {
        self.write_frame(0x02, &msg.encode())
    }

    fn recv<T: Codec>(&mut self) -> io::Result<T> {
        loop {
            let frame = self.read_frame()?;
            match frame.opcode {
                0x02 => return Ok(T::decode(&frame.payload)),
                0x08 => return Err(io::Error::new(io::ErrorKind::ConnectionReset, "close")),
                0x09 => self.write_frame(0x0A, &frame.payload)?,
                _ => {}
            }
        }
    }
}
```

### WebSocket Accept Key

Requires SHA-1 and Base64 (the only non-trivial utility code):

```rust
fn websocket_accept_key(key: &str) -> String {
    let mut input = key.to_string();
    input.push_str("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    let hash = sha1(input.as_bytes());
    base64_encode(&hash)
}
```

SHA-1 is ~70 lines to implement (the algorithm is public and well-documented).
Base64 encode is ~30 lines.

### Multi-Client Broadcasting

The `App` struct manages connected clients:

```rust
struct App {
    state: Mutex<AppState>,
    clients: Mutex<Vec<std::sync::mpsc::Sender<Vec<u8>>>>,
}

impl App {
    fn register_client(&self, tx: std::sync::mpsc::Sender<Vec<u8>>) {
        self.clients.lock().unwrap().push(tx);
    }

    fn unregister_client(&self, tx: &std::sync::mpsc::Sender<Vec<u8>>) {
        self.clients.lock().unwrap().retain(|c| !std::sync::mpsc::Sender::same_channel(c, tx));
    }

    fn broadcast(&self, msg: &impl Codec) {
        let bytes = msg.encode();
        self.clients.lock().unwrap().retain(|tx| tx.send(bytes.clone()).is_ok());
    }

    fn handle_message(&self, msg: FrontendMsg) {
        match msg {
            FrontendMsg::SetChannel { index, value } => {
                self.state.lock().unwrap().update_channel(index, value);
                self.broadcast(&BackendMsg::State(self.current_state()));
            }
        }
    }
}
```

### Line Count

| Component | Lines |
|-----------|-------|
| Server loop (accept + spawn) | ~15 |
| Connection handler | ~50 |
| HTTP request parser | ~60 |
| HTTP response builder | ~40 |
| Routing + static files | ~60 |
| WebSocket (blocking) | ~100 |
| SHA-1 + Base64 | ~100 |
| Multi-client broadcast | ~30 |
| **Total** | **~455** |

### Tradeoffs

**Why this works:**

- Blocking code is trivially readable — no `async`, no `Pin`, no `Poll`, no reactor
- `std::net` and `std::thread` are zero-dependency, stable, well-tested
- Thread-per-connection is correct for the expected client count (a few frontends)
- Clean separation: I/O threads know nothing about app logic, app knows nothing about sockets

**When it wouldn't work:**

- 1000+ concurrent connections (thread count / stack memory)
- Internet-facing without a reverse proxy (no TLS, no slowloris protection)
- Need for HTTP/2 or HTTP/3 (irrelevant for local tool UI)

**TLS:** Skip it. Use plain HTTP for localhost/LAN. For remote access, put
nginx or caddy in front — they handle TLS, HTTP/2, and abuse protection,
your server stays simple.
