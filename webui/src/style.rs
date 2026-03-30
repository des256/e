use {
    crate::{
        color::Color,
        ffi,
        runtime::Context,
    },
    std::{
        cell::RefCell,
        collections::HashSet,
        fmt::Write,
    },
};

// -- CSS class definition --

/// CSS class definition builder.
///
/// Build with [`css`], chain style methods, add pseudo-classes with
/// [`hover`](CssDef::hover)/[`active`](CssDef::active)/[`focus`](CssDef::focus),
/// and register via [`Context::css_class`].
///
/// # Examples
///
/// ```ignore
/// use webui::*;
///
/// with_context(|context| {
///     context.css_class("btn", css()
///         .padding_xy(16, 8).border(1, 0xcccccc).border_radius(4)
///         .cursor_pointer().select_none()
///         .transition("background", 0.15, EASE)
///         .hover(|s| s.background(0xe0e0e0))
///         .active(|s| s.background(0xd0d0d0)));
/// });
/// ```
pub struct CssDef {
    entries: Vec<(String, String)>,
    hover_entries: Option<Vec<(String, String)>>,
    active_entries: Option<Vec<(String, String)>>,
    focus_entries: Option<Vec<(String, String)>>,
    transitions: Vec<String>,
}

/// Start building a CSS class definition.
pub fn css() -> CssDef {
    CssDef {
        entries: Vec::new(),
        hover_entries: None,
        active_entries: None,
        focus_entries: None,
        transitions: Vec::new(),
    }
}

// -- raw property --

impl CssDef {
    /// Set an arbitrary CSS property.
    pub fn prop(mut self, name: &str, value: &str) -> Self {
        self.entries.push((name.to_string(), value.to_string()));
        self
    }
}

// -- sizing --

impl CssDef {
    /// Fixed width and height in pixels.
    pub fn size(self, w: u32, h: u32) -> Self {
        self.prop("width", &format!("{w}px")).prop("height", &format!("{h}px"))
    }

    /// Fixed width in pixels.
    pub fn width(self, px: u32) -> Self { self.prop("width", &format!("{px}px")) }

    /// Fixed height in pixels.
    pub fn height(self, px: u32) -> Self { self.prop("height", &format!("{px}px")) }

    /// Height as a percentage.
    pub fn height_pct(self, p: u32) -> Self { self.prop("height", &format!("{p}%")) }

    /// Minimum width in pixels.
    pub fn min_width(self, px: u32) -> Self { self.prop("min-width", &format!("{px}px")) }

    /// Maximum width as a percentage.
    pub fn max_width_pct(self, p: u32) -> Self { self.prop("max-width", &format!("{p}%")) }

    /// Maximum height as a percentage.
    pub fn max_height_pct(self, p: u32) -> Self { self.prop("max-height", &format!("{p}%")) }
}

// -- spacing --

impl CssDef {
    /// Uniform padding in pixels.
    pub fn padding(self, px: u32) -> Self { self.prop("padding", &format!("{px}px")) }

    /// Horizontal and vertical padding in pixels.
    pub fn padding_xy(self, x: u32, y: u32) -> Self {
        self.prop("padding", &format!("{y}px {x}px"))
    }

    /// Bottom margin in pixels (may be negative).
    pub fn margin_bottom(self, px: i32) -> Self { self.prop("margin-bottom", &format!("{px}px")) }
}

// -- positioning --

impl CssDef {
    /// `position: relative`.
    pub fn relative(self) -> Self { self.prop("position", "relative") }

    /// `position: absolute`.
    pub fn absolute(self) -> Self { self.prop("position", "absolute") }

    /// `position: fixed`.
    pub fn fixed(self) -> Self { self.prop("position", "fixed") }

    /// `inset` in pixels.
    pub fn inset(self, px: u32) -> Self { self.prop("inset", &format!("{px}px")) }

    /// `top: 100%`.
    pub fn top_full(self) -> Self { self.prop("top", "100%") }

    /// `top` as a percentage.
    pub fn top_pct(self, p: u32) -> Self { self.prop("top", &format!("{p}%")) }

    /// `left` in pixels.
    pub fn left(self, px: u32) -> Self { self.prop("left", &format!("{px}px")) }

    /// `z-index`.
    pub fn z_index(self, z: i32) -> Self { self.prop("z-index", &format!("{z}")) }
}

// -- display --

impl CssDef {
    /// Flexbox center (both axes).
    pub fn center(self) -> Self {
        self.prop("display", "flex")
            .prop("align-items", "center")
            .prop("justify-content", "center")
    }

    /// `display:flex; flex-direction:row`.
    pub fn row(self) -> Self {
        self.prop("display", "flex").prop("flex-direction", "row")
    }

    /// `justify-content: center`.
    pub fn justify_center(self) -> Self { self.prop("justify-content", "center") }
}

// -- colors --

impl CssDef {
    /// Background color.
    pub fn background(self, c: impl Into<Color>) -> Self {
        self.prop("background", &c.into().to_css())
    }

    /// Text color.
    pub fn color(self, c: impl Into<Color>) -> Self {
        self.prop("color", &c.into().to_css())
    }
}

// -- borders --

impl CssDef {
    /// Solid border.
    pub fn border(self, width: u32, c: impl Into<Color>) -> Self {
        self.prop("border", &format!("{width}px solid {}", c.into().to_css()))
    }

    /// `border-radius` in pixels.
    pub fn border_radius(self, px: u32) -> Self {
        self.prop("border-radius", &format!("{px}px"))
    }

    /// Solid bottom border.
    pub fn border_bottom(self, width: u32, c: impl Into<Color>) -> Self {
        self.prop("border-bottom", &format!("{width}px solid {}", c.into().to_css()))
    }

    /// `border-color`.
    pub fn border_color(self, c: impl Into<Color>) -> Self {
        self.prop("border-color", &c.into().to_css())
    }

    /// `border-bottom-color`.
    pub fn border_bottom_color(self, c: impl Into<Color>) -> Self {
        self.prop("border-bottom-color", &c.into().to_css())
    }

    /// `border-radius: 50%` (circle).
    pub fn round(self) -> Self { self.prop("border-radius", "50%") }

    /// `outline: none`.
    pub fn outline_none(self) -> Self { self.prop("outline", "none") }
}

// -- shadows --

impl CssDef {
    /// `box-shadow` without spread.
    pub fn shadow(self, x: i32, y: i32, blur: u32, c: impl Into<Color>) -> Self {
        self.prop("box-shadow", &format!("{x}px {y}px {blur}px {}", c.into().to_css()))
    }

    /// `box-shadow` with spread.
    pub fn box_shadow(self, x: i32, y: i32, blur: u32, spread: u32, c: impl Into<Color>) -> Self {
        self.prop(
            "box-shadow",
            &format!("{x}px {y}px {blur}px {spread}px {}", c.into().to_css()),
        )
    }
}

// -- misc --

impl CssDef {
    /// `cursor: pointer`.
    pub fn cursor_pointer(self) -> Self { self.prop("cursor", "pointer") }

    /// `user-select: none`.
    pub fn select_none(self) -> Self { self.prop("user-select", "none") }

    /// `pointer-events: none`.
    pub fn pointer_events_none(self) -> Self { self.prop("pointer-events", "none") }

    /// `overflow: auto`.
    pub fn overflow_auto(self) -> Self { self.prop("overflow", "auto") }

    /// `opacity`.
    pub fn opacity(self, v: f32) -> Self { self.prop("opacity", &format!("{v}")) }

    /// `transform: scale(v)`.
    pub fn scale(self, v: f32) -> Self { self.prop("transform", &format!("scale({v})")) }

    /// `transform: scaleY(v)`.
    pub fn scale_y(self, v: f32) -> Self { self.prop("transform", &format!("scaleY({v})")) }

    /// Arbitrary `transform`.
    pub fn transform(self, t: &str) -> Self { self.prop("transform", t) }

    /// `transform-origin`.
    pub fn transform_origin(self, origin: &str) -> Self { self.prop("transform-origin", origin) }
}

// -- transitions --

impl CssDef {
    /// Add a CSS transition.
    ///
    /// Multiple calls are combined into a single `transition` property.
    pub fn transition(mut self, property: &str, duration_s: f32, timing: &str) -> Self {
        self.transitions.push(format!("{property} {duration_s}s {timing}"));
        self
    }
}

// -- pseudo-classes --

impl CssDef {
    /// Define `:hover` styles.
    pub fn hover(mut self, f: impl FnOnce(CssDef) -> CssDef) -> Self {
        self.hover_entries = Some(f(css()).entries);
        self
    }

    /// Define `:active` styles.
    pub fn active(mut self, f: impl FnOnce(CssDef) -> CssDef) -> Self {
        self.active_entries = Some(f(css()).entries);
        self
    }

    /// Define `:focus` styles.
    pub fn focus(mut self, f: impl FnOnce(CssDef) -> CssDef) -> Self {
        self.focus_entries = Some(f(css()).entries);
        self
    }
}

// -- rule generation --

impl CssDef {
    /// Generate the CSS rule text for the given class name.
    pub(crate) fn to_rules(&self, name: &str) -> String {
        let mut out = String::new();
        let mut props = entries_to_css(&self.entries);
        if !self.transitions.is_empty() {
            if !props.is_empty() {
                props.push(';');
            }
            write!(props, "transition:{}", self.transitions.join(",")).unwrap();
        }
        write!(out, ".{name}{{{props}}}").unwrap();
        if let Some(ref entries) = self.hover_entries {
            write!(out, ".{name}:hover{{{}}}", entries_to_css(entries)).unwrap();
        }
        if let Some(ref entries) = self.active_entries {
            write!(out, ".{name}:active{{{}}}", entries_to_css(entries)).unwrap();
        }
        if let Some(ref entries) = self.focus_entries {
            write!(out, ".{name}:focus{{{}}}", entries_to_css(entries)).unwrap();
        }
        out
    }
}

fn entries_to_css(entries: &[(String, String)]) -> String {
    entries.iter().map(|(k, v)| format!("{k}:{v}")).collect::<Vec<_>>().join(";")
}

// -- stylesheet injection --

thread_local! {
    static REGISTERED: RefCell<HashSet<String>> = RefCell::new(HashSet::new());
    static STYLE_EL: RefCell<Option<ffi::Element>> = RefCell::new(None);
}

fn ensure_style_element() -> ffi::Element {
    STYLE_EL.with(|cell| {
        let mut opt = cell.borrow_mut();
        if let Some(el) = *opt {
            return el;
        }
        let el = ffi::Element::create("style");
        let head = ffi::head();
        head.append_child_element(el);
        *opt = Some(el);
        el
    })
}

impl Context {
    /// Register a named CSS class. Duplicate registrations are ignored.
    ///
    /// Generates CSS rules from `def` and appends them to a shared
    /// `<style>` element in the document head.
    pub fn css_class(&self, name: &str, def: CssDef) {
        let already = REGISTERED.with(|r| r.borrow().contains(name));
        if already {
            return;
        }
        REGISTERED.with(|r| {
            r.borrow_mut().insert(name.to_string());
        });
        let rules = def.to_rules(name);
        ensure_style_element().append_text_content(&rules);
    }
}

// -- timing constants --

/// `ease` timing function.
pub const EASE: &str = "ease";
/// `ease-in` timing function.
pub const EASE_IN: &str = "ease-in";
/// `ease-out` timing function.
pub const EASE_OUT: &str = "ease-out";
/// `linear` timing function.
pub const LINEAR: &str = "linear";
