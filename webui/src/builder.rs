use {
    crate::{
        color::Color,
        ffi::{Event, MouseEvent, KeyboardEvent},
        node::{ElementData, EventBinding, Node},
        runtime::Context,
    },
    std::rc::Rc,
};

// -- element builder --

/// Element builder. Chain layout, style, event, and child methods,
/// then convert to [`Node`] via [`.into()`](Into::into).
///
/// Constructed by [`div`], [`span`], or [`element`]. Named
/// `ElementBuilder` to distinguish from [`ffi::Element`](crate::ffi::Element)
/// (the DOM handle type).
pub struct ElementBuilder {
    data: ElementData,
}

impl ElementBuilder {
    fn new(tag: &'static str) -> Self {
        ElementBuilder {
            data: ElementData {
                tag,
                attrs: Vec::new(),
                styles: Vec::new(),
                classes: Vec::new(),
                events: Vec::new(),
                children: Vec::new(),
                reactive_style: None,
                reactive_class: None,
                reactive_attrs: Vec::new(),
            },
        }
    }
}

/// `ElementBuilder` → `Node::Element`.
impl From<ElementBuilder> for Node {
    fn from(element: ElementBuilder) -> Node {
        Node::Element(element.data)
    }
}

// -- raw setters --

impl ElementBuilder {
    /// Set an HTML attribute.
    pub fn attr(mut self, name: &'static str, value: &str) -> Self {
        self.data.attrs.push((name, value.to_string()));
        self
    }

    /// Set an inline style property.
    pub fn style(mut self, prop: &'static str, value: &str) -> Self {
        self.data.styles.push((prop, value.to_string()));
        self
    }

    /// Add a CSS class name.
    pub fn class(mut self, name: &str) -> Self {
        self.data.classes.push(name.to_string());
        self
    }
}

// -- layout builders --

impl ElementBuilder {
    /// `display:flex; flex-direction:row`.
    pub fn row(self) -> Self {
        self.style("display", "flex").style("flex-direction", "row")
    }

    /// `display:flex; flex-direction:column`.
    pub fn col(self) -> Self {
        self.style("display", "flex").style("flex-direction", "column")
    }

    /// `gap` in pixels.
    pub fn gap(self, px: u32) -> Self {
        self.style("gap", &format!("{px}px"))
    }

    /// Fixed width in pixels.
    pub fn width(self, px: u32) -> Self {
        self.style("width", &format!("{px}px"))
    }

    /// Width as a percentage.
    pub fn width_pct(self, p: u32) -> Self {
        self.style("width", &format!("{p}%"))
    }

    /// Fixed height in pixels.
    pub fn height(self, px: u32) -> Self {
        self.style("height", &format!("{px}px"))
    }

    /// Height as a percentage.
    pub fn height_pct(self, p: u32) -> Self {
        self.style("height", &format!("{p}%"))
    }

    /// Uniform padding in pixels.
    pub fn padding(self, px: u32) -> Self {
        self.style("padding", &format!("{px}px"))
    }

    /// Horizontal and vertical padding in pixels.
    pub fn padding_xy(self, x: u32, y: u32) -> Self {
        self.style("padding", &format!("{y}px {x}px"))
    }

    /// `margin: 0 auto` (horizontally centered).
    pub fn margin_auto(self) -> Self {
        self.style("margin", "0 auto")
    }

    /// `flex-grow: 1`.
    pub fn grow(self) -> Self {
        self.style("flex-grow", "1")
    }

    /// `flex-shrink`.
    pub fn shrink(self, n: u32) -> Self {
        self.style("flex-shrink", &format!("{n}"))
    }

    /// `align-items: center`.
    pub fn align_center(self) -> Self {
        self.style("align-items", "center")
    }

    /// `justify-content: space-between`.
    pub fn justify_between(self) -> Self {
        self.style("justify-content", "space-between")
    }

    /// `justify-content: center`.
    pub fn justify_center(self) -> Self {
        self.style("justify-content", "center")
    }

    /// `text-align: right`.
    pub fn align_right(self) -> Self {
        self.style("text-align", "right")
    }

    /// `font-weight: 500`.
    pub fn bold(self) -> Self {
        self.style("font-weight", "500")
    }

    /// `font-variant-numeric: tabular-nums`.
    pub fn tabular_nums(self) -> Self {
        self.style("font-variant-numeric", "tabular-nums")
    }

    /// `cursor: pointer`.
    pub fn cursor_pointer(self) -> Self {
        self.style("cursor", "pointer")
    }

    /// `overflow: hidden`.
    pub fn overflow_hidden(self) -> Self {
        self.style("overflow", "hidden")
    }

    /// `position: relative`.
    pub fn relative(self) -> Self {
        self.style("position", "relative")
    }

    /// `position: absolute`.
    pub fn absolute(self) -> Self {
        self.style("position", "absolute")
    }

    /// `border-radius` in pixels.
    pub fn border_radius(self, px: u32) -> Self {
        self.style("border-radius", &format!("{px}px"))
    }

    /// Background color.
    pub fn bg(self, c: impl Into<Color>) -> Self {
        self.style("background", &c.into().to_css())
    }

    /// Text color.
    pub fn color(self, c: impl Into<Color>) -> Self {
        self.style("color", &c.into().to_css())
    }
}

// -- events --

impl ElementBuilder {
    /// Bind a raw DOM event by name.
    pub fn on(
        mut self,
        event: &'static str,
        handler: impl Fn(&Context, Event) + 'static,
    ) -> Self {
        self.data.events.push(EventBinding {
            name: event,
            handler: Rc::new(handler),
        });
        self
    }

    /// Bind a `click` event.
    pub fn on_click(self, handler: impl Fn(&Context, MouseEvent) + 'static) -> Self {
        self.on("click", move |context, e| {
            handler(context, MouseEvent(e.0))
        })
    }

    /// Bind an `input` event.
    pub fn on_input(self, handler: impl Fn(&Context, Event) + 'static) -> Self {
        self.on("input", handler)
    }

    /// Bind a `mousedown` event.
    pub fn on_mousedown(
        self,
        handler: impl Fn(&Context, MouseEvent) + 'static,
    ) -> Self {
        self.on("mousedown", move |context, e| {
            handler(context, MouseEvent(e.0))
        })
    }

    /// Bind a `keydown` event.
    pub fn on_keydown(
        self,
        handler: impl Fn(&Context, KeyboardEvent) + 'static,
    ) -> Self {
        self.on("keydown", move |context, e| {
            handler(context, KeyboardEvent(e.0))
        })
    }
}

// -- reactive bindings --

impl ElementBuilder {
    /// Reactive inline style, re-evaluated when signals change.
    ///
    /// The closure should return a CSS string like `"width:50%"`.
    pub fn reactive_style(mut self, f: impl Fn(&Context) -> String + 'static) -> Self {
        self.data.reactive_style = Some(Box::new(f));
        self
    }

    /// Reactive class name, re-evaluated when signals change.
    pub fn reactive_class(mut self, f: impl Fn(&Context) -> String + 'static) -> Self {
        self.data.reactive_class = Some(Box::new(f));
        self
    }

    /// Reactive attribute, re-evaluated when signals change.
    ///
    /// Special-cases `"value"` to set the DOM property instead of the
    /// HTML attribute, preserving cursor position in input elements.
    pub fn reactive_attr(
        mut self,
        name: &'static str,
        f: impl Fn(&Context) -> String + 'static,
    ) -> Self {
        self.data.reactive_attrs.push((name, Box::new(f)));
        self
    }
}

// -- children --

impl ElementBuilder {
    /// Append a single child node.
    pub fn child(mut self, child: impl Into<Node>) -> Self {
        self.data.children.push(child.into());
        self
    }

    /// Append multiple child nodes.
    pub fn children<I: IntoIterator<Item = Node>>(mut self, children: I) -> Self {
        self.data.children.extend(children);
        self
    }
}

// -- constructors --

/// Create a `<div>` element builder.
pub fn div() -> ElementBuilder {
    ElementBuilder::new("div")
}

/// Create a `<span>` element builder.
pub fn span() -> ElementBuilder {
    ElementBuilder::new("span")
}

/// Create an element builder for an arbitrary HTML tag.
pub fn element(tag: &'static str) -> ElementBuilder {
    ElementBuilder::new(tag)
}

/// Create a text node.
pub fn text(s: &str) -> Node {
    Node::Text(s.to_string())
}

/// Create a reactive node that re-evaluates when signals change.
///
/// Only the subtree produced by the closure is re-patched.
pub fn reactive(f: impl Fn(&Context) -> Node + 'static) -> Node {
    Node::Reactive(Box::new(f))
}

/// Create an empty node (no DOM output).
pub fn empty() -> Node {
    Node::Empty
}
