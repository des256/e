use std::rc::Rc;

use crate::{
    ffi::Event,
    runtime::Context,
};

// -- node tree --

/// Lightweight description of a DOM subtree.
///
/// Views produce `Node` values. The [patcher](crate::patcher::mount)
/// turns them into real DOM elements. Build element nodes via the
/// [`ElementBuilder`](crate::builder::ElementBuilder) builder; use
/// [`text`](crate::builder::text) and [`reactive`](crate::builder::reactive)
/// for the other variants.
pub enum Node {
    /// A DOM element with tag, attributes, styles, events, and children.
    Element(ElementData),
    /// A text node.
    Text(String),
    /// A reactive subtree. The closure re-evaluates when its signal
    /// dependencies change; only this subtree is re-patched.
    Reactive(Box<dyn Fn(&Context) -> Node>),
    /// Produces no DOM output.
    Empty,
}

// -- element data --

/// All data for a single DOM element, built by
/// [`ElementBuilder`](crate::builder::ElementBuilder).
pub struct ElementData {
    /// HTML tag name (`"div"`, `"span"`, `"input"`, etc.).
    pub(crate) tag: &'static str,
    /// Static HTML attributes (`id`, `href`, `type`, etc.).
    pub(crate) attrs: Vec<(&'static str, String)>,
    /// Static inline style properties.
    pub(crate) styles: Vec<(&'static str, String)>,
    /// Static CSS class names.
    pub(crate) classes: Vec<String>,
    /// Event listener bindings.
    pub(crate) events: Vec<EventBinding>,
    /// Child nodes.
    pub(crate) children: Vec<Node>,
    /// Reactive inline style closure, set via
    /// [`ElementBuilder::reactive_style`](crate::builder::ElementBuilder::reactive_style).
    pub(crate) reactive_style: Option<Box<dyn Fn(&Context) -> String>>,
    /// Reactive class name closure, set via
    /// [`ElementBuilder::reactive_class`](crate::builder::ElementBuilder::reactive_class).
    pub(crate) reactive_class: Option<Box<dyn Fn(&Context) -> String>>,
    /// Reactive attribute closures, set via
    /// [`ElementBuilder::reactive_attr`](crate::builder::ElementBuilder::reactive_attr).
    pub(crate) reactive_attrs: Vec<(&'static str, Box<dyn Fn(&Context) -> String>)>,
}

// -- event binding --

/// An event listener binding stored in the node tree.
///
/// Created by [`ElementBuilder::on`](crate::builder::ElementBuilder::on) and the typed
/// convenience wrappers
/// ([`ElementBuilder::on_click`](crate::builder::ElementBuilder::on_click), etc.).
pub struct EventBinding {
    /// DOM event name (`"click"`, `"input"`, `"mousedown"`, etc.).
    pub(crate) name: &'static str,
    /// Handler closure. Receives the reactive [`Context`] and a raw
    /// [`Event`] handle.
    pub(crate) handler: Rc<dyn Fn(&Context, Event)>,
}
