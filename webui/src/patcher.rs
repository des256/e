use crate::{
    ffi::{self, Element, Event, JsHandle, MouseEvent, KeyboardEvent},
    node::{ElementData, Node},
    runtime::Context,
};

// -- mount --

/// Mount a [`Node`] tree into a parent DOM element.
///
/// Recursively creates real DOM nodes from the tree. Reactive nodes
/// set up effects that re-patch only their subtree when signals change.
pub fn mount(node: Node, parent: Element) {
    match node {
        Node::Element(data) => mount_element(data, parent),
        Node::Text(t) => {
            let text_node = ffi::create_text_node_str(&t);
            parent.append_child(text_node);
        }
        Node::Reactive(f) => mount_reactive(f, parent),
        Node::Empty => {}
    }
}

// -- element mounting --

fn mount_element(data: ElementData, parent: Element) {
    let el = Element::create(data.tag);

    // Static inline styles.
    if !data.styles.is_empty() {
        let s: String = data.styles.iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(";");
        el.set_attribute("style", &s);
    }

    // Static classes.
    for cls in &data.classes {
        el.class_list_add(cls);
    }

    // Static attributes.
    for (name, value) in &data.attrs {
        el.set_attribute(name, value);
    }

    // Event listeners.
    for binding in &data.events {
        let handler = binding.handler.clone();
        let cb_id = ffi::register_callback(move |event_handle| {
            handler(&Context::new(), Event(event_handle));
        });
        el.add_event_listener(binding.name, cb_id);
    }

    // Reactive class binding.
    if let Some(f) = data.reactive_class {
        let el2 = el;
        Context::new().effect(move |context| {
            el2.set_attribute("class", &f(context));
        });
    }

    // Reactive style binding.
    if let Some(f) = data.reactive_style {
        let el2 = el;
        Context::new().effect(move |context| {
            el2.set_attribute("style", &f(context));
        });
    }

    // Reactive attribute bindings.
    for (name, f) in data.reactive_attrs {
        let el2 = el;
        Context::new().effect(move |context| {
            let val = f(context);
            if name == "value" {
                el2.set_value(&val);
            } else {
                el2.set_attribute(name, &val);
            }
        });
    }

    // Children.
    for child in data.children {
        mount(child, el);
    }

    parent.append_child_element(el);
}

// -- reactive mounting --

/// Mount a reactive node: place start/end comment markers, then create an
/// effect that rebuilds only the content between them when dependencies change.
fn mount_reactive(f: Box<dyn Fn(&Context) -> Node>, parent: Element) {
    let start = ffi::create_comment_str("reactive");
    let end = ffi::create_comment_str("/reactive");
    parent.append_child(start);
    parent.append_child(end);

    Context::new().effect(move |context| {
        clear_between(start, end);
        let node = f(context);
        let tmp = Element::create("div");
        mount(node, tmp);
        while let Some(child) = tmp.first_child() {
            parent.insert_before(child, Some(end));
        }
    });
}

/// Remove all nodes strictly between two sibling markers.
fn clear_between(start: JsHandle, end: JsHandle) {
    let parent = start.parent_element();
    loop {
        match start.next_sibling() {
            Some(sibling) if sibling == end => break,
            Some(sibling) => parent.remove_child(sibling),
            None => break,
        }
    }
}

// -- document-level events --

impl Context {
    /// Register a document-level mouse event listener.
    ///
    /// Useful for drag handling and click-outside detection.
    pub fn on_document_mouse(
        &self,
        event: &str,
        handler: impl Fn(&Context, MouseEvent) + 'static,
    ) {
        let cb_id = ffi::register_callback(move |event_handle| {
            handler(&Context::new(), MouseEvent(event_handle));
        });
        ffi::document_add_event_listener_str(event, cb_id);
    }

    /// Register a document-level keyboard event listener.
    pub fn on_document_keyboard(
        &self,
        event: &str,
        handler: impl Fn(&Context, KeyboardEvent) + 'static,
    ) {
        let cb_id = ffi::register_callback(move |event_handle| {
            handler(&Context::new(), KeyboardEvent(event_handle));
        });
        ffi::document_add_event_listener_str(event, cb_id);
    }
}
