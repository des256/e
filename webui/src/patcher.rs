use crate::{
    ffi::{self, Element, Event, JsHandle, MouseEvent, KeyboardEvent},
    node::{ElementData, Node},
    runtime::Context,
};

// -- ownership --

/// Release `handle` when the running effect is torn down. At root scope
/// there is no owner and the node lives for the rest of the program.
fn owned(handle: JsHandle) {
    Context::new().on_cleanup(move || handle.release());
}

// -- mount --

/// Mount a [`Node`] tree into a parent DOM element.
///
/// Recursively creates real DOM nodes from the tree. Every node, event
/// callback and effect created here belongs to the running effect and is
/// released when that effect re-runs or is disposed; a mount at root
/// scope lives for the rest of the program. Reactive nodes set up effects
/// that rebuild only their own subtree when signals change.
pub fn mount(node: Node, parent: Element) {
    match node {
        Node::Element(data) => mount_element(data, parent),
        Node::Text(t) => {
            let text_node = ffi::create_text_node_str(&t);
            owned(text_node);
            parent.append_child(text_node);
        }
        Node::Reactive(f) => mount_reactive(f, parent),
        Node::Empty => {}
    }
}

// -- element mounting --

fn mount_element(data: ElementData, parent: Element) {
    let el = Element::create(data.tag);
    owned(el.into());

    // Static inline styles.
    if !data.styles.is_empty() {
        let s: String = data.styles.iter()
            .map(|(k, v)| format!("{k}:{v}"))
            .collect::<Vec<_>>()
            .join(";");
        el.set_attribute("style", &s);
    }

    // Static classes; a class string may hold several tokens.
    for cls in &data.classes {
        for token in cls.split_whitespace() {
            el.class_list_add(token);
        }
    }

    // Static attributes.
    for (name, value) in &data.attrs {
        el.set_attribute(name, value);
    }

    // Event listeners, freed with the element.
    for binding in &data.events {
        let handler = binding.handler.clone();
        let cb_id = ffi::register_callback(move |event_handle| {
            handler(&Context::new(), Event(event_handle));
        });
        Context::new().on_cleanup(move || ffi::unregister_callback(cb_id));
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

    // Children — mounted before reactive attrs so that <select> elements
    // have their <option> children when set_value runs.
    for child in data.children {
        mount(child, el);
    }

    // Reactive attribute bindings (after children).
    for (name, f) in data.reactive_attrs {
        let el2 = el;
        Context::new().effect(move |context| {
            let val = f(context);
            if name == "value" {
                el2.set_value(&val);
            } else if name == "checked" {
                el2.set_checked(!val.is_empty());
            } else {
                el2.set_attribute(name, &val);
            }
        });
    }

    parent.append_child_element(el);
}

// -- reactive mounting --

/// Mount a reactive node: place start/end comment markers, then create
/// an effect that rebuilds the subtree between them when deps change.
///
/// The markers belong to the enclosing scope. Everything between them
/// belongs to the effect: before each rebuild the runtime tears the
/// previous run down (nested effects, listeners, handles), and the body
/// then removes the old nodes and mounts the new ones in their place.
fn mount_reactive(f: Box<dyn Fn(&Context) -> Node>, parent: Element) {
    let start = ffi::create_comment_str("reactive");
    let end = ffi::create_comment_str("/reactive");
    owned(start);
    owned(end);
    parent.append_child(start);
    parent.append_child(end);

    Context::new().effect(move |context| {
        ffi::remove_siblings_between(start, end);
        // The parent is read here, not captured: a block mounted into a
        // fragment is moved into the document afterwards.
        let parent = end.parent_element();
        let fragment = ffi::create_fragment_node();
        mount(f(context), fragment);
        parent.insert_before(fragment.into(), Some(end));
        fragment.release();
    });
}

// -- document-level events --

impl Context {
    /// Register a document-level mouse event listener.
    ///
    /// Useful for drag handling and click-outside detection. Document
    /// listeners live for the rest of the program; register them at root
    /// scope, not inside a reactive block.
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

    /// Register a document-level keyboard event listener. Lives for the
    /// rest of the program, like [`on_document_mouse`](Self::on_document_mouse).
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
