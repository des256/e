use std::{
    alloc::{self, Layout},
    cell::RefCell,
};

// -- extern imports --

#[link(wasm_import_module = "env")]
extern "C" {
    // -- document --

    fn create_element(tag_ptr: *const u8, tag_len: usize) -> u32;
    fn create_text_node(ptr: *const u8, len: usize) -> u32;
    fn create_comment(ptr: *const u8, len: usize) -> u32;
    fn document_head() -> u32;
    fn document_add_event_listener(name_ptr: *const u8, name_len: usize, cb_id: u32);

    // -- element --

    fn set_attribute(
        handle: u32,
        name_ptr: *const u8,
        name_len: usize,
        val_ptr: *const u8,
        val_len: usize,
    );
    fn class_list_add(handle: u32, cls_ptr: *const u8, cls_len: usize);
    fn add_event_listener(handle: u32, name_ptr: *const u8, name_len: usize, cb_id: u32);
    fn set_value(handle: u32, ptr: *const u8, len: usize);
    fn append_text_content(handle: u32, ptr: *const u8, len: usize);

    // -- node tree --

    fn append_child(parent: u32, child: u32);
    fn insert_before(parent: u32, node: u32, ref_handle: u32);
    fn remove_child(parent: u32, child: u32);
    fn first_child(handle: u32) -> u32;
    fn next_sibling(handle: u32) -> u32;
    fn parent_node(handle: u32) -> u32;

    // -- event properties --

    fn event_target(handle: u32) -> u32;
    fn event_current_target(handle: u32) -> u32;
    fn mouse_event_client_x(handle: u32) -> i32;
    fn mouse_event_client_y(handle: u32) -> i32;
    fn keyboard_event_key(handle: u32, buf_ptr: *mut u8, buf_len: usize) -> usize;

    // -- callbacks --

    fn register_event_slot(cb_id: u32) -> u32;
}

// -- handle types --

/// Opaque handle to a JS object. Used for comment nodes, generic DOM
/// nodes, and any JS value that does not need a typed wrapper.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct JsHandle(pub(crate) u32);

/// Handle to a DOM element.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Element(pub(crate) u32);

/// Handle to a DOM event.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Event(pub(crate) u32);

/// Handle to a DOM mouse event.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MouseEvent(pub(crate) u32);

/// Handle to a DOM keyboard event.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct KeyboardEvent(pub(crate) u32);

// -- conversions --

impl From<Element> for JsHandle {
    fn from(el: Element) -> Self { JsHandle(el.0) }
}

impl From<JsHandle> for Element {
    fn from(h: JsHandle) -> Self { Element(h.0) }
}

// -- string helper --

/// Pass a Rust `&str` as (pointer, length) to a closure that calls
/// an extern function expecting raw string data.
fn with_str<R>(s: &str, f: impl FnOnce(*const u8, usize) -> R) -> R {
    f(s.as_ptr(), s.len())
}

// -- document operations --

impl Element {
    /// Create a DOM element with the given tag name.
    pub fn create(tag: &str) -> Element {
        Element(with_str(tag, |p, l| unsafe { create_element(p, l) }))
    }

    /// Set an HTML attribute.
    pub fn set_attribute(&self, name: &str, value: &str) {
        let np = name.as_ptr();
        let nl = name.len();
        let vp = value.as_ptr();
        let vl = value.len();
        unsafe { set_attribute(self.0, np, nl, vp, vl) }
    }

    /// Add a CSS class.
    pub fn class_list_add(&self, cls: &str) {
        with_str(cls, |p, l| unsafe { class_list_add(self.0, p, l) })
    }

    /// Attach an event listener. `cb_id` is from [`register_callback`].
    pub fn add_event_listener(&self, name: &str, cb_id: u32) {
        with_str(name, |p, l| unsafe { add_event_listener(self.0, p, l, cb_id) })
    }

    /// Set the `value` property (for input elements).
    pub fn set_value(&self, value: &str) {
        with_str(value, |p, l| unsafe { set_value(self.0, p, l) })
    }

    /// Append text to `textContent` (for `<style>` elements).
    pub fn append_text_content(&self, text: &str) {
        with_str(text, |p, l| unsafe { append_text_content(self.0, p, l) })
    }

    /// Append a child node.
    pub fn append_child(&self, child: JsHandle) {
        unsafe { append_child(self.0, child.0) }
    }

    /// Append a child element.
    pub fn append_child_element(&self, child: Element) {
        unsafe { append_child(self.0, child.0) }
    }

    /// Insert `node` before `reference`. If `reference` is `None`, appends.
    pub fn insert_before(&self, node: JsHandle, reference: Option<JsHandle>) {
        let ref_h = reference.map_or(0, |h| h.0);
        unsafe { insert_before(self.0, node.0, ref_h) }
    }

    /// Remove a child node.
    pub fn remove_child(&self, child: JsHandle) {
        unsafe { remove_child(self.0, child.0) }
    }

    /// First child node, or `None`.
    pub fn first_child(&self) -> Option<JsHandle> {
        let h = unsafe { first_child(self.0) };
        if h == 0 { None } else { Some(JsHandle(h)) }
    }
}

// -- JsHandle node operations --

impl JsHandle {
    /// Next sibling node, or `None`.
    pub fn next_sibling(&self) -> Option<JsHandle> {
        let h = unsafe { next_sibling(self.0) };
        if h == 0 { None } else { Some(JsHandle(h)) }
    }

    /// Parent node as an [`Element`].
    pub fn parent_element(&self) -> Element {
        Element(unsafe { parent_node(self.0) })
    }
}

// -- document-level operations --

/// Create a text node.
pub fn create_text_node_str(text: &str) -> JsHandle {
    JsHandle(with_str(text, |p, l| unsafe { create_text_node(p, l) }))
}

/// Create a comment node.
pub fn create_comment_str(text: &str) -> JsHandle {
    JsHandle(with_str(text, |p, l| unsafe { create_comment(p, l) }))
}

/// Get the `<head>` element.
pub fn head() -> Element {
    Element(unsafe { document_head() })
}

/// Attach an event listener to the document. `cb_id` is from
/// [`register_callback`].
pub fn document_add_event_listener_str(name: &str, cb_id: u32) {
    with_str(name, |p, l| unsafe { document_add_event_listener(p, l, cb_id) })
}

// -- event property accessors --

impl Event {
    /// The event target element.
    pub fn target(&self) -> Element {
        Element(unsafe { event_target(self.0) })
    }

    /// The element the listener is attached to.
    pub fn current_target(&self) -> Element {
        Element(unsafe { event_current_target(self.0) })
    }
}

impl MouseEvent {
    /// Horizontal cursor position in client coordinates.
    pub fn client_x(&self) -> i32 {
        unsafe { mouse_event_client_x(self.0) }
    }

    /// Vertical cursor position in client coordinates.
    pub fn client_y(&self) -> i32 {
        unsafe { mouse_event_client_y(self.0) }
    }

    /// The event target element.
    pub fn target(&self) -> Element {
        Element(unsafe { event_target(self.0) })
    }

    /// The element the listener is attached to.
    pub fn current_target(&self) -> Element {
        Element(unsafe { event_current_target(self.0) })
    }
}

impl KeyboardEvent {
    /// The key value (e.g. `"Enter"`, `"a"`, `"ArrowUp"`).
    pub fn key(&self) -> String {
        let mut buf = [0u8; 64];
        let len = unsafe { keyboard_event_key(self.0, buf.as_mut_ptr(), buf.len()) };
        let actual = len.min(buf.len());
        String::from_utf8_lossy(&buf[..actual]).into_owned()
    }

    /// The event target element.
    pub fn target(&self) -> Element {
        Element(unsafe { event_target(self.0) })
    }
}

// -- callback registry --

thread_local! {
    static CALLBACKS: RefCell<Vec<Option<Box<dyn FnMut(u32)>>>> =
        RefCell::new(Vec::new());
}

/// Register a callback closure. Returns an integer ID that can be
/// passed to [`Element::add_event_listener`] or
/// [`document_add_event_listener_str`].
///
/// The JS glue pre-allocates a handle slot for each callback's event
/// argument, reusing it on every dispatch.
pub fn register_callback(f: impl FnMut(u32) + 'static) -> u32 {
    CALLBACKS.with(|cbs| {
        let mut cbs = cbs.borrow_mut();
        let id = cbs.len() as u32;
        cbs.push(Some(Box::new(f)));
        // Tell JS to allocate a reusable event handle slot for this ID.
        unsafe { register_event_slot(id) };
        id
    })
}

/// Entry point called by JS when an event fires.
///
/// JS passes the callback ID and the handle of the event object
/// (stored in the pre-allocated slot). The closure is temporarily
/// taken out of the registry so the borrow is released before
/// calling user code (avoids re-entrancy panics if a handler
/// calls [`register_callback`]).
#[no_mangle]
pub extern "C" fn callback_dispatch(cb_id: u32, event_handle: u32) {
    let mut f = CALLBACKS.with(|cbs| {
        cbs.borrow_mut()
            .get_mut(cb_id as usize)
            .and_then(|slot| slot.take())
    });
    if let Some(ref mut f) = f {
        f(event_handle);
    }
    // Put it back (unless a nested register_callback overwrote the slot).
    if let Some(f) = f {
        CALLBACKS.with(|cbs| {
            let mut cbs = cbs.borrow_mut();
            if let Some(slot) = cbs.get_mut(cb_id as usize) {
                if slot.is_none() {
                    *slot = Some(f);
                }
            }
        });
    }
}

// -- memory exports --

/// Allocate `len` bytes of WASM linear memory. Called by JS to write
/// strings into Rust-owned buffers.
#[no_mangle]
pub extern "C" fn alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let layout = Layout::from_size_align(len, 1).unwrap();
    unsafe { alloc::alloc(layout) }
}

/// Free a previously allocated buffer.
#[no_mangle]
pub extern "C" fn dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let layout = Layout::from_size_align(len, 1).unwrap();
    unsafe { alloc::dealloc(ptr, layout) }
}

// -- tests --

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: register a callback bypassing register_event_slot (extern).
    fn test_register(f: impl FnMut(u32) + 'static) -> u32 {
        CALLBACKS.with(|cbs| {
            let mut cbs = cbs.borrow_mut();
            let id = cbs.len() as u32;
            cbs.push(Some(Box::new(f)));
            id
        })
    }

    #[test]
    fn callback_dispatch_invokes_closure() {
        CALLBACKS.with(|cbs| cbs.borrow_mut().clear());
        let called = std::rc::Rc::new(std::cell::Cell::new(0u32));
        let called2 = called.clone();
        let id = test_register(move |handle| { called2.set(handle); });
        callback_dispatch(id, 42);
        assert_eq!(called.get(), 42);
        callback_dispatch(id, 99);
        assert_eq!(called.get(), 99);
    }

    #[test]
    fn callback_dispatch_reentrant_register() {
        CALLBACKS.with(|cbs| cbs.borrow_mut().clear());
        let outer_ran = std::rc::Rc::new(std::cell::Cell::new(false));
        let outer_ran2 = outer_ran.clone();
        let inner_ran = std::rc::Rc::new(std::cell::Cell::new(false));
        let inner_ran2 = inner_ran.clone();
        // Outer callback registers a new callback during dispatch.
        let id = test_register(move |_handle| {
            outer_ran2.set(true);
            let inner = inner_ran2.clone();
            let _new_id = test_register(move |_| { inner.set(true); });
        });
        // Should not panic (no re-entrancy borrow conflict).
        callback_dispatch(id, 0);
        assert!(outer_ran.get());
    }
}
