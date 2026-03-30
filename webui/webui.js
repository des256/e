// webui.js — JS glue for the webui WASM frontend framework.
//
// Provides DOM operations, a handle table for JS objects, callback
// dispatch, and string marshalling. No external dependencies.
//
// Usage:
//
//   import { webui_imports, webui_init } from "./webui.js";
//
//   const wasm = await WebAssembly.instantiateStreaming(
//       fetch("app.wasm"),
//       { env: webui_imports() },
//   );
//   webui_init(wasm.instance);

// -- handle table --

// Index 0 is permanently null (represents None/missing).
const handles = [null];

// Reverse lookup: reuse existing handle for objects already tracked.
const obj_to_handle = new WeakMap();

function handle_alloc(obj) {
    const id = handles.length;
    handles.push(obj);
    if (typeof obj === "object" && obj !== null) {
        obj_to_handle.set(obj, id);
    }
    return id;
}

// Return existing handle for an object if one exists, otherwise allocate.
function handle_for(obj) {
    if (obj === null || obj === undefined) return 0;
    const existing = obj_to_handle.get(obj);
    if (existing !== undefined) return existing;
    return handle_alloc(obj);
}

// -- callback state --

// Each callback ID gets a pre-allocated handle slot for its event
// argument. Reused on every dispatch to avoid unbounded table growth.
const cb_event_slots = [];

// -- WASM instance (set by webui_init) --

let wasm = null;

// -- string helpers --

const decoder = new TextDecoder("utf-8");
const encoder = new TextEncoder();

function read_str(ptr, len) {
    return decoder.decode(new Uint8Array(wasm.exports.memory.buffer, ptr, len));
}

function write_str(s, buf_ptr, buf_len) {
    const encoded = encoder.encode(s);
    const copy_len = Math.min(encoded.length, buf_len);
    const target = new Uint8Array(wasm.exports.memory.buffer, buf_ptr, copy_len);
    target.set(encoded.subarray(0, copy_len));
    return encoded.length;
}

// -- import functions --

export function webui_imports() {
    return {
        // -- document --

        create_element(tag_ptr, tag_len) {
            return handle_alloc(document.createElement(read_str(tag_ptr, tag_len)));
        },

        create_text_node(ptr, len) {
            return handle_alloc(document.createTextNode(read_str(ptr, len)));
        },

        create_comment(ptr, len) {
            return handle_alloc(document.createComment(read_str(ptr, len)));
        },

        document_head() {
            return handle_for(document.head);
        },

        document_add_event_listener(name_ptr, name_len, cb_id) {
            const name = read_str(name_ptr, name_len);
            const slot = cb_event_slots[cb_id];
            document.addEventListener(name, (event) => {
                handles[slot] = event;
                wasm.exports.callback_dispatch(cb_id, slot);
            });
        },

        // -- element --

        set_attribute(handle, name_ptr, name_len, val_ptr, val_len) {
            handles[handle].setAttribute(
                read_str(name_ptr, name_len),
                read_str(val_ptr, val_len),
            );
        },

        class_list_add(handle, cls_ptr, cls_len) {
            handles[handle].classList.add(read_str(cls_ptr, cls_len));
        },

        add_event_listener(handle, name_ptr, name_len, cb_id) {
            const name = read_str(name_ptr, name_len);
            const slot = cb_event_slots[cb_id];
            handles[handle].addEventListener(name, (event) => {
                handles[slot] = event;
                wasm.exports.callback_dispatch(cb_id, slot);
            });
        },

        set_value(handle, ptr, len) {
            handles[handle].value = read_str(ptr, len);
        },

        append_text_content(handle, ptr, len) {
            handles[handle].textContent += read_str(ptr, len);
        },

        // -- node tree --

        append_child(parent, child) {
            handles[parent].appendChild(handles[child]);
        },

        insert_before(parent, node, ref_handle) {
            const ref_node = ref_handle === 0 ? null : handles[ref_handle];
            handles[parent].insertBefore(handles[node], ref_node);
        },

        remove_child(parent, child) {
            handles[parent].removeChild(handles[child]);
        },

        first_child(handle) {
            return handle_for(handles[handle].firstChild);
        },

        next_sibling(handle) {
            return handle_for(handles[handle].nextSibling);
        },

        parent_node(handle) {
            return handle_for(handles[handle].parentNode);
        },

        // -- event properties --

        event_target(handle) {
            return handle_for(handles[handle].target);
        },

        event_current_target(handle) {
            return handle_for(handles[handle].currentTarget);
        },

        mouse_event_client_x(handle) {
            return handles[handle].clientX | 0;
        },

        mouse_event_client_y(handle) {
            return handles[handle].clientY | 0;
        },

        keyboard_event_key(handle, buf_ptr, buf_len) {
            return write_str(handles[handle].key, buf_ptr, buf_len);
        },

        // -- callbacks --

        register_event_slot(cb_id) {
            // Pre-allocate a handle slot for this callback's event argument.
            const slot = handle_alloc(null);
            cb_event_slots[cb_id] = slot;
            return slot;
        },
    };
}

// -- initialization --

export function webui_init(instance) {
    wasm = instance;
}
