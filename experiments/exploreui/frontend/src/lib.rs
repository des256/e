//! ExploreUI frontend — reactive WASM UI with WebSocket sync.
//!
//! Renders three sliders (x, y, z) bound to a [`Vec3<f32>`] signal and
//! synchronizes the value with the backend via binary WebSocket.

use {
    base::{vec3, Vec3},
    shared::{Codec, Message},
    std::cell::Cell,
    webui::*,
};

// -- websocket FFI --

#[link(wasm_import_module = "env")]
extern "C" {
    fn ws_send(ptr: *const u8, len: usize);
}

/// Send a binary message over the WebSocket connection.
fn send_ws(data: &[u8]) {
    unsafe { ws_send(data.as_ptr(), data.len()) }
}

/// Encode and send the current value.
fn send_value(v: Vec3<f32>) {
    let mut buf = Vec::new();
    Message::Value(v).encode(&mut buf);
    send_ws(&buf);
}

// -- global signal --

thread_local! {
    static VALUE_SIGNAL: Cell<Option<Signal<Vec3<f32>>>> = Cell::new(None);
}

// -- helpers --

/// Build a labeled slider row for one component of the Vec3.
fn slider_row(
    label: &'static str,
    value: Signal<Vec3<f32>>,
    get: fn(Vec3<f32>) -> f32,
    set: fn(Vec3<f32>, f32) -> Vec3<f32>,
) -> Node {
    div()
        .row()
        .gap(16)
        .align_center()
        .child(span().child(text(label)).style("width", "16px"))
        .child(
            element("input")
                .attr("type", "range")
                .attr("min", "0")
                .attr("max", "100")
                .attr("step", "0.1")
                .style("width", "400px")
                .reactive_attr("value", move |cx| {
                    get(value.get(cx)).to_string()
                })
                .on_input(move |cx, e| {
                    let val_str = e.target().get_value();
                    if let Ok(f) = val_str.parse::<f32>() {
                        let v = set(value.get(cx), f);
                        value.set(cx, v);
                        send_value(v);
                    }
                }),
        )
        .child(reactive(move |cx| {
            text(&format!("{:.1}", get(value.get(cx))))
        }))
        .into()
}

// -- entry points --

/// Initialize the UI and mount it into the document body.
#[no_mangle]
pub extern "C" fn main() {
    with_context(|cx| {
        let value = cx.signal(vec3(0.0f32, 0.0, 0.0));
        VALUE_SIGNAL.with(|s| s.set(Some(value)));

        let view = div()
            .col()
            .gap(16)
            .padding(24)
            .child(text("ExploreUI — Vec3 Editor"))
            .child(slider_row("X", value, |v| v.x, |v, f| vec3(f, v.y, v.z)))
            .child(slider_row("Y", value, |v| v.y, |v, f| vec3(v.x, f, v.z)))
            .child(slider_row("Z", value, |v| v.z, |v, f| vec3(v.x, v.y, f)));

        mount(view.into(), ffi::body());
    });
}

/// Called from JS when a binary WebSocket message arrives.
#[no_mangle]
pub extern "C" fn on_ws_message(ptr: *const u8, len: usize) {
    let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
    if let Ok((Message::Value(v), _)) = Message::decode(bytes) {
        VALUE_SIGNAL.with(|sig| {
            if let Some(signal) = sig.get() {
                with_context(|cx| {
                    signal.set(cx, v);
                });
            }
        });
    }
}
