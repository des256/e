import { webui_imports, webui_init } from "./webui.js";

let wasm = null;
let ws = null;

const app_env = {
    ws_send(ptr, len) {
        const bytes = new Uint8Array(wasm.exports.memory.buffer, ptr, len);
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(bytes.slice());
        }
    },
};

const env = { ...webui_imports(), ...app_env };

const result = await WebAssembly.instantiateStreaming(
    fetch("app.wasm"),
    { env },
);

wasm = result.instance;
webui_init(result.instance);
wasm.exports.main();

// Connect WebSocket.
const protocol = location.protocol === "https:" ? "wss:" : "ws:";
ws = new WebSocket(`${protocol}//${location.host}/ws`);
ws.binaryType = "arraybuffer";

ws.onmessage = (event) => {
    const data = new Uint8Array(event.data);
    const ptr = wasm.exports.alloc(data.length);
    new Uint8Array(wasm.exports.memory.buffer, ptr, data.length).set(data);
    wasm.exports.on_ws_message(ptr, data.length);
    wasm.exports.dealloc(ptr, data.length);
};
