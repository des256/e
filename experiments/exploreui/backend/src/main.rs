use {
    ::http::{
        websocket::{Message as WsMessage, WebSocket},
        *,
    },
    math::vec3,
    shared::{Codec, Message},
    std::sync::{Arc, Mutex},
};

// -- state --

struct State {
    value: math::Vec3<f32>,
    clients: Vec<WebSocket>,
}

// -- main --

fn main() {
    let state = Arc::new(Mutex::new(State {
        value: vec3(0.0f32, 0.0, 0.0),
        clients: Vec::new(),
    }));

    let addr = "127.0.0.1:8080".parse().unwrap();

    let handle = Server::new(addr, |req| {
        serve::static_file("experiments/exploreui/frontend/dist", &req.path)
    })
    .on_websocket(move |mut ws, _req| {
        {
            let mut s = state.lock().unwrap();
            let mut buf = Vec::new();
            Message::Value(s.value).encode(&mut buf);
            let _ = ws.send_binary(&buf);
            if let Ok(writer) = ws.try_clone() {
                s.clients.push(writer);
            }
        }

        while let Ok(msg) = ws.recv() {
            if let WsMessage::Binary(data) = msg {
                if let Ok((Message::Value(v), _)) = Message::decode(&data) {
                    let mut s = state.lock().unwrap();
                    s.value = v;
                    let mut buf = Vec::new();
                    Message::Value(v).encode(&mut buf);
                    s.clients
                        .retain_mut(|c: &mut WebSocket| c.send_binary(&buf).is_ok());
                }
            }
        }
    })
    .start()
    .unwrap();

    println!("listening on http://{}", handle.addr());
    loop {
        std::thread::park();
    }
}
