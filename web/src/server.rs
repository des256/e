use crate::{
    http::ParseResult,
    websocket::WebSocket,
    Method, Request, Response,
};
use std::{
    io::{self, Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    sync::Arc,
};

/// Handler function type: receives a request, returns a response.
pub type Handler = Arc<dyn Fn(&Request) -> Response + Send + Sync>;

/// WebSocket handler function type: receives an upgraded connection and the
/// original request. The handler blocks the connection thread for its
/// lifetime.
pub type WsHandler = Arc<dyn Fn(WebSocket, Request) + Send + Sync>;

/// A thread-per-connection HTTP/1.1 server with optional WebSocket support.
///
/// Create with [`Server::new`], optionally add a WebSocket handler with
/// [`on_websocket`](Server::on_websocket), then call [`start`](Server::start).
///
/// # Examples
///
/// HTTP only:
///
/// ```no_run
/// use std::net::SocketAddr;
/// use web::*;
///
/// let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
/// let handle = Server::new(addr, |_req| {
///     Response::new(200)
///         .header("Content-Type", "text/plain")
///         .body_str("Hello, World!")
/// }).start().unwrap();
/// ```
///
/// With WebSocket:
///
/// ```no_run
/// use std::net::SocketAddr;
/// use web::*;
/// use web::websocket::Message;
///
/// let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
/// let handle = Server::new(addr, |req| {
///     serve::static_file("dist", &req.path)
/// }).on_websocket(|mut ws, _req| {
///     ws.send_text("hello").unwrap();
///     while let Ok(msg) = ws.recv() {
///         match msg {
///             Message::Text(t) => ws.send_text(&t).unwrap(),
///             Message::Binary(b) => ws.send_binary(&b).unwrap(),
///         }
///     }
/// }).start().unwrap();
/// ```
pub struct Server {
    addr: SocketAddr,
    handler: Handler,
    ws_handler: Option<WsHandler>,
}

impl Server {
    /// Create a new server bound to `addr`, dispatching HTTP requests to
    /// `handler`.
    pub fn new(addr: SocketAddr, handler: impl Fn(&Request) -> Response + Send + Sync + 'static) -> Self {
        Server {
            addr,
            handler: Arc::new(handler),
            ws_handler: None,
        }
    }

    /// Register a WebSocket handler.
    ///
    /// When a client sends an HTTP request with `Upgrade: websocket`, the
    /// server performs the upgrade handshake and passes the resulting
    /// [`WebSocket`] and original [`Request`] to this handler. The handler
    /// runs on the connection thread and should block for the connection
    /// lifetime.
    pub fn on_websocket(mut self, handler: impl Fn(WebSocket, Request) + Send + Sync + 'static) -> Self {
        self.ws_handler = Some(Arc::new(handler));
        self
    }

    /// Start listening and serving.
    ///
    /// Spawns an accept thread and returns immediately. The returned
    /// [`ServerHandle`] provides the bound address (useful when binding to
    /// port 0).
    pub fn start(self) -> io::Result<ServerHandle> {
        let listener = TcpListener::bind(self.addr)?;
        let local_addr = listener.local_addr()?;
        let handler = self.handler;
        let ws_handler = self.ws_handler;

        let join = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let stream = match stream {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let handler = handler.clone();
                let ws_handler = ws_handler.clone();
                std::thread::spawn(move || {
                    if let Err(e) = handle_connection(stream, &handler, &ws_handler) {
                        eprintln!("web: connection error: {e}");
                    }
                });
            }
        });

        Ok(ServerHandle {
            addr: local_addr,
            _join: join,
        })
    }
}

/// Handle returned by [`Server::start`], providing the bound address.
pub struct ServerHandle {
    addr: SocketAddr,
    _join: std::thread::JoinHandle<()>,
}

impl ServerHandle {
    /// The address the server is listening on.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
}

fn is_websocket_upgrade(request: &Request) -> bool {
    request
        .header("upgrade")
        .map(|v| v.eq_ignore_ascii_case("websocket"))
        .unwrap_or(false)
}

fn handle_connection(
    mut stream: TcpStream,
    handler: &Handler,
    ws_handler: &Option<WsHandler>,
) -> io::Result<()> {
    let mut buf = vec![0u8; 8192];
    let mut filled = 0;

    loop {
        let n = stream.read(&mut buf[filled..])?;
        if n == 0 {
            return Ok(());
        }
        filled += n;

        match Request::parse(&buf[..filled]) {
            ParseResult::Ok(request, consumed) => {
                buf.copy_within(consumed..filled, 0);
                filled -= consumed;

                // WebSocket upgrade
                if is_websocket_upgrade(&request) {
                    if let Some(ws_handler) = ws_handler {
                        let ws = WebSocket::upgrade(stream, &request)?;
                        ws_handler(ws, request);
                        return Ok(());
                    }
                    // No ws_handler registered — respond 404
                    let resp = Response::new(404).body_str("Not Found");
                    stream.write_all(&resp.to_bytes())?;
                    return Ok(());
                }

                let keep_alive = request
                    .header("connection")
                    .map(|v| v.eq_ignore_ascii_case("keep-alive"))
                    .unwrap_or(true);

                let response = if request.method == Method::Head {
                    let mut resp = handler(&request);
                    resp.body = Vec::new();
                    resp
                } else {
                    handler(&request)
                };

                stream.write_all(&response.to_bytes())?;

                if !keep_alive {
                    return Ok(());
                }
            }
            ParseResult::Incomplete => {
                if filled == buf.len() {
                    buf.resize(filled + 8192, 0);
                }
            }
            ParseResult::BadRequest => {
                let resp = Response::new(400).body_str("Bad Request");
                stream.write_all(&resp.to_bytes())?;
                return Ok(());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::websocket::{self, Message};

    fn start_server(handler: impl Fn(&Request) -> Response + Send + Sync + 'static) -> ServerHandle {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        Server::new(addr, handler).start().unwrap()
    }

    fn get(addr: SocketAddr, path: &str, close: bool) -> String {
        let mut stream = TcpStream::connect(addr).unwrap();
        let conn = if close { "close" } else { "keep-alive" };
        let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\nConnection: {conn}\r\n\r\n");
        stream.write_all(req.as_bytes()).unwrap();
        let mut resp = Vec::new();
        if close {
            stream.read_to_end(&mut resp).unwrap();
        } else {
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap();
            resp.extend_from_slice(&buf[..n]);
        }
        String::from_utf8_lossy(&resp).into_owned()
    }

    #[test]
    fn hello_world() {
        let handle = start_server(|_| {
            Response::new(200)
                .header("Content-Type", "text/plain")
                .body_str("Hello, World!")
        });
        let resp = get(handle.addr(), "/", true);
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Type: text/plain"));
        assert!(resp.contains("Content-Length: 13"));
        assert!(resp.ends_with("Hello, World!"));
    }

    #[test]
    fn keep_alive() {
        let handle = start_server(|req| {
            Response::new(200)
                .header("Content-Type", "text/plain")
                .body_str(&format!("path={}", req.path))
        });

        let mut stream = TcpStream::connect(handle.addr()).unwrap();
        for path in ["/first", "/second"] {
            let req = format!("GET {path} HTTP/1.1\r\nHost: localhost\r\n\r\n");
            stream.write_all(req.as_bytes()).unwrap();
            let mut buf = [0u8; 4096];
            let n = stream.read(&mut buf).unwrap();
            let text = String::from_utf8_lossy(&buf[..n]);
            assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
            assert!(text.ends_with(&format!("path={path}")));
        }
    }

    #[test]
    fn static_file_serving() {
        let dir = std::env::temp_dir().join("web_test_server_static");
        let _ = std::fs::create_dir_all(&dir);
        std::fs::write(dir.join("page.html"), "<h1>Test</h1>").unwrap();

        let dir_str = dir.to_str().unwrap().to_string();
        let handle = start_server(move |req| crate::serve::static_file(&dir_str, &req.path));

        let resp = get(handle.addr(), "/page.html", true);
        assert!(resp.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(resp.contains("Content-Type: text/html; charset=utf-8"));
        assert!(resp.ends_with("<h1>Test</h1>"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn not_found() {
        let handle = start_server(|req| {
            crate::serve::static_file("/tmp/web_test_nonexistent", &req.path)
        });
        let resp = get(handle.addr(), "/nope.html", true);
        assert!(resp.starts_with("HTTP/1.1 404 Not Found\r\n"));
    }

    #[test]
    fn path_traversal_rejected() {
        let dir = std::env::temp_dir().join("web_test_server_traversal");
        let _ = std::fs::create_dir_all(&dir);

        let dir_str = dir.to_str().unwrap().to_string();
        let handle = start_server(move |req| crate::serve::static_file(&dir_str, &req.path));

        let resp = get(handle.addr(), "/../../../etc/passwd", true);
        assert!(resp.starts_with("HTTP/1.1 400 Bad Request\r\n"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    // --- WebSocket integration tests ---

    /// Send an upgrade request and read back the 101 response, returning
    /// any leftover bytes after the HTTP response boundary.
    fn ws_upgrade(stream: &mut TcpStream, path: &str, key: &str) -> Vec<u8> {
        let req = format!(
            "GET {path} HTTP/1.1\r\n\
             Host: localhost\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {key}\r\n\
             Sec-WebSocket-Version: 13\r\n\
             \r\n"
        );
        stream.write_all(req.as_bytes()).unwrap();

        // Read until we have the full HTTP response (ends with \r\n\r\n)
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = stream.read(&mut tmp).unwrap();
            buf.extend_from_slice(&tmp[..n]);
            if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let resp = String::from_utf8_lossy(&buf[..pos]).to_string();
                assert!(resp.starts_with("HTTP/1.1 101 Switching Protocols"));
                // Return any bytes after the HTTP response
                return buf[pos + 4..].to_vec();
            }
        }
    }

    /// Read a complete server-to-client frame from `stream`, prepending any
    /// `leftover` bytes from a previous read.
    fn read_server_frame(stream: &mut TcpStream, leftover: &[u8]) -> (u8, Vec<u8>) {
        let mut buf = leftover.to_vec();
        let mut tmp = [0u8; 4096];

        // Ensure we have at least the 2-byte header
        while buf.len() < 2 {
            let n = stream.read(&mut tmp).unwrap();
            buf.extend_from_slice(&tmp[..n]);
        }

        let opcode = buf[0] & 0x0F;
        let len = buf[1] as usize; // works for payloads < 126
        let total = 2 + len;

        while buf.len() < total {
            let n = stream.read(&mut tmp).unwrap();
            buf.extend_from_slice(&tmp[..n]);
        }

        let payload = buf[2..total].to_vec();
        (opcode, payload)
    }

    #[test]
    fn websocket_echo() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle = Server::new(addr, |_| Response::new(404))
            .on_websocket(|mut ws, _req| {
                while let Ok(msg) = ws.recv() {
                    match msg {
                        Message::Text(t) => { let _ = ws.send_text(&t); }
                        Message::Binary(b) => { let _ = ws.send_binary(&b); }
                    }
                }
            })
            .start()
            .unwrap();

        let mut stream = TcpStream::connect(handle.addr()).unwrap();
        let leftover = ws_upgrade(&mut stream, "/ws", "dGhlIHNhbXBsZSBub25jZQ==");
        assert!(leftover.is_empty());

        // Send a masked text frame: "hello"
        let frame = crate::websocket::build_client_frame(
            websocket::opcode::TEXT,
            b"hello",
            [0x12, 0x34, 0x56, 0x78],
        );
        stream.write_all(&frame).unwrap();

        let (opcode, payload) = read_server_frame(&mut stream, &[]);
        assert_eq!(opcode, websocket::opcode::TEXT);
        assert_eq!(payload, b"hello");
    }

    #[test]
    fn websocket_receives_request_path() {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let handle = Server::new(addr, |_| Response::new(404))
            .on_websocket(|mut ws, req| {
                let _ = ws.send_text(&req.path);
            })
            .start()
            .unwrap();

        let mut stream = TcpStream::connect(handle.addr()).unwrap();
        let leftover = ws_upgrade(&mut stream, "/my/path", "dGhlIHNhbXBsZSBub25jZQ==");

        // The handler sends the path immediately after upgrade — may be in leftover
        let (opcode, payload) = read_server_frame(&mut stream, &leftover);
        assert_eq!(opcode, websocket::opcode::TEXT);
        assert_eq!(payload, b"/my/path");
    }
}
