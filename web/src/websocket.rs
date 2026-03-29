//! WebSocket framing, upgrade handshake, and message I/O.
//!
//! A [`WebSocket`] wraps a [`TcpStream`] after the HTTP upgrade handshake.
//! Frames are read and written using the blocking stream; ping/pong is
//! handled transparently by [`recv`](WebSocket::recv).
//!
//! # Examples
//!
//! ```no_run
//! use std::net::SocketAddr;
//! use web::*;
//! use web::websocket::Message;
//!
//! let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
//! let handle = Server::new(addr, |_| Response::new(404))
//!     .on_websocket(|mut ws, _req| {
//!         ws.send_text("hello").unwrap();
//!         while let Ok(msg) = ws.recv() {
//!             match msg {
//!                 Message::Text(t) => ws.send_text(&t).unwrap(),
//!                 Message::Binary(b) => ws.send_binary(&b).unwrap(),
//!             }
//!         }
//!     })
//!     .start()
//!     .unwrap();
//! ```

use crate::{
    base64,
    sha1::sha1,
    Request, Response,
};
use std::{
    io::{self, Read, Write},
    net::TcpStream,
};

/// WebSocket frame opcodes.
pub mod opcode {
    /// Continuation frame.
    pub const CONTINUATION: u8 = 0x00;
    /// UTF-8 text frame.
    pub const TEXT: u8 = 0x01;
    /// Binary frame.
    pub const BINARY: u8 = 0x02;
    /// Connection close.
    pub const CLOSE: u8 = 0x08;
    /// Ping.
    pub const PING: u8 = 0x09;
    /// Pong (reply to ping).
    pub const PONG: u8 = 0x0A;
}

/// A raw WebSocket frame.
#[derive(Debug)]
pub struct Frame {
    /// Frame opcode (see [`opcode`]).
    pub opcode: u8,
    /// Frame payload (unmasked).
    pub payload: Vec<u8>,
}

/// A decoded WebSocket message.
#[derive(Debug, PartialEq, Eq)]
pub enum Message {
    /// UTF-8 text message.
    Text(String),
    /// Binary message.
    Binary(Vec<u8>),
}

/// A WebSocket connection over a TCP stream.
///
/// Created by [`WebSocket::upgrade`] from an HTTP connection, or directly
/// from a [`TcpStream`] that has already completed the handshake.
///
/// # Examples
///
/// ```no_run
/// use web::websocket::{WebSocket, Message};
/// use web::Request;
/// use std::net::TcpStream;
///
/// fn handle(stream: TcpStream, req: &Request) {
///     let mut ws = WebSocket::upgrade(stream, req).unwrap();
///     ws.send_text("connected").unwrap();
///     loop {
///         match ws.recv() {
///             Ok(Message::Text(t)) => ws.send_text(&t).unwrap(),
///             Ok(Message::Binary(b)) => ws.send_binary(&b).unwrap(),
///             Err(_) => break,
///         }
///     }
/// }
/// ```
pub struct WebSocket {
    stream: TcpStream,
}

impl WebSocket {
    /// Perform the WebSocket upgrade handshake and return a [`WebSocket`].
    ///
    /// Reads the `Sec-WebSocket-Key` from `request`, computes the accept
    /// key, writes the `101 Switching Protocols` response, and returns the
    /// connection ready for frame I/O.
    pub fn upgrade(mut stream: TcpStream, request: &Request) -> io::Result<Self> {
        let key = request
            .header("sec-websocket-key")
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "missing Sec-WebSocket-Key"))?;

        let accept = accept_key(key);

        let response = Response::new(101)
            .header("Upgrade", "websocket")
            .header("Connection", "Upgrade")
            .header("Sec-WebSocket-Accept", accept);

        stream.write_all(&response.to_bytes())?;

        Ok(WebSocket { stream })
    }

    /// Wrap an already-upgraded stream (handshake completed externally).
    pub fn from_stream(stream: TcpStream) -> Self {
        WebSocket { stream }
    }

    /// Clone the underlying stream for split reader/writer usage.
    ///
    /// The returned `WebSocket` shares the same connection. Use one side
    /// for reading and the other for writing on separate threads.
    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(WebSocket {
            stream: self.stream.try_clone()?,
        })
    }

    /// Read a raw frame from the connection.
    pub fn recv_frame(&mut self) -> io::Result<Frame> {
        let mut header = [0u8; 2];
        self.stream.read_exact(&mut header)?;

        let opcode = header[0] & 0x0F;
        let masked = header[1] & 0x80 != 0;
        let mut len = (header[1] & 0x7F) as u64;

        if len == 126 {
            let mut buf = [0u8; 2];
            self.stream.read_exact(&mut buf)?;
            len = u16::from_be_bytes(buf) as u64;
        } else if len == 127 {
            let mut buf = [0u8; 8];
            self.stream.read_exact(&mut buf)?;
            len = u64::from_be_bytes(buf);
        }

        let mask = if masked {
            let mut buf = [0u8; 4];
            self.stream.read_exact(&mut buf)?;
            Some(buf)
        } else {
            None
        };

        let mut payload = vec![0u8; len as usize];
        self.stream.read_exact(&mut payload)?;

        if let Some(mask) = mask {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }

        Ok(Frame { opcode, payload })
    }

    /// Write a raw frame to the connection (server-to-client, unmasked).
    pub fn send_frame(&mut self, opcode: u8, data: &[u8]) -> io::Result<()> {
        let mut header = vec![0x80 | opcode]; // FIN + opcode

        if data.len() < 126 {
            header.push(data.len() as u8);
        } else if data.len() < 65536 {
            header.push(126);
            header.extend_from_slice(&(data.len() as u16).to_be_bytes());
        } else {
            header.push(127);
            header.extend_from_slice(&(data.len() as u64).to_be_bytes());
        }

        self.stream.write_all(&header)?;
        self.stream.write_all(data)
    }

    /// Send a UTF-8 text message.
    pub fn send_text(&mut self, text: &str) -> io::Result<()> {
        self.send_frame(opcode::TEXT, text.as_bytes())
    }

    /// Send a binary message.
    pub fn send_binary(&mut self, data: &[u8]) -> io::Result<()> {
        self.send_frame(opcode::BINARY, data)
    }

    /// Send a close frame with an optional status code.
    pub fn send_close(&mut self, code: Option<u16>) -> io::Result<()> {
        match code {
            Some(c) => self.send_frame(opcode::CLOSE, &c.to_be_bytes()),
            None => self.send_frame(opcode::CLOSE, &[]),
        }
    }

    /// Receive the next data message, handling ping/pong transparently.
    ///
    /// Returns [`Message::Text`] or [`Message::Binary`]. Ping frames are
    /// answered with pong automatically. A close frame returns an error
    /// with [`ErrorKind::ConnectionReset`](io::ErrorKind::ConnectionReset).
    pub fn recv(&mut self) -> io::Result<Message> {
        loop {
            let frame = self.recv_frame()?;
            match frame.opcode {
                opcode::TEXT => {
                    let text = String::from_utf8(frame.payload).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in text frame")
                    })?;
                    return Ok(Message::Text(text));
                }
                opcode::BINARY => return Ok(Message::Binary(frame.payload)),
                opcode::CLOSE => {
                    let _ = self.send_frame(opcode::CLOSE, &frame.payload);
                    return Err(io::Error::new(io::ErrorKind::ConnectionReset, "close"));
                }
                opcode::PING => {
                    self.send_frame(opcode::PONG, &frame.payload)?;
                }
                _ => {} // ignore pong, continuation, unknown
            }
        }
    }
}

// --- WebSocket accept key computation ---

const WS_MAGIC: &[u8] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

fn accept_key(client_key: &str) -> String {
    let mut input = Vec::with_capacity(client_key.len() + WS_MAGIC.len());
    input.extend_from_slice(client_key.as_bytes());
    input.extend_from_slice(WS_MAGIC);
    base64::encode(&sha1(&input))
}

// --- Client-side frame helpers for testing ---

/// Build a masked client-to-server frame (for tests across the crate).
#[cfg(test)]
pub(crate) fn build_client_frame(opcode: u8, data: &[u8], mask_key: [u8; 4]) -> Vec<u8> {
    let mut frame = vec![0x80 | opcode];

    if data.len() < 126 {
        frame.push(0x80 | data.len() as u8); // masked bit set
    } else if data.len() < 65536 {
        frame.push(0x80 | 126);
        frame.extend_from_slice(&(data.len() as u16).to_be_bytes());
    } else {
        frame.push(0x80 | 127);
        frame.extend_from_slice(&(data.len() as u64).to_be_bytes());
    }

    frame.extend_from_slice(&mask_key);

    let mut masked = data.to_vec();
    for (i, byte) in masked.iter_mut().enumerate() {
        *byte ^= mask_key[i % 4];
    }
    frame.extend_from_slice(&masked);

    frame
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::net::{SocketAddr, TcpListener};

    #[test]
    fn accept_key_rfc_vector() {
        // RFC 6455 Section 4.2.2 example
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn frame_round_trip_small() {
        let (mut server, mut client) = connected_pair();

        // Client sends a masked text frame
        let data = b"hello";
        let frame_bytes = build_client_frame(opcode::TEXT, data, [0xAA, 0xBB, 0xCC, 0xDD]);
        client.write_all(&frame_bytes).unwrap();

        // Server reads and unmasks it
        let frame = server.recv_frame().unwrap();
        assert_eq!(frame.opcode, opcode::TEXT);
        assert_eq!(frame.payload, b"hello");
    }

    #[test]
    fn frame_round_trip_medium() {
        let (mut server, mut client) = connected_pair();

        // 200-byte payload (uses 16-bit extended length)
        let data = vec![0x42u8; 200];
        let frame_bytes = build_client_frame(opcode::BINARY, &data, [1, 2, 3, 4]);
        client.write_all(&frame_bytes).unwrap();

        let frame = server.recv_frame().unwrap();
        assert_eq!(frame.opcode, opcode::BINARY);
        assert_eq!(frame.payload, data);
    }

    #[test]
    fn send_frame_small() {
        let (mut server, mut client) = connected_pair();

        server.send_text("hi").unwrap();

        // Read raw bytes on client side (unmasked server frame)
        let mut buf = [0u8; 64];
        let n = client.read(&mut buf).unwrap();
        let bytes = &buf[..n];

        assert_eq!(bytes[0], 0x80 | opcode::TEXT); // FIN + text
        assert_eq!(bytes[1], 2); // length, no mask bit
        assert_eq!(&bytes[2..4], b"hi");
    }

    #[test]
    fn recv_handles_ping() {
        let (mut server, mut client) = connected_pair();

        // Client sends ping, then a text frame
        let ping = build_client_frame(opcode::PING, b"ping", [1, 2, 3, 4]);
        let text = build_client_frame(opcode::TEXT, b"after-ping", [5, 6, 7, 8]);
        client.write_all(&ping).unwrap();
        client.write_all(&text).unwrap();

        // recv() should skip the ping (sending pong) and return the text
        let msg = server.recv().unwrap();
        assert_eq!(msg, Message::Text("after-ping".into()));

        // Client should have received a pong
        let mut buf = [0u8; 64];
        let n = client.read(&mut buf).unwrap();
        assert_eq!(buf[0], 0x80 | opcode::PONG);
        assert_eq!(&buf[2..n], b"ping");
    }

    #[test]
    fn recv_close_returns_error() {
        let (mut server, mut client) = connected_pair();

        let close = build_client_frame(opcode::CLOSE, &1000u16.to_be_bytes(), [1, 2, 3, 4]);
        client.write_all(&close).unwrap();

        let err = server.recv().unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::ConnectionReset);
    }

    #[test]
    fn send_binary() {
        let (mut server, mut client) = connected_pair();

        let data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        server.send_binary(&data).unwrap();

        let mut buf = [0u8; 64];
        let n = client.read(&mut buf).unwrap();
        assert_eq!(buf[0], 0x80 | opcode::BINARY);
        assert_eq!(buf[1], 4);
        assert_eq!(&buf[2..n], &data);
    }

    #[test]
    fn try_clone_split() {
        let (server, mut client) = connected_pair();

        let mut writer = server.try_clone().unwrap();

        // Write from the clone, read from the original's underlying client
        writer.send_text("from-clone").unwrap();

        let mut buf = [0u8; 64];
        let n = client.read(&mut buf).unwrap();
        let payload = &buf[2..n]; // skip 2-byte header
        assert_eq!(payload, b"from-clone");
    }

    // --- Helpers ---

    /// Create a connected (server-side WebSocket, raw client TcpStream) pair.
    fn connected_pair() -> (WebSocket, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0".parse::<SocketAddr>().unwrap()).unwrap();
        let addr = listener.local_addr().unwrap();

        let client = TcpStream::connect(addr).unwrap();
        let (server_stream, _) = listener.accept().unwrap();

        (WebSocket::from_stream(server_stream), client)
    }
}
