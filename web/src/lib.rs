//! Minimal HTTP/WebSocket server with static file serving.
//!
//! Thread-per-connection, blocking I/O, zero dependencies beyond `std`.
//!
//! # Examples
//!
//! HTTP server:
//!
//! ```no_run
//! use std::net::SocketAddr;
//! use web::*;
//!
//! let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
//! let handle = Server::new(addr, |_req| {
//!     Response::new(200)
//!         .header("Content-Type", "text/plain")
//!         .body_str("Hello, World!")
//! }).start().unwrap();
//! ```
//!
//! With WebSocket:
//!
//! ```no_run
//! use std::net::SocketAddr;
//! use web::*;
//! use web::websocket::Message;
//!
//! let addr: SocketAddr = "127.0.0.1:8080".parse().unwrap();
//! let handle = Server::new(addr, |req| {
//!     serve::static_file("dist", &req.path)
//! }).on_websocket(|mut ws, _req| {
//!     ws.send_text("hello").unwrap();
//!     while let Ok(msg) = ws.recv() {
//!         match msg {
//!             Message::Text(t) => ws.send_text(&t).unwrap(),
//!             Message::Binary(b) => ws.send_binary(&b).unwrap(),
//!         }
//!     }
//! }).start().unwrap();
//! ```

pub mod http;
pub mod serve;
pub mod websocket;

mod sha1;
mod base64;
mod server;
pub use server::*;

pub use http::{Method, Request, Response};
pub use websocket::WebSocket;
