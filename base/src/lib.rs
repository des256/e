//! Foundation crate: async executor, channels, logging, and serial port.
//!
//! # Async Runtime
//!
//! A single-threaded async executor ([`Executor`]) and multi-producer
//! multi-consumer channel ([`channel`], [`Sender`], [`Receiver`]).
//!
//! # Logging
//!
//! The [`log`] module provides logging macros.

// -- async runtime --

/// Logging macros and utilities.
pub mod log;

mod executor;
pub use executor::*;

mod channel;
pub use channel::*;

mod epoch;
pub use epoch::*;

// -- serial port --

#[cfg(not(target_arch = "wasm32"))]
mod serial;
#[cfg(not(target_arch = "wasm32"))]
pub use serial::*;
