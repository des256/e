//! Shared types for the exploreui experiment.
//!
//! Defines the [`Message`] enum exchanged between frontend and backend
//! over WebSocket as binary via [`Codec`].

pub use base::Vec3;
pub use codec::{Codec, CodecError};

// -- messages --

/// Messages exchanged between frontend and backend over WebSocket.
#[derive(Debug, PartialEq, Codec)]
pub enum Message {
    /// Current or desired value of the shared [`Vec3<f32>`].
    Value(Vec3<f32>),
}
