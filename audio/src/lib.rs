//! Audio capture and playback via PulseAudio.
//!
//! Each module provides a `create` function that spawns a background thread
//! and returns a [`Handle`](audioin::Handle) / [`Listener`](audioin::Listener)
//! pair for non-blocking control and data flow.

/// Audio capture from a PulseAudio source.
pub mod audioin;
/// Audio playback to a PulseAudio sink.
pub mod audioout;

mod pulse;
