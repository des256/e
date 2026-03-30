//! Minimal reactive WASM frontend framework.
//!
//! # Reactive Core
//!
//! [`Signal`] handles are `Copy`-able indices into an arena owned by the
//! [`Runtime`](runtime). Effects ([`Context::effect`]) auto-track which
//! signals they read and re-run when those signals change.
//!
//! # View Description
//!
//! Views are functions that return [`Node`] values built with the
//! [`ElementBuilder`]. Layout helpers ([`ElementBuilder::row`],
//! [`ElementBuilder::col`], [`ElementBuilder::gap`], etc.) emit inline
//! styles; CSS does the actual layout.
//!
//! # Styling
//!
//! [`CssDef`] builds CSS class rules with pseudo-class support
//! ([`CssDef::hover`], [`CssDef::active`], [`CssDef::focus`]) and
//! transitions. Register classes via [`Context::css_class`].
//!
//! # DOM Patcher
//!
//! [`mount`] turns a [`Node`] tree into real DOM elements. Reactive
//! nodes re-evaluate only their subtree when signals change.

// -- reactive core --

pub mod runtime;

// -- color --

pub mod color;

// -- JS interop --

pub mod ffi;

// -- view description --

pub mod node;
pub mod builder;

// -- styling --

pub mod style;

// -- DOM patcher --

pub mod patcher;

pub use runtime::{Context, Signal, with_context};
pub use color::Color;
pub use ffi::Element;
pub use node::Node;
pub use builder::{ElementBuilder, div, span, element, text, reactive, empty};
pub use style::{CssDef, css, EASE, EASE_IN, EASE_OUT, LINEAR};
pub use patcher::mount;
