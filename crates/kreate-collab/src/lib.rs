//! Editing one scene together, live.
//!
//! - [`wire`]: the frames a participant and the relay exchange.
//! - [`relay`]: the server that orders edits and keeps the session's scene.
//! - [`replica`]: a participant's bookkeeping — what to send, what to take
//!   in, and how undo keeps to one's own edits.
//! - [`client`]: the network side of a participant.
//! - [`session`]: the three put together, for the application to drive once
//!   a frame.

pub mod client;
pub mod relay;
pub mod replica;
pub mod session;
pub mod wire;

pub use relay::{Relay, RelayConfig};
pub use session::{Session, SessionEvent};
pub use wire::{Peer, Presence, DEFAULT_PORT};
