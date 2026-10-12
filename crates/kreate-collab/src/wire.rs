//! What travels between a participant and the relay.
//!
//! One frame per message, so a mesh never has to be escaped into text:
//!
//! ```text
//! u32 header length · u32 body length · header JSON · body bytes
//! ```
//!
//! both lengths big endian. The header is a [`Msg`]; the body is a patch
//! (see `kreate_io::sync`) for the messages that carry one, and empty
//! otherwise.

use std::io::{self, Read, Write};

use serde::{Deserialize, Serialize};

/// The protocol's version. A participant and a relay that differ refuse each
/// other at the door rather than misread each other later.
pub const PROTOCOL: u32 = 1;

/// The port a relay listens on unless told otherwise.
pub const DEFAULT_PORT: u16 = 7878;

/// The largest frame either side accepts: a scene's worth of meshes and
/// textures, and not a length an attacker picked to exhaust memory.
pub const MAX_FRAME: usize = 1 << 30;

/// Where a participant is looking and what they hold.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Presence {
    /// The viewpoint, in the world.
    pub eye: [f64; 3],
    /// Unit vectors along the view and up the screen.
    pub forward: [f64; 3],
    pub up: [f64; 3],
    /// Vertical field of view, radians, and width over height.
    pub fov: f64,
    pub aspect: f64,
    /// The objects they have selected, by identity.
    pub selection: Vec<u64>,
    /// The object they are editing the inside of, if any.
    pub editing: Option<u64>,
}

/// Someone in the session.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Peer {
    pub site: u16,
    pub name: String,
    /// Their colour, sRGB.
    pub color: [u8; 3],
}

/// One message's header.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Msg {
    // --- participant to relay ----------------------------------------------
    /// The first thing a participant says.
    Hello {
        protocol: u32,
        name: String,
        room: String,
        key: String,
    },
    /// An edit; the body is the patch.
    Patch,
    /// Where they are looking now.
    Look(Presence),
    /// Asks to edit the inside of an object alone.
    Lock {
        uid: u64,
    },
    /// Lets go of one.
    Unlock {
        uid: u64,
    },

    // --- relay to participant ----------------------------------------------
    /// The door opened. `seeded` says whether a [`Msg::Snapshot`] follows; if
    /// not, this participant is the first, and its scene becomes the
    /// session's — it sends the whole of it as its first patch.
    Welcome {
        site: u16,
        color: [u8; 3],
        peers: Vec<Peer>,
        seeded: bool,
    },
    /// The session's scene as it stands; the body is a patch of all of it.
    Snapshot {
        seq: u64,
    },
    /// Someone else's edit, in the order the relay took it; the body is
    /// the patch.
    Applied {
        from: u16,
        seq: u64,
    },
    /// The relay took this participant's own oldest unanswered patch.
    Ack {
        seq: u64,
    },
    Joined(Peer),
    Left {
        site: u16,
    },
    /// Where someone else is looking.
    Seen {
        site: u16,
        presence: Presence,
    },
    /// Who is editing the inside of what: (object identity, participant).
    Locks {
        held: Vec<(u64, u16)>,
    },
    /// The door stayed shut, or the relay is closing it.
    Refused {
        reason: String,
    },
}

/// A message and its body.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub msg: Msg,
    pub body: Vec<u8>,
}

impl Frame {
    pub fn new(msg: Msg) -> Self {
        Self {
            msg,
            body: Vec::new(),
        }
    }

    pub fn with_body(msg: Msg, body: Vec<u8>) -> Self {
        Self { msg, body }
    }

    pub fn write_to(&self, out: &mut impl Write) -> io::Result<()> {
        let header = serde_json::to_vec(&self.msg).expect("a message always serialises");
        let mut head = [0u8; 8];
        head[..4].copy_from_slice(&(header.len() as u32).to_be_bytes());
        head[4..].copy_from_slice(&(self.body.len() as u32).to_be_bytes());
        out.write_all(&head)?;
        out.write_all(&header)?;
        out.write_all(&self.body)?;
        out.flush()
    }

    pub fn read_from(input: &mut impl Read) -> io::Result<Self> {
        let mut head = [0u8; 8];
        input.read_exact(&mut head)?;
        let header_len = u32::from_be_bytes(head[..4].try_into().unwrap()) as usize;
        let body_len = u32::from_be_bytes(head[4..].try_into().unwrap()) as usize;
        if header_len > 1 << 20 || body_len > MAX_FRAME {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "a frame larger than the protocol allows",
            ));
        }
        let mut header = vec![0u8; header_len];
        input.read_exact(&mut header)?;
        let mut body = vec![0u8; body_len];
        input.read_exact(&mut body)?;
        let msg = serde_json::from_slice(&header)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Self { msg, body })
    }
}

/// The colour the relay gives participant `site`: eight that tell apart on
/// a grey viewport, then round again.
pub fn site_color(site: u16) -> [u8; 3] {
    const PALETTE: [[u8; 3]; 8] = [
        [255, 120, 60],
        [70, 170, 255],
        [120, 220, 90],
        [235, 90, 200],
        [250, 210, 60],
        [80, 220, 210],
        [170, 120, 255],
        [255, 90, 100],
    ];
    PALETTE[usize::from(site.saturating_sub(1)) % PALETTE.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_reads_back_as_written() {
        let frame = Frame::with_body(Msg::Applied { from: 3, seq: 9 }, vec![1, 2, 3]);
        let mut bytes = Vec::new();
        frame.write_to(&mut bytes).unwrap();
        let back = Frame::read_from(&mut bytes.as_slice()).unwrap();
        assert_eq!(back, frame);
    }

    #[test]
    fn an_absurd_length_is_refused_before_it_is_allocated() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&2u32.to_be_bytes());
        bytes.extend_from_slice(&u32::MAX.to_be_bytes());
        assert!(Frame::read_from(&mut bytes.as_slice()).is_err());
    }
}
