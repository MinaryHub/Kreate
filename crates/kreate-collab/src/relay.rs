//! The relay: the one place a session's edits are put in order.
//!
//! It keeps the session's scene, applying every patch as it passes, so a
//! participant who joins late starts from the scene as it stands rather than
//! from a replay of everything since the first. It does no editing of its
//! own; it orders, stores and forwards.
//!
//! Plain TCP, one thread per participant. Meant for a studio's own network:
//! a room key keeps out the wrong session, not an attacker on the wire.

use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufReader, BufWriter};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use kreate_scene::Document;

use crate::wire::{site_color, Frame, Msg, Peer, Presence, PROTOCOL};

/// A relay's settings.
#[derive(Clone, Debug)]
pub struct RelayConfig {
    /// What a participant must present to come in. Empty lets anyone in.
    pub key: String,
    /// The most people in one room.
    pub max_peers: usize,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            key: String::new(),
            max_peers: 8,
        }
    }
}

/// A running relay. Dropping it stops taking new participants and closes the
/// ones it has.
pub struct Relay {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    rooms: Arc<Mutex<HashMap<String, Room>>>,
}

#[derive(Default)]
struct Room {
    doc: Document,
    seeded: bool,
    seq: u64,
    next_site: u16,
    peers: BTreeMap<u16, Member>,
    looks: HashMap<u16, Presence>,
    locks: BTreeMap<u64, u16>,
}

struct Member {
    peer: Peer,
    tx: Sender<Frame>,
    /// To close it from here when the relay stops.
    stream: TcpStream,
}

impl Room {
    fn broadcast(&self, frame: &Frame, except: Option<u16>) {
        for (site, member) in &self.peers {
            if Some(*site) != except {
                let _ = member.tx.send(frame.clone());
            }
        }
    }

    fn locks_frame(&self) -> Frame {
        Frame::new(Msg::Locks {
            held: self.locks.iter().map(|(u, s)| (*u, *s)).collect(),
        })
    }
}

impl Relay {
    /// Starts listening on `addr` (port 0 picks one).
    pub fn start(addr: &str, config: RelayConfig) -> io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let rooms: Arc<Mutex<HashMap<String, Room>>> = Arc::default();
        {
            let stop = stop.clone();
            let rooms = rooms.clone();
            let config = Arc::new(config);
            thread::Builder::new()
                .name("kreate-relay".into())
                .spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                let rooms = rooms.clone();
                                let config = config.clone();
                                let _ = thread::Builder::new()
                                    .name("kreate-relay-peer".into())
                                    .spawn(move || {
                                        let _ = serve(stream, &rooms, &config);
                                    });
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(50));
                            }
                            Err(_) => thread::sleep(Duration::from_millis(50)),
                        }
                    }
                })?;
        }
        Ok(Self { addr, stop, rooms })
    }

    /// Where it is listening.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// How many people are in all its rooms.
    pub fn peer_count(&self) -> usize {
        self.rooms
            .lock()
            .map(|r| r.values().map(|room| room.peers.len()).sum())
            .unwrap_or(0)
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Ok(rooms) = self.rooms.lock() {
            for room in rooms.values() {
                for member in room.peers.values() {
                    let _ = member.stream.shutdown(std::net::Shutdown::Both);
                }
            }
        }
    }
}

/// One participant, from the door to leaving.
fn serve(
    stream: TcpStream,
    rooms: &Mutex<HashMap<String, Room>>,
    config: &RelayConfig,
) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;
    let mut reader = BufReader::new(stream.try_clone()?);
    let refuse = |reason: &str| -> io::Result<()> {
        let mut w = BufWriter::new(stream.try_clone()?);
        Frame::new(Msg::Refused {
            reason: reason.into(),
        })
        .write_to(&mut w)
    };

    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    let hello = Frame::read_from(&mut reader)?;
    stream.set_read_timeout(None)?;
    let Msg::Hello {
        protocol,
        name,
        room: room_name,
        key,
    } = hello.msg
    else {
        return refuse("expected a hello");
    };
    if protocol != PROTOCOL {
        return refuse("a different version of Kreate");
    }
    if key != config.key {
        return refuse("wrong session key");
    }

    // The writer: its own thread, so a slow participant never holds the room.
    let (tx, rx) = mpsc::channel::<Frame>();
    {
        let mut writer = BufWriter::new(stream.try_clone()?);
        thread::spawn(move || {
            for frame in rx {
                if frame.write_to(&mut writer).is_err() {
                    break;
                }
            }
        });
    }

    let site = {
        let mut rooms = rooms.lock().expect("relay lock");
        let room = rooms.entry(room_name.clone()).or_default();
        if room.peers.len() >= config.max_peers {
            drop(rooms);
            return refuse("the session is full");
        }
        room.next_site += 1;
        let site = room.next_site;
        let name = name.chars().take(40).collect::<String>();
        let peer = Peer {
            site,
            name,
            color: site_color(site),
        };
        let _ = tx.send(Frame::new(Msg::Welcome {
            site,
            color: peer.color,
            peers: room.peers.values().map(|m| m.peer.clone()).collect(),
            seeded: room.seeded,
        }));
        if room.seeded {
            match kreate_io::sync::encode_all(&room.doc) {
                Ok(bytes) => {
                    let _ = tx.send(Frame::with_body(Msg::Snapshot { seq: room.seq }, bytes));
                }
                Err(_) => {
                    drop(rooms);
                    return refuse("the session's scene could not be sent");
                }
            }
        }
        for (other, presence) in &room.looks {
            let _ = tx.send(Frame::new(Msg::Seen {
                site: *other,
                presence: presence.clone(),
            }));
        }
        let _ = tx.send(room.locks_frame());
        room.broadcast(&Frame::new(Msg::Joined(peer.clone())), None);
        room.peers.insert(
            site,
            Member {
                peer,
                tx: tx.clone(),
                stream: stream.try_clone()?,
            },
        );
        site
    };

    let result = (|| -> io::Result<()> {
        loop {
            let frame = Frame::read_from(&mut reader)?;
            let mut rooms = rooms.lock().expect("relay lock");
            let Some(room) = rooms.get_mut(&room_name) else {
                return Ok(());
            };
            match frame.msg {
                Msg::Patch => {
                    // Kept only if it reads: a patch the relay cannot apply
                    // would be one the others cannot either.
                    if kreate_io::sync::apply(&mut room.doc, &frame.body).is_err() {
                        continue;
                    }
                    room.seeded = true;
                    room.seq += 1;
                    let seq = room.seq;
                    let _ = tx.send(Frame::new(Msg::Ack { seq }));
                    room.broadcast(
                        &Frame::with_body(Msg::Applied { from: site, seq }, frame.body),
                        Some(site),
                    );
                }
                Msg::Look(presence) => {
                    room.looks.insert(site, presence.clone());
                    room.broadcast(&Frame::new(Msg::Seen { site, presence }), Some(site));
                }
                Msg::Lock { uid } => {
                    room.locks.entry(uid).or_insert(site);
                    let frame = room.locks_frame();
                    room.broadcast(&frame, None);
                }
                Msg::Unlock { uid } => {
                    if room.locks.get(&uid) == Some(&site) {
                        room.locks.remove(&uid);
                    }
                    let frame = room.locks_frame();
                    room.broadcast(&frame, None);
                }
                _ => {}
            }
        }
    })();

    let mut rooms = rooms.lock().expect("relay lock");
    if let Some(room) = rooms.get_mut(&room_name) {
        room.peers.remove(&site);
        room.looks.remove(&site);
        room.locks.retain(|_, s| *s != site);
        room.broadcast(&Frame::new(Msg::Left { site }), None);
        let frame = room.locks_frame();
        room.broadcast(&frame, None);
        // An empty room's scene stays, so the last one out can come back to
        // it: a relay that forgot on every disconnect would lose work to a
        // dropped Wi-Fi.
    }
    result
}
