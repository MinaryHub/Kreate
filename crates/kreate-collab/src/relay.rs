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
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use kreate_scene::Document;

use crate::wire::{site_color, Frame, Msg, Peer, Presence, PROTOCOL};

/// Frames queued for one participant before it counts as not reading and is
/// let go. Without a bound, one that stopped reading would hold every edit
/// of the session in memory for as long as it stayed connected.
const QUEUE: usize = 1024;

/// Connections open at once, signed in or not. One thread each.
const MAX_CONNECTIONS: usize = 256;

/// Rooms kept. An empty room's scene is kept for its people to come back
/// to; past this many, the empty ones are let go first.
const MAX_ROOMS: usize = 64;

/// The largest hello: names and a key, never a scene.
const MAX_HELLO: usize = 64 * 1024;

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
    /// Who was told to seed the room and has not yet: until that scene
    /// arrives, anyone else who comes in waits for it in `pending` rather
    /// than being told to seed too — two seeds of two scenes, numbered from
    /// the same counter, overwrite each other's objects.
    seeder: Option<u16>,
    /// Participants waiting for the seed, to be sent it as a snapshot.
    pending: Vec<u16>,
    seq: u64,
    next_site: u16,
    peers: BTreeMap<u16, Member>,
    looks: HashMap<u16, Presence>,
    locks: BTreeMap<u64, u16>,
}

struct Member {
    peer: Peer,
    tx: SyncSender<Frame>,
    /// To close it from here when the relay stops.
    stream: TcpStream,
}

impl Room {
    fn broadcast(&self, frame: &Frame, except: Option<u16>) {
        for (site, member) in &self.peers {
            if Some(*site) != except {
                member.send(frame.clone());
            }
        }
    }

    /// The scene as it stands, to the participants waiting for the seed.
    fn send_pending(&mut self) {
        let waiting = std::mem::take(&mut self.pending);
        if waiting.is_empty() {
            return;
        }
        let bytes = kreate_io::sync::encode_all(&self.doc).ok();
        for site in waiting {
            if let Some(member) = self.peers.get(&site) {
                match &bytes {
                    Some(b) => {
                        member.send(Frame::with_body(Msg::Snapshot { seq: self.seq }, b.clone()))
                    }
                    None => member.close(),
                }
            }
        }
    }

    fn locks_frame(&self) -> Frame {
        Frame::new(Msg::Locks {
            held: self.locks.iter().map(|(u, s)| (*u, *s)).collect(),
        })
    }
}

impl Member {
    /// Queues a frame; a participant whose queue is full is not reading, and
    /// is let go rather than let grow it.
    fn send(&self, frame: Frame) {
        if let Err(TrySendError::Full(_)) = self.tx.try_send(frame) {
            self.close();
        }
    }

    fn close(&self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
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
                    let open = Arc::new(AtomicUsize::new(0));
                    while !stop.load(Ordering::Relaxed) {
                        match listener.accept() {
                            Ok((stream, _)) => {
                                if open.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
                                    drop(stream);
                                    continue;
                                }
                                open.fetch_add(1, Ordering::Relaxed);
                                let rooms = rooms.clone();
                                let config = config.clone();
                                let open = open.clone();
                                let spawned = thread::Builder::new()
                                    .name("kreate-relay-peer".into())
                                    .spawn({
                                        let open = open.clone();
                                        move || {
                                            let _ = serve(stream, &rooms, &config);
                                            open.fetch_sub(1, Ordering::Relaxed);
                                        }
                                    });
                                if spawned.is_err() {
                                    open.fetch_sub(1, Ordering::Relaxed);
                                }
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
    // Bounded before anything is known of who is knocking: the general limit
    // is a gigabyte, which is a scene's, not a hello's.
    let hello = Frame::read_limited(&mut reader, MAX_HELLO, 0)?;
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
    if room_name.is_empty() || room_name.chars().count() > 64 {
        return refuse("a session's name is 1 to 64 characters");
    }

    // The writer: its own thread, so a slow participant never holds the room.
    let (tx, rx) = mpsc::sync_channel::<Frame>(QUEUE);
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
        let mut rooms = rooms.lock().unwrap_or_else(|p| p.into_inner());
        if rooms
            .get(&room_name)
            .is_some_and(|room| room.peers.len() >= config.max_peers)
        {
            drop(rooms);
            return refuse("the session is full");
        }
        if !rooms.contains_key(&room_name) && rooms.len() >= MAX_ROOMS {
            // Room for one more by letting an empty one go, or none.
            let empty = rooms
                .iter()
                .find(|(_, r)| r.peers.is_empty())
                .map(|(k, _)| k.clone());
            match empty {
                Some(k) => {
                    rooms.remove(&k);
                }
                None => {
                    drop(rooms);
                    return refuse("the relay has no room for another session");
                }
            }
        }
        let room = rooms.entry(room_name.clone()).or_default();
        // Checked: numbers are never reused, and one that wrapped to 0 would be
        // a participant handing out ordinary, colliding object ids — or, in a
        // debug build, a panic with the lock held that poisons every room.
        let Some(site) = room.next_site.checked_add(1) else {
            drop(rooms);
            return refuse("the session has had too many participants; start a new one");
        };
        room.next_site = site;
        let name = name.chars().take(40).collect::<String>();
        let peer = Peer {
            site,
            name,
            color: site_color(site),
        };
        // The first in seeds the room; anyone after waits for the scene,
        // even while that seed is still on its way.
        let seeds = !room.seeded && room.seeder.is_none();
        let _ = tx.try_send(Frame::new(Msg::Welcome {
            site,
            color: peer.color,
            peers: room.peers.values().map(|m| m.peer.clone()).collect(),
            seeded: !seeds,
        }));
        if seeds {
            room.seeder = Some(site);
        } else if room.seeded {
            match kreate_io::sync::encode_all(&room.doc) {
                Ok(bytes) => {
                    let _ = tx.try_send(Frame::with_body(Msg::Snapshot { seq: room.seq }, bytes));
                }
                Err(_) => {
                    drop(rooms);
                    return refuse("the session's scene could not be sent");
                }
            }
        } else {
            room.pending.push(site);
        }
        for (other, presence) in &room.looks {
            let _ = tx.try_send(Frame::new(Msg::Seen {
                site: *other,
                presence: presence.clone(),
            }));
        }
        let _ = tx.try_send(room.locks_frame());
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
            let mut rooms = rooms.lock().unwrap_or_else(|p| p.into_inner());
            let Some(room) = rooms.get_mut(&room_name) else {
                return Ok(());
            };
            match frame.msg {
                // Nothing but the seed counts before the room is seeded: an
                // edit from someone still waiting for the scene is of a scene
                // that is about to be replaced.
                Msg::Patch if !room.seeded && room.seeder != Some(site) => {}
                Msg::Patch => {
                    // Kept only if it reads: a patch the relay cannot apply
                    // would be one the others cannot either. And the sender is
                    // let go rather than passed over in silence: it would wait
                    // for an answer that never comes, take the next one for
                    // this one's, and from then on be out of step for good.
                    if kreate_io::sync::apply(&mut room.doc, &frame.body).is_err() {
                        // Through the writer, which owns the stream's output:
                        // written from here it could land inside a frame.
                        let _ = tx.try_send(Frame::new(Msg::Refused {
                            reason: "a change could not be read by the session; join again".into(),
                        }));
                        return Ok(());
                    }
                    let seeding = !room.seeded;
                    room.seeded = true;
                    room.seeder = None;
                    room.seq += 1;
                    let seq = room.seq;
                    if let Some(me) = room.peers.get(&site) {
                        me.send(Frame::new(Msg::Ack { seq }));
                    }
                    if seeding {
                        // Those who waited get the scene whole, not the patch.
                        let waiting = room.pending.clone();
                        for (other, member) in &room.peers {
                            if *other != site && !waiting.contains(other) {
                                member.send(Frame::with_body(
                                    Msg::Applied { from: site, seq },
                                    frame.body.clone(),
                                ));
                            }
                        }
                        room.send_pending();
                    } else {
                        room.broadcast(
                            &Frame::with_body(Msg::Applied { from: site, seq }, frame.body),
                            Some(site),
                        );
                    }
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

    let mut rooms = rooms.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(room) = rooms.get_mut(&room_name) {
        room.peers.remove(&site);
        room.pending.retain(|s| *s != site);
        if room.seeder == Some(site) && !room.seeded {
            // Gone before its scene arrived: the room is seeded empty, and
            // those waiting start from that rather than waiting for ever.
            room.seeder = None;
            room.seeded = true;
            room.send_pending();
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{Connection, Incoming};
    use crate::session::Session;

    fn settle(sessions: &mut [(&mut Session, &mut Document)]) {
        for _ in 0..100 {
            for (s, d) in sessions.iter_mut() {
                s.tick(d, None);
                s.flush(d);
            }
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn two_who_join_an_empty_session_at_once_end_with_one_scene() {
        // Both knock before either has sent anything: only the first may seed,
        // and the second must end with the first's scene, not overwrite it.
        let relay = Relay::start("127.0.0.1:0", RelayConfig::default()).unwrap();
        let at = format!("127.0.0.1:{}", relay.addr().port());
        // Numbered from the same counter, A's first object and B's share an
        // identity: two seeds would overwrite one with the other and keep the
        // rest, a scene that is neither's.
        let mut a = Document::new();
        a.add_mesh_object("FromA", kreate_mesh::primitives::cube(1.0));
        a.add_mesh_object("AlsoA", kreate_mesh::primitives::cube(1.0));
        let mut b = Document::new();
        b.add_mesh_object("FromB", kreate_mesh::primitives::cube(2.0));
        let mut sa = Session::join(&at, "A", "").unwrap();
        let mut sb = Session::join(&at, "B", "").unwrap();
        settle(&mut [(&mut sa, &mut a), (&mut sb, &mut b)]);
        assert!(sa.is_live() && sb.is_live());
        let names = |d: &Document| {
            let mut n: Vec<String> = d.objects().map(|(_, o)| o.name.clone()).collect();
            n.sort();
            n
        };
        assert_eq!(names(&a), names(&b), "one scene, the same for both");
        let one_seed = names(&a) == ["AlsoA", "FromA"] || names(&a) == ["FromB"];
        assert!(
            one_seed,
            "one participant's scene, not a mix: {:?}",
            names(&a)
        );
    }

    #[test]
    fn a_patch_the_relay_cannot_read_ends_that_connection_and_says_why() {
        let relay = Relay::start("127.0.0.1:0", RelayConfig::default()).unwrap();
        let at = format!("127.0.0.1:{}", relay.addr().port());
        let conn = Connection::open(&at, "X", "main", "").unwrap();
        // The welcome first: this one seeds, so its first patch is read.
        let mut refused = None;
        for _ in 0..200 {
            for item in conn.poll() {
                match item {
                    Incoming::Frame(f) if matches!(f.msg, Msg::Welcome { .. }) => {
                        conn.send(Frame::with_body(Msg::Patch, b"not a patch".to_vec()));
                    }
                    Incoming::Frame(f) => {
                        if let Msg::Refused { reason } = f.msg {
                            refused = Some(reason);
                        }
                    }
                    Incoming::Closed(_) => {}
                }
            }
            if refused.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(refused.is_some(), "the sender is told, not left waiting");
    }

    #[test]
    fn a_hello_is_bounded_before_the_key_is_checked() {
        use std::io::Write;
        let relay = Relay::start("127.0.0.1:0", RelayConfig::default()).unwrap();
        let mut s = TcpStream::connect(relay.addr()).unwrap();
        // A hello announcing a gigabyte of body: refused on the header alone.
        let mut head = Vec::new();
        head.extend_from_slice(&2u32.to_be_bytes());
        head.extend_from_slice(&(1u32 << 30).to_be_bytes());
        s.write_all(&head).unwrap();
        s.write_all(b"{}").unwrap();
        s.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut buf = [0u8; 1];
        let read = std::io::Read::read(&mut s, &mut buf);
        assert!(matches!(read, Ok(0) | Err(_)), "the connection is dropped");
        assert_eq!(relay.peer_count(), 0);
    }
}
