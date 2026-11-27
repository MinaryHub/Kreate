//! The relay: the one place a session's edits are put in order.
//!
//! It keeps the session's scene, applying every patch as it passes, so a
//! participant who joins late starts from the scene as it stands rather than
//! from a replay of everything since the first. It does no editing of its
//! own; it orders, stores and forwards.
//!
//! Supports optional TLS 1.2/1.3 (`tls_cert`/`tls_key`), editor vs read-only
//! viewer keys, relay-enforced object edit locks, lock-free snapshot encoding,
//! and atomic room state persistence (`state_dir`).

use std::collections::{BTreeMap, HashMap};
use std::io::{self, BufReader, BufWriter, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use kreate_scene::{DatablockKind, Document, Uid};

use crate::wire::{site_color, Frame, Msg, Peer, Presence, Role, PROTOCOL};

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
    /// What an editor must present to come in. Empty lets anyone in as editor.
    pub key: String,
    /// Optional read-only key. Presenting this (when non-empty and different
    /// from `key`) admits a participant with [`Role::Viewer`].
    pub viewer_key: Option<String>,
    /// The most people in one room.
    pub max_peers: usize,
    /// Optional directory where room scenes are persisted atomically across
    /// relay restarts.
    pub state_dir: Option<PathBuf>,
    /// Optional PEM certificate chain path to enable TLS.
    pub tls_cert: Option<PathBuf>,
    /// Optional PEM private key path to enable TLS.
    pub tls_key: Option<PathBuf>,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            key: String::new(),
            viewer_key: None,
            max_peers: 8,
            state_dir: None,
            tls_cert: None,
            tls_key: None,
        }
    }
}

type RoomMap = Mutex<HashMap<String, Arc<Mutex<Room>>>>;

/// A running relay. Dropping it stops taking new participants and closes the
/// ones it has.
pub struct Relay {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    rooms: Arc<RoomMap>,
    accept_thread: Option<thread::JoinHandle<()>>,
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

    /// Collects the channels and streams of participants waiting for the seed
    /// along with a copy-on-write snapshot of `doc` so encoding runs outside
    /// the room mutex.
    fn take_pending_snapshot(&mut self) -> Option<PendingSnapshotJob> {
        let waiting = std::mem::take(&mut self.pending);
        if waiting.is_empty() {
            return None;
        }
        let targets: Vec<_> = waiting
            .into_iter()
            .filter_map(|site| {
                let m = self.peers.get(&site)?;
                let s = m.stream.try_clone().ok()?;
                Some((m.tx.clone(), s))
            })
            .collect();
        if targets.is_empty() {
            return None;
        }
        Some((self.doc.clone(), self.seq, targets))
    }

    fn locks_frame(&self) -> Frame {
        Frame::new(Msg::Locks {
            held: self.locks.iter().map(|(u, s)| (*u, *s)).collect(),
        })
    }
}

type PendingSnapshotJob = (Document, u64, Vec<(SyncSender<Frame>, TcpStream)>);

fn deliver_pending_snapshot(job: Option<PendingSnapshotJob>) {
    let Some((doc, seq, targets)) = job else {
        return;
    };
    let bytes = kreate_io::sync::encode_all(&doc).ok();
    for (tx, stream) in targets {
        match &bytes {
            Some(b) => {
                if let Err(TrySendError::Full(_)) =
                    tx.try_send(Frame::with_body(Msg::Snapshot { seq }, b.clone()))
                {
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                }
            }
            None => {
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        }
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

/// Constant-time string comparison for session keys.
fn same_secret(a: &str, b: &str) -> bool {
    let ab = a.as_bytes();
    let bb = b.as_bytes();
    let max_len = ab.len().max(bb.len());
    let mut diff = ab.len() ^ bb.len();
    for i in 0..max_len {
        let x = ab.get(i).copied().unwrap_or(0);
        let y = bb.get(i).copied().unwrap_or(0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}

fn room_state_path(dir: &Path, room_name: &str) -> PathBuf {
    let mut hex = String::with_capacity(room_name.len() * 2 + 6);
    for b in room_name.as_bytes() {
        use std::fmt::Write as _;
        let _ = write!(&mut hex, "{b:02x}");
    }
    hex.push_str(".ksnap");
    dir.join(hex)
}

fn save_room_snapshot(dir: &Path, room_name: &str, doc: &Document) {
    let Ok(bytes) = kreate_io::sync::encode_all(doc) else {
        return;
    };
    let target = room_state_path(dir, room_name);
    let tmp = target.with_extension(format!("ksnap.tmp.{}", std::process::id()));
    let write_res = (|| -> io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&bytes)?;
        f.sync_all()?;
        Ok(())
    })();
    if write_res.is_ok() {
        let _ = std::fs::rename(&tmp, &target);
    } else {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn load_room_snapshot(dir: &Path, room_name: &str) -> Option<Document> {
    let path = room_state_path(dir, room_name);
    let bytes = std::fs::read(path).ok()?;
    let mut doc = Document::new();
    kreate_io::sync::apply(&mut doc, &bytes).ok()?;
    Some(doc)
}

/// Returns `true` if the patch touches an object (or a mesh belonging to an
/// object) currently locked by another site.
fn patch_hits_foreign_lock(room: &Room, site: u16, body: &[u8]) -> bool {
    if room.locks.is_empty() {
        return false;
    }
    let Some(touched) = kreate_io::sync::peek_touched_uids(body) else {
        return false;
    };
    for (kind, uid) in touched {
        if room.locks.get(&uid).is_some_and(|owner| *owner != site) {
            return true;
        }
        if kind == Some(DatablockKind::Mesh) {
            for (&locked_obj_uid, &owner) in &room.locks {
                if owner == site {
                    continue;
                }
                if let Some(obj_id) = room.doc.object_by_uid(Uid::from_raw(locked_obj_uid)) {
                    if let Some(mesh_id) = room.doc.object(obj_id).and_then(|o| o.mesh()) {
                        if room.doc.mesh_uid(mesh_id).map(|u| u.raw()) == Some(uid) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

enum PeerInput {
    Plain(BufReader<TcpStream>),
    Tls(Receiver<io::Result<Frame>>),
}

impl PeerInput {
    fn read_frame(&mut self) -> io::Result<Frame> {
        match self {
            PeerInput::Plain(r) => Frame::read_from(r),
            PeerInput::Tls(rx) => rx.recv().map_err(|_| {
                io::Error::new(io::ErrorKind::UnexpectedEof, "tls pump disconnected")
            })?,
        }
    }
}

impl Relay {
    /// Starts listening on `addr` (port 0 picks one).
    pub fn start(addr: &str, config: RelayConfig) -> io::Result<Self> {
        let tls_config = match (&config.tls_cert, &config.tls_key) {
            (Some(cert), Some(key)) => Some(crate::tls::load_server_config(cert, key)?),
            (None, None) => None,
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "both tls_cert and tls_key must be provided to enable TLS",
                ));
            }
        };
        if let Some(dir) = &config.state_dir {
            std::fs::create_dir_all(dir)?;
        }
        let listener = TcpListener::bind(addr)?;
        let addr = listener.local_addr()?;
        listener.set_nonblocking(true)?;
        let stop = Arc::new(AtomicBool::new(false));
        let rooms: Arc<RoomMap> = Arc::default();
        let accept_thread = {
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
                                let tls_config = tls_config.clone();
                                let open = open.clone();
                                let spawned = thread::Builder::new()
                                    .name("kreate-relay-peer".into())
                                    .spawn({
                                        let open = open.clone();
                                        move || {
                                            let _ = serve(stream, &rooms, &config, tls_config);
                                            open.fetch_sub(1, Ordering::Relaxed);
                                        }
                                    });
                                if spawned.is_err() {
                                    open.fetch_sub(1, Ordering::Relaxed);
                                }
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                                thread::sleep(Duration::from_millis(10));
                            }
                            Err(_) => thread::sleep(Duration::from_millis(10)),
                        }
                    }
                })?
        };
        Ok(Self {
            addr,
            stop,
            rooms,
            accept_thread: Some(accept_thread),
        })
    }

    /// Where it is listening.
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }

    /// How many people are in all its rooms.
    pub fn peer_count(&self) -> usize {
        self.rooms
            .lock()
            .map(|r| {
                r.values()
                    .map(|room| room.lock().map(|g| g.peers.len()).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.accept_thread.take() {
            let _ = handle.join();
        }
        if let Ok(rooms) = self.rooms.lock() {
            for room in rooms.values() {
                if let Ok(guard) = room.lock() {
                    for member in guard.peers.values() {
                        let _ = member.stream.shutdown(std::net::Shutdown::Both);
                    }
                }
            }
        }
    }
}

/// One participant, from the door to leaving.
fn serve(
    mut stream: TcpStream,
    rooms: &RoomMap,
    config: &RelayConfig,
    tls_config: Option<Arc<rustls::ServerConfig>>,
) -> io::Result<()> {
    stream.set_nonblocking(false)?;
    stream.set_nodelay(true)?;

    let (hello, mut tls_state) = if let Some(cfg) = tls_config {
        let (conn, in_buf, hello) =
            crate::tls::server_handshake_and_hello(&mut stream, cfg, MAX_HELLO)?;
        (hello, Some((conn, in_buf)))
    } else {
        let mut reader = BufReader::new(stream.try_clone()?);
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        // Bounded before anything is known of who is knocking: the general limit
        // is a gigabyte, which is a scene's, not a hello's.
        let hello = Frame::read_limited(&mut reader, MAX_HELLO, 0)?;
        stream.set_read_timeout(None)?;
        (hello, None)
    };

    let mut refuse = |reason: &str| -> io::Result<()> {
        let frame = Frame::new(Msg::Refused {
            reason: reason.into(),
        });
        if let Some((conn, _)) = tls_state.as_mut() {
            crate::tls::write_server_frame(conn, &mut stream, &frame)
        } else {
            let mut w = BufWriter::new(stream.try_clone()?);
            frame.write_to(&mut w)
        }
    };

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
    let role = if same_secret(&key, &config.key) {
        Role::Editor
    } else if config
        .viewer_key
        .as_deref()
        .is_some_and(|vk| !vk.is_empty() && same_secret(&key, vk))
    {
        Role::Viewer
    } else {
        return refuse("wrong session key");
    };
    if room_name.is_empty() || room_name.chars().count() > 64 {
        return refuse("a session's name is 1 to 64 characters");
    }

    // Set up transport I/O channels after Hello validation.
    let (tx, rx) = mpsc::sync_channel::<Frame>(QUEUE);
    let mut input = if let Some((conn, in_buf)) = tls_state.take() {
        let (in_tx, in_rx) = mpsc::channel::<io::Result<Frame>>();
        let pump_stream = stream.try_clone()?;
        thread::Builder::new()
            .name("kreate-relay-tls".into())
            .spawn(move || {
                crate::tls::pump_server_tls(pump_stream, conn, in_buf, rx, in_tx);
            })?;
        PeerInput::Tls(in_rx)
    } else {
        let mut writer = BufWriter::new(stream.try_clone()?);
        thread::spawn(move || {
            for frame in rx {
                if frame.write_to(&mut writer).is_err() {
                    break;
                }
            }
        });
        PeerInput::Plain(BufReader::new(stream.try_clone()?))
    };

    // Look up or create this room's dedicated mutex so traffic in one room
    // never blocks other rooms on the relay.
    let room_handle = {
        let mut map = rooms.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(existing) = map.get(&room_name) {
            existing.clone()
        } else {
            if map.len() >= MAX_ROOMS {
                let empty = map
                    .iter()
                    .find(|(_, r)| r.lock().map(|g| g.peers.is_empty()).unwrap_or(false))
                    .map(|(k, _)| k.clone());
                match empty {
                    Some(k) => {
                        map.remove(&k);
                    }
                    None => {
                        drop(map);
                        let _ = tx.try_send(Frame::new(Msg::Refused {
                            reason: "the relay has no room for another session".into(),
                        }));
                        return Ok(());
                    }
                }
            }
            let mut initial = Room::default();
            if let Some(dir) = &config.state_dir {
                if let Some(restored) = load_room_snapshot(dir, &room_name) {
                    initial.doc = restored;
                    initial.seeded = true;
                }
            }
            let created = Arc::new(Mutex::new(initial));
            map.insert(room_name.clone(), created.clone());
            created
        }
    };

    let (site, join_snapshot) = {
        let mut room = room_handle.lock().unwrap_or_else(|p| p.into_inner());
        if room.peers.len() >= config.max_peers {
            drop(room);
            let _ = tx.try_send(Frame::new(Msg::Refused {
                reason: "the session is full".into(),
            }));
            return Ok(());
        }
        // Checked: numbers are never reused, and one that wrapped to 0 would be
        // a participant handing out ordinary, colliding object ids — or, in a
        // debug build, a panic with the lock held that poisons every room.
        let Some(site) = room.next_site.checked_add(1) else {
            drop(room);
            let _ = tx.try_send(Frame::new(Msg::Refused {
                reason: "the session has had too many participants; start a new one".into(),
            }));
            return Ok(());
        };
        room.next_site = site;
        let name = name.chars().take(40).collect::<String>();
        let peer = Peer {
            site,
            name,
            color: site_color(site),
            role,
        };
        // Only an editor may seed an empty room; viewers wait for the scene.
        let seeds = role == Role::Editor && !room.seeded && room.seeder.is_none();
        let _ = tx.try_send(Frame::new(Msg::Welcome {
            site,
            color: peer.color,
            peers: room.peers.values().map(|m| m.peer.clone()).collect(),
            seeded: !seeds,
            role,
        }));
        let join_snapshot = if seeds {
            room.seeder = Some(site);
            None
        } else if room.seeded {
            // Clone the CoW Document under the mutex and encode outside it.
            Some((room.doc.clone(), room.seq))
        } else {
            room.pending.push(site);
            None
        };
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
        (site, join_snapshot)
    };

    // Encode late-joiner snapshot outside the room mutex so other peers do not stall.
    if let Some((doc_snap, seq)) = join_snapshot {
        match kreate_io::sync::encode_all(&doc_snap) {
            Ok(bytes) => {
                let _ = tx.try_send(Frame::with_body(Msg::Snapshot { seq }, bytes));
            }
            Err(_) => {
                let _ = tx.try_send(Frame::new(Msg::Refused {
                    reason: "the session's scene could not be sent".into(),
                }));
                let _ = stream.shutdown(std::net::Shutdown::Both);
            }
        }
    }

    let result = (|| -> io::Result<()> {
        loop {
            let frame = input.read_frame()?;
            let mut pending_job = None;
            let mut lock_rollback = None;
            let mut persist_doc = None;
            {
                let mut room = room_handle.lock().unwrap_or_else(|p| p.into_inner());
                match frame.msg {
                    Msg::Patch if role == Role::Viewer => {
                        let _ = tx.try_send(Frame::new(Msg::Refused {
                            reason: "read-only viewer cannot modify the session".into(),
                        }));
                        return Ok(());
                    }
                    // Nothing but the seed counts before the room is seeded: an
                    // edit from someone still waiting for the scene is of a scene
                    // that is about to be replaced.
                    Msg::Patch if !room.seeded && room.seeder != Some(site) => {}
                    Msg::Patch => {
                        if room.seeded && patch_hits_foreign_lock(&room, site, &frame.body) {
                            let seq = room.seq;
                            if let Some(me) = room.peers.get(&site) {
                                me.send(Frame::new(Msg::Ack { seq }));
                            }
                            lock_rollback = Some((room.doc.clone(), seq));
                        } else {
                            // Kept only if it reads: a patch the relay cannot apply
                            // would be one the others cannot either. And the sender is
                            // let go rather than passed over in silence: it would wait
                            // for an answer that never comes, take the next one for
                            // this one's, and from then on be out of step for good.
                            if kreate_io::sync::apply(&mut room.doc, &frame.body).is_err() {
                                let _ = tx.try_send(Frame::new(Msg::Refused {
                                    reason: "a change could not be read by the session; join again"
                                        .into(),
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
                                pending_job = room.take_pending_snapshot();
                            } else {
                                room.broadcast(
                                    &Frame::with_body(Msg::Applied { from: site, seq }, frame.body),
                                    Some(site),
                                );
                            }
                            if config.state_dir.is_some() {
                                persist_doc = Some(room.doc.clone());
                            }
                        }
                    }
                    Msg::Look(presence) => {
                        room.looks.insert(site, presence.clone());
                        room.broadcast(&Frame::new(Msg::Seen { site, presence }), Some(site));
                    }
                    Msg::Lock { uid } => {
                        if role == Role::Editor {
                            room.locks.entry(uid).or_insert(site);
                            let frame = room.locks_frame();
                            room.broadcast(&frame, None);
                        }
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
            if let Some((doc_snap, seq)) = lock_rollback {
                if let Ok(bytes) = kreate_io::sync::encode_all(&doc_snap) {
                    let _ = tx.try_send(Frame::with_body(Msg::Snapshot { seq }, bytes));
                }
            }
            deliver_pending_snapshot(pending_job);
            if let (Some(dir), Some(doc_snap)) = (&config.state_dir, persist_doc) {
                save_room_snapshot(dir, &room_name, &doc_snap);
            }
        }
    })();

    let pending_job = {
        let mut room = room_handle.lock().unwrap_or_else(|p| p.into_inner());
        room.peers.remove(&site);
        room.pending.retain(|s| *s != site);
        let job = if room.seeder == Some(site) && !room.seeded {
            // Gone before its scene arrived: the room is seeded empty, and
            // those waiting start from that rather than waiting for ever.
            room.seeder = None;
            room.seeded = true;
            room.take_pending_snapshot()
        } else {
            None
        };
        room.looks.remove(&site);
        room.locks.retain(|_, s| *s != site);
        room.broadcast(&Frame::new(Msg::Left { site }), None);
        let frame = room.locks_frame();
        room.broadcast(&frame, None);
        job
    };
    deliver_pending_snapshot(pending_job);
    // An empty room's scene stays, so the last one out can come back to
    // it: a relay that forgot on every disconnect would lose work to a
    // dropped Wi-Fi.
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

    #[test]
    fn viewer_key_grants_read_only_access_and_refuses_patches() {
        let relay = Relay::start(
            "127.0.0.1:0",
            RelayConfig {
                key: "edit-secret".into(),
                viewer_key: Some("view-only".into()),
                ..Default::default()
            },
        )
        .unwrap();
        let at = format!("127.0.0.1:{}", relay.addr().port());
        let mut host_doc = Document::new();
        host_doc.add_mesh_object("Hero", kreate_mesh::primitives::cube(2.0));
        let mut editor = Session::join(&at, "Editor", "edit-secret").unwrap();
        settle(&mut [(&mut editor, &mut host_doc)]);

        let mut view_doc = Document::new();
        let mut viewer = Session::join(&at, "Viewer", "view-only").unwrap();
        settle(&mut [(&mut editor, &mut host_doc), (&mut viewer, &mut view_doc)]);
        assert_eq!(viewer.role, Role::Viewer);
        assert_eq!(view_doc.object_count(), 1);

        // Even if a raw client sends a Patch on a viewer connection, the relay refuses it.
        let raw_viewer = Connection::open(&at, "Rogue", "main", "view-only").unwrap();
        let mut refused = None;
        for _ in 0..200 {
            for item in raw_viewer.poll() {
                match item {
                    Incoming::Frame(f)
                        if matches!(
                            f.msg,
                            Msg::Welcome {
                                role: Role::Viewer,
                                ..
                            }
                        ) =>
                    {
                        let patch = kreate_io::sync::encode_all(&Document::new()).unwrap();
                        raw_viewer.send(Frame::with_body(Msg::Patch, patch));
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
        assert!(refused.is_some(), "relay must refuse patches from viewers");
    }

    #[test]
    fn relay_enforces_object_lock_against_other_peers() {
        let relay = Relay::start("127.0.0.1:0", RelayConfig::default()).unwrap();
        let at = format!("127.0.0.1:{}", relay.addr().port());
        let mut a = Document::new();
        let cube_a = a.add_mesh_object("LockedCube", kreate_mesh::primitives::cube(2.0));
        let mut sa = Session::join(&at, "A", "").unwrap();
        settle(&mut [(&mut sa, &mut a)]);

        let mut b = Document::new();
        let mut sb = Session::join(&at, "B", "").unwrap();
        settle(&mut [(&mut sa, &mut a), (&mut sb, &mut b)]);

        let cube_uid = a.object(cube_a).unwrap().uid.raw();
        sa.lock(cube_uid);
        settle(&mut [(&mut sa, &mut a), (&mut sb, &mut b)]);

        // B tries to move the cube locked by A.
        let cube_b = b.objects().next().unwrap().0;
        b.object_mut(cube_b).unwrap().transform.translation.x = 99.0;
        settle(&mut [(&mut sa, &mut a), (&mut sb, &mut b)]);

        // Relay rolled B back and never applied the edit to A.
        assert_eq!(a.object(cube_a).unwrap().transform.translation.x, 0.0);
        let cube_b_after = b.objects().next().unwrap().1;
        assert_eq!(cube_b_after.transform.translation.x, 0.0);
    }

    #[test]
    fn room_state_persists_across_relay_restart() {
        let dir = std::env::temp_dir().join(format!(
            "kreate-relay-state-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        {
            let relay = Relay::start(
                "127.0.0.1:0",
                RelayConfig {
                    state_dir: Some(dir.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
            let at = format!("127.0.0.1:{}", relay.addr().port());
            let mut a = Document::new();
            a.add_mesh_object("PersistedProp", kreate_mesh::primitives::cube(3.0));
            let mut sa = Session::join(&at, "A", "").unwrap();
            settle(&mut [(&mut sa, &mut a)]);
        }

        // Start a brand-new relay pointing at the same state directory.
        let relay2 = Relay::start(
            "127.0.0.1:0",
            RelayConfig {
                state_dir: Some(dir.clone()),
                ..Default::default()
            },
        )
        .unwrap();
        let at2 = format!("127.0.0.1:{}", relay2.addr().port());
        let mut late_doc = Document::new();
        let mut late = Session::join(&at2, "Late", "").unwrap();
        settle(&mut [(&mut late, &mut late_doc)]);
        assert!(late.is_live());
        assert_eq!(late_doc.object_count(), 1);
        assert_eq!(late_doc.objects().next().unwrap().1.name, "PersistedProp");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
