//! A participant, whole: the connection, the replica and who else is there,
//! driven by the application once a frame.

use std::collections::{BTreeMap, HashMap};
use std::io;
use std::time::{Duration, Instant};

use kreate_scene::Document;

use crate::client::{Connection, Incoming};
use crate::relay::{Relay, RelayConfig};
use crate::replica::Replica;
use crate::wire::{Frame, Msg, Peer, Presence};

/// How often one's own edits go out while they are being made: a drag is
/// seen moving, without a patch per frame.
const SEND_EVERY: Duration = Duration::from_millis(50);
/// How often where one is looking goes out.
const LOOK_EVERY: Duration = Duration::from_millis(100);

/// Something the application may want to tell the person.
#[derive(Clone, Debug, PartialEq)]
pub enum SessionEvent {
    /// In, as participant `site`.
    Connected {
        site: u16,
    },
    /// The scene was replaced by the session's: history no longer applies.
    SceneReplaced,
    /// Someone else's edit changed the scene.
    Changed,
    Joined(Peer),
    Left(Peer),
    /// The relay would not let us in, or the line went.
    Closed(String),
}

#[derive(Clone, Debug, PartialEq)]
enum State {
    Knocking,
    AwaitingScene,
    Live,
    Closed,
}

/// A shared-editing session as one participant sees it.
pub struct Session {
    conn: Connection,
    /// The relay this participant is hosting, if it is.
    relay: Option<Relay>,
    replica: Option<Replica>,
    state: State,
    /// Where the relay is, as typed or as hosted.
    pub address: String,
    pub name: String,
    pub color: [u8; 3],
    pub peers: BTreeMap<u16, Peer>,
    pub looks: HashMap<u16, Presence>,
    /// Who is editing the inside of what: object identity → participant.
    pub locks: BTreeMap<u64, u16>,
    last_send: Instant,
    last_look: Option<(Presence, Instant)>,
}

impl Session {
    /// Hosts a session on this machine at `port` and joins it.
    ///
    /// The scene open here becomes the session's.
    pub fn host(port: u16, name: &str, key: &str) -> io::Result<Self> {
        let relay = Relay::start(
            &format!("0.0.0.0:{port}"),
            RelayConfig {
                key: key.into(),
                ..Default::default()
            },
        )?;
        let local = format!("127.0.0.1:{}", relay.addr().port());
        let mut session = Self::join(&local, name, key)?;
        session.address = format!("{}:{}", lan_address(), relay.addr().port());
        session.relay = Some(relay);
        Ok(session)
    }

    /// Joins the session at `address`. Its scene replaces the one open here.
    pub fn join(address: &str, name: &str, key: &str) -> io::Result<Self> {
        let address = if address.contains(':') {
            address.to_string()
        } else {
            format!("{address}:{}", crate::wire::DEFAULT_PORT)
        };
        let conn = Connection::open(&address, name, "main", key)?;
        Ok(Self {
            conn,
            relay: None,
            replica: None,
            state: State::Knocking,
            address,
            name: name.into(),
            color: [255, 255, 255],
            peers: BTreeMap::new(),
            looks: HashMap::new(),
            locks: BTreeMap::new(),
            last_send: Instant::now(),
            last_look: None,
        })
    }

    pub fn is_hosting(&self) -> bool {
        self.relay.is_some()
    }

    pub fn is_live(&self) -> bool {
        self.state == State::Live
    }

    pub fn is_closed(&self) -> bool {
        self.state == State::Closed
    }

    /// This participant's number, once in.
    pub fn site(&self) -> Option<u16> {
        self.replica.as_ref().map(|r| r.site())
    }

    /// Patches not yet taken by the relay.
    pub fn waiting(&self) -> usize {
        self.replica.as_ref().map_or(0, |r| r.waiting())
    }

    /// Once a frame: sends what changed here, takes in what changed
    /// elsewhere, and says where this participant is looking.
    pub fn tick(&mut self, doc: &mut Document, look: Option<Presence>) -> Vec<SessionEvent> {
        let mut events = Vec::new();
        let arrived = self.conn.poll();
        let remote_edit = arrived
            .iter()
            .any(|i| matches!(i, Incoming::Frame(f) if matches!(f.msg, Msg::Applied { .. })));

        // Ours first, always before theirs land: see `Replica::incoming`.
        if self.state == State::Live && (remote_edit || self.last_send.elapsed() >= SEND_EVERY) {
            self.flush(doc);
        }

        for item in arrived {
            match item {
                Incoming::Frame(frame) => self.take(frame, doc, &mut events),
                Incoming::Closed(reason) => {
                    if self.state != State::Closed {
                        self.state = State::Closed;
                        events.push(SessionEvent::Closed(reason));
                    }
                }
            }
        }

        if self.state == State::Live {
            if let Some(look) = look {
                let due = match &self.last_look {
                    Some((last, at)) => *last != look && at.elapsed() >= LOOK_EVERY,
                    None => true,
                };
                if due {
                    self.conn.send(Frame::new(Msg::Look(look.clone())));
                    self.last_look = Some((look, Instant::now()));
                }
            }
        }
        events
    }

    /// Sends whatever changed here now, without waiting for the next turn.
    pub fn flush(&mut self, doc: &Document) {
        if self.state != State::Live {
            return;
        }
        if let Some(replica) = &mut self.replica {
            if let Some(bytes) = replica.outgoing(doc) {
                self.conn.send(Frame::with_body(Msg::Patch, bytes));
            }
        }
        self.last_send = Instant::now();
    }

    /// After an undo or redo here: keeps everyone else's edits. See
    /// [`Replica::keep_others`].
    pub fn after_undo(&mut self, doc: &mut Document) {
        if let Some(replica) = &mut self.replica {
            replica.keep_others(doc);
        }
    }

    /// Asks to edit the inside of `uid` alone.
    pub fn lock(&mut self, uid: u64) {
        self.conn.send(Frame::new(Msg::Lock { uid }));
    }

    pub fn unlock(&mut self, uid: u64) {
        self.conn.send(Frame::new(Msg::Unlock { uid }));
    }

    /// Who else is editing the inside of `uid`, if anyone.
    pub fn held_by_other(&self, uid: u64) -> Option<&Peer> {
        let site = *self.locks.get(&uid)?;
        if Some(site) == self.site() {
            return None;
        }
        self.peers.get(&site)
    }

    fn take(&mut self, frame: Frame, doc: &mut Document, events: &mut Vec<SessionEvent>) {
        match frame.msg {
            Msg::Welcome {
                site,
                color,
                peers,
                seeded,
            } => {
                self.color = color;
                self.peers = peers.into_iter().map(|p| (p.site, p)).collect();
                let mut replica = Replica::new(doc, site);
                if seeded {
                    self.state = State::AwaitingScene;
                } else {
                    if let Some(bytes) = replica.seed(doc) {
                        self.conn.send(Frame::with_body(Msg::Patch, bytes));
                    }
                    self.state = State::Live;
                }
                self.replica = Some(replica);
                events.push(SessionEvent::Connected { site });
            }
            Msg::Snapshot { .. } => {
                if let Some(replica) = &mut self.replica {
                    match replica.adopt_snapshot(doc, &frame.body) {
                        Ok(()) => {
                            self.state = State::Live;
                            events.push(SessionEvent::SceneReplaced);
                        }
                        Err(e) => {
                            self.state = State::Closed;
                            events.push(SessionEvent::Closed(e.to_string()));
                        }
                    }
                }
            }
            Msg::Applied { from, .. } => {
                if let Some(replica) = &mut self.replica {
                    if let Ok(true) = replica.incoming(doc, from, &frame.body) {
                        if events.last() != Some(&SessionEvent::Changed) {
                            events.push(SessionEvent::Changed);
                        }
                    }
                }
            }
            Msg::Ack { .. } => {
                if let Some(replica) = &mut self.replica {
                    replica.ack();
                }
            }
            Msg::Joined(peer) => {
                self.peers.insert(peer.site, peer.clone());
                events.push(SessionEvent::Joined(peer));
            }
            Msg::Left { site } => {
                self.looks.remove(&site);
                if let Some(peer) = self.peers.remove(&site) {
                    events.push(SessionEvent::Left(peer));
                }
            }
            Msg::Seen { site, presence } => {
                self.looks.insert(site, presence);
            }
            Msg::Locks { held } => {
                self.locks = held.into_iter().collect();
            }
            Msg::Refused { reason } => {
                self.state = State::Closed;
                events.push(SessionEvent::Closed(reason));
            }
            _ => {}
        }
    }
}

/// This machine's address on the local network, for telling others where to
/// join: the interface a packet to the outside would leave by. Nothing is
/// sent; connecting a UDP socket only picks a route.
pub fn lan_address() -> String {
    std::net::UdpSocket::bind("0.0.0.0:0")
        .and_then(|s| {
            s.connect("192.0.2.1:9")?;
            s.local_addr()
        })
        .map(|a| a.ip().to_string())
        .unwrap_or_else(|_| "127.0.0.1".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(sessions: &mut [(&mut Session, &mut Document)]) {
        for _ in 0..100 {
            for (s, d) in sessions.iter_mut() {
                s.tick(d, None);
                s.flush(d);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn two_people_edit_one_scene_over_the_network() {
        let mut a = Document::new();
        a.add_mesh_object("Cube", kreate_mesh::primitives::cube(2.0));
        let mut host = Session::host(0, "A", "secret").unwrap();
        let port = host.relay.as_ref().unwrap().addr().port();
        let mut b = Document::new();
        b.add_mesh_object("Lost", kreate_mesh::primitives::cube(1.0));
        // Give the host's seed time to land before B knocks.
        settle(&mut [(&mut host, &mut a)]);
        let mut guest = Session::join(&format!("127.0.0.1:{port}"), "B", "secret").unwrap();
        settle(&mut [(&mut host, &mut a), (&mut guest, &mut b)]);
        assert!(guest.is_live());
        assert_eq!(b.object_count(), 1, "B's scene is the session's");
        assert_eq!(b.objects().next().unwrap().1.name, "Cube");
        assert_eq!(host.peers.len(), 1);

        b.add_mesh_object("FromB", kreate_mesh::primitives::cube(1.0));
        let cube = a.objects().next().unwrap().0;
        a.object_mut(cube).unwrap().transform.translation.z = 2.0;
        settle(&mut [(&mut host, &mut a), (&mut guest, &mut b)]);
        for doc in [&a, &b] {
            assert_eq!(doc.object_count(), 2);
            let cube = doc.objects().find(|(_, o)| o.name == "Cube").unwrap().1;
            assert_eq!(cube.transform.translation.z, 2.0);
        }
        assert_eq!(host.waiting(), 0);
        assert_eq!(guest.waiting(), 0);
    }

    #[test]
    fn the_wrong_key_is_turned_away() {
        let mut a = Document::new();
        let mut host = Session::host(0, "A", "right").unwrap();
        let port = host.relay.as_ref().unwrap().addr().port();
        let mut b = Document::new();
        let mut guest = Session::join(&format!("127.0.0.1:{port}"), "B", "wrong").unwrap();
        let mut closed = false;
        for _ in 0..100 {
            host.tick(&mut a, None);
            closed |= guest
                .tick(&mut b, None)
                .iter()
                .any(|e| matches!(e, SessionEvent::Closed(_)));
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(closed);
        assert!(!guest.is_live());
    }

    #[test]
    fn locks_and_looks_reach_the_others() {
        let mut a = Document::new();
        let mut host = Session::host(0, "A", "").unwrap();
        let port = host.relay.as_ref().unwrap().addr().port();
        let mut b = Document::new();
        settle(&mut [(&mut host, &mut a)]);
        let mut guest = Session::join(&format!("127.0.0.1:{port}"), "B", "").unwrap();
        settle(&mut [(&mut host, &mut a), (&mut guest, &mut b)]);
        host.lock(42);
        let look = Presence {
            eye: [1.0, 2.0, 3.0],
            ..Default::default()
        };
        for _ in 0..50 {
            host.tick(&mut a, Some(look.clone()));
            guest.tick(&mut b, None);
            std::thread::sleep(Duration::from_millis(10));
        }
        let host_site = host.site().unwrap();
        assert_eq!(guest.held_by_other(42).map(|p| p.site), Some(host_site));
        assert_eq!(host.held_by_other(42), None, "one's own lock");
        assert_eq!(guest.looks.get(&host_site).map(|l| l.eye), Some(look.eye));
    }
}
