//! A participant's connection to a relay.
//!
//! Two threads behind a pair of channels: one reads frames into
//! [`Connection::poll`], one writes what [`Connection::send`] queues. The
//! application's frame never waits on the network.

use std::io::{self, BufReader, BufWriter};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::thread;
use std::time::Duration;

use crate::wire::{Frame, Msg, PROTOCOL};

/// What arrives, or that the line went.
#[derive(Debug)]
pub enum Incoming {
    Frame(Frame),
    Closed(String),
}

/// An open connection to a relay.
pub struct Connection {
    tx: Sender<Frame>,
    rx: Receiver<Incoming>,
    stream: TcpStream,
}

impl Connection {
    /// Connects and says hello. Addresses prefixed with `tls://` negotiate TLS
    /// using the operating system's root certificate store. The relay's answer
    /// arrives through [`Connection::poll`].
    pub fn open(addr: &str, name: &str, room: &str, key: &str) -> io::Result<Self> {
        if let Some((sni, dial)) = crate::tls::parse_tls_addr(addr) {
            let tls_cfg = crate::tls::default_client_config()?;
            return Self::open_tls_with_config(&dial, sni, tls_cfg, name, room, key);
        }
        let target = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such address"))?;
        let stream = TcpStream::connect_timeout(&target, Duration::from_secs(5))?;
        stream.set_nodelay(true)?;
        let mut writer = BufWriter::new(stream.try_clone()?);
        Frame::new(Msg::Hello {
            protocol: PROTOCOL,
            name: name.into(),
            room: room.into(),
            key: key.into(),
        })
        .write_to(&mut writer)?;

        let (tx, out) = mpsc::channel::<Frame>();
        thread::Builder::new()
            .name("kreate-collab-send".into())
            .spawn(move || {
                for frame in out {
                    if frame.write_to(&mut writer).is_err() {
                        break;
                    }
                }
            })?;
        let (inbox, rx) = mpsc::channel::<Incoming>();
        let mut reader = BufReader::new(stream.try_clone()?);
        thread::Builder::new()
            .name("kreate-collab-recv".into())
            .spawn(move || loop {
                match Frame::read_from(&mut reader) {
                    Ok(frame) => {
                        if inbox.send(Incoming::Frame(frame)).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = inbox.send(Incoming::Closed(e.to_string()));
                        break;
                    }
                }
            })?;
        Ok(Self { tx, rx, stream })
    }

    /// Connects over TLS using a caller-supplied `rustls::ClientConfig` (for
    /// custom enterprise CAs or self-signed studio relays).
    pub fn open_tls_with_config(
        dial_addr: &str,
        server_name: rustls::pki_types::ServerName<'static>,
        tls_config: std::sync::Arc<rustls::ClientConfig>,
        name: &str,
        room: &str,
        key: &str,
    ) -> io::Result<Self> {
        let target = dial_addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no such address"))?;
        let mut stream = TcpStream::connect_timeout(&target, Duration::from_secs(5))?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(Duration::from_secs(10)))?;
        let mut conn = rustls::ClientConnection::new(tls_config, server_name)
            .map_err(|e| io::Error::other(format!("tls client init: {e}")))?;
        while conn.is_handshaking() {
            conn.complete_io(&mut stream)?;
        }
        let hello = Frame::new(Msg::Hello {
            protocol: PROTOCOL,
            name: name.into(),
            room: room.into(),
            key: key.into(),
        });
        crate::tls::write_client_frame(&mut conn, &mut stream, &hello)?;
        let handle = stream.try_clone()?;
        let (tx, out) = mpsc::channel::<Frame>();
        let (inbox, rx) = mpsc::channel::<Incoming>();
        thread::Builder::new()
            .name("kreate-collab-tls".into())
            .spawn(move || {
                crate::tls::pump_client_tls(stream, conn, out, inbox);
            })?;
        Ok(Self {
            tx,
            rx,
            stream: handle,
        })
    }

    pub fn send(&self, frame: Frame) {
        let _ = self.tx.send(frame);
    }

    /// Everything that has arrived, without waiting.
    pub fn poll(&self) -> Vec<Incoming> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(item) => out.push(item),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    out.push(Incoming::Closed("connection closed".into()));
                    break;
                }
            }
        }
        out
    }

    /// Waits up to `timeout` for the next arrival. For tests and tools.
    pub fn wait(&self, timeout: Duration) -> Option<Incoming> {
        self.rx.recv_timeout(timeout).ok()
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
    }
}
