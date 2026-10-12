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
    /// Connects and says hello. The relay's answer arrives through
    /// [`Connection::poll`].
    pub fn open(addr: &str, name: &str, room: &str, key: &str) -> io::Result<Self> {
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
