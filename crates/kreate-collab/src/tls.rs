//! Optional TLS 1.2/1.3 transport helpers for `kreate-collab` using `rustls`.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rustls::pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection};

use crate::wire::{Frame, Msg, DEFAULT_PORT, MAX_FRAME, MAX_HEADER};

/// Builds a `rustls::ServerConfig` from PEM certificate chain and private key files.
pub fn load_server_config(cert_path: &Path, key_path: &Path) -> io::Result<Arc<ServerConfig>> {
    let certs: Vec<CertificateDer<'static>> = CertificateDer::pem_file_iter(cert_path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("tls cert: {e}")))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("tls cert: {e}")))?;
    if certs.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "tls certificate file contains no certificates",
        ));
    }
    let key = PrivateKeyDer::from_pem_file(key_path)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("tls key: {e}")))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| io::Error::other(format!("tls config: {e}")))?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("tls cert/key: {e}")))?;
    Ok(Arc::new(cfg))
}

/// Builds a `rustls::ClientConfig` trusting the operating system's native root CAs.
pub fn default_client_config() -> io::Result<Arc<ClientConfig>> {
    let mut roots = RootCertStore::empty();
    let loaded = rustls_native_certs::load_native_certs();
    for cert in loaded.certs {
        let _ = roots.add(cert);
    }
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let cfg = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| io::Error::other(format!("tls config: {e}")))?
        .with_root_certificates(roots)
        .with_no_client_auth();
    Ok(Arc::new(cfg))
}

/// Parses a `tls://host[:port]` address into `(server_name, "host:port")`.
/// Returns `None` for plain TCP addresses.
pub fn parse_tls_addr(addr: &str) -> Option<(ServerName<'static>, String)> {
    let rest = addr.strip_prefix("tls://")?;
    let (host, dial) = if let Some(bracketed) = rest.strip_prefix('[') {
        let end = bracketed.find(']')?;
        let host = &bracketed[..end];
        let after = &bracketed[end + 1..];
        let dial = if after.starts_with(':') {
            rest.to_string()
        } else {
            format!("[{host}]:{DEFAULT_PORT}")
        };
        (host, dial)
    } else if let Some((h, _p)) = rest.rsplit_once(':') {
        (h, rest.to_string())
    } else {
        (rest, format!("{rest}:{DEFAULT_PORT}"))
    };
    let sni = ServerName::try_from(host.to_string()).ok()?;
    Some((sni, dial))
}

/// Try to pop one complete [`Frame`] from `buf`, enforcing header and body caps
/// before copying. Returns `Ok(None)` if more bytes are needed.
pub fn try_pop_frame(
    buf: &mut Vec<u8>,
    max_header: usize,
    max_body: usize,
) -> io::Result<Option<Frame>> {
    if buf.len() < 8 {
        return Ok(None);
    }
    let header_len = u32::from_be_bytes(buf[..4].try_into().unwrap()) as usize;
    let body_len = u32::from_be_bytes(buf[4..8].try_into().unwrap()) as usize;
    if header_len > max_header || body_len > max_body {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "a frame larger than the protocol allows",
        ));
    }
    let total = 8usize
        .checked_add(header_len)
        .and_then(|n| n.checked_add(body_len))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "frame length overflow"))?;
    if buf.len() < total {
        return Ok(None);
    }
    let msg: Msg = serde_json::from_slice(&buf[8..8 + header_len])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let body = buf[8 + header_len..total].to_vec();
    buf.drain(..total);
    Ok(Some(Frame { msg, body }))
}

/// Flushes all pending encrypted TLS output bytes on `stream`.
pub fn flush_server_tls(conn: &mut ServerConnection, stream: &mut TcpStream) -> io::Result<()> {
    while conn.wants_write() {
        conn.write_tls(stream)?;
    }
    stream.flush()
}

/// Flushes all pending encrypted TLS output bytes on `stream`.
pub fn flush_client_tls(conn: &mut ClientConnection, stream: &mut TcpStream) -> io::Result<()> {
    while conn.wants_write() {
        conn.write_tls(stream)?;
    }
    stream.flush()
}

/// Serializes and sends one [`Frame`] over a `ServerConnection`.
pub fn write_server_frame(
    conn: &mut ServerConnection,
    stream: &mut TcpStream,
    frame: &Frame,
) -> io::Result<()> {
    frame.write_to(&mut conn.writer())?;
    flush_server_tls(conn, stream)
}

/// Serializes and sends one [`Frame`] over a `ClientConnection`.
pub fn write_client_frame(
    conn: &mut ClientConnection,
    stream: &mut TcpStream,
    frame: &Frame,
) -> io::Result<()> {
    frame.write_to(&mut conn.writer())?;
    flush_client_tls(conn, stream)
}

/// Completes a server TLS handshake and reads the initial bounded `Hello` frame.
pub fn server_handshake_and_hello(
    stream: &mut TcpStream,
    config: Arc<ServerConfig>,
    max_hello: usize,
) -> io::Result<(ServerConnection, Vec<u8>, Frame)> {
    let mut conn = ServerConnection::new(config)
        .map_err(|e| io::Error::other(format!("tls server init: {e}")))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    while conn.is_handshaking() {
        conn.complete_io(stream)?;
    }
    let mut in_buf = Vec::new();
    let mut scratch = [0u8; 4096];
    loop {
        if let Some(frame) = try_pop_frame(&mut in_buf, max_hello, 0)? {
            return Ok((conn, in_buf, frame));
        }
        let n = conn.read_tls(stream)?;
        if n == 0 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "eof before hello",
            ));
        }
        conn.process_new_packets()
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("tls: {e}")))?;
        loop {
            match conn.reader().read(&mut scratch) {
                Ok(0) => break,
                Ok(k) => in_buf.extend_from_slice(&scratch[..k]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) => return Err(e),
            }
        }
    }
}

/// Pumps frames full-duplex over a handshaked `ServerConnection`: outgoing frames
/// from `out_rx` are encrypted and sent; incoming frames are decrypted and pushed
/// to `in_tx`.
pub fn pump_server_tls(
    mut stream: TcpStream,
    mut conn: ServerConnection,
    mut in_buf: Vec<u8>,
    out_rx: std::sync::mpsc::Receiver<Frame>,
    in_tx: std::sync::mpsc::Sender<io::Result<Frame>>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(5)));
    let mut scratch = [0u8; 16384];
    loop {
        // Flush any already-buffered plaintext frames first.
        loop {
            match try_pop_frame(&mut in_buf, MAX_HEADER, MAX_FRAME) {
                Ok(Some(frame)) => {
                    if in_tx.send(Ok(frame)).is_err() {
                        return;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = in_tx.send(Err(e));
                    return;
                }
            }
        }
        // Drain all queued outgoing frames.
        loop {
            match out_rx.try_recv() {
                Ok(frame) => {
                    if write_server_frame(&mut conn, &mut stream, &frame).is_err() {
                        return;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }
        // Poll incoming encrypted bytes with a short timeout so outgoing frames
        // are never starved.
        match conn.read_tls(&mut stream) {
            Ok(0) => {
                let _ = in_tx.send(Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "peer closed tls connection",
                )));
                return;
            }
            Ok(_) => {
                if let Err(e) = conn.process_new_packets() {
                    let _ = in_tx.send(Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tls: {e}"),
                    )));
                    return;
                }
                if flush_server_tls(&mut conn, &mut stream).is_err() {
                    return;
                }
                loop {
                    match conn.reader().read(&mut scratch) {
                        Ok(0) => break,
                        Ok(k) => in_buf.extend_from_slice(&scratch[..k]),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(e) => {
                            let _ = in_tx.send(Err(e));
                            return;
                        }
                    }
                }
            }
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(e) => {
                let _ = in_tx.send(Err(e));
                return;
            }
        }
    }
}

/// Pumps frames full-duplex over a handshaked `ClientConnection`.
pub fn pump_client_tls(
    mut stream: TcpStream,
    mut conn: ClientConnection,
    out_rx: std::sync::mpsc::Receiver<Frame>,
    inbox: std::sync::mpsc::Sender<crate::client::Incoming>,
) {
    let _ = stream.set_read_timeout(Some(Duration::from_millis(5)));
    let mut in_buf = Vec::new();
    let mut scratch = [0u8; 16384];
    loop {
        loop {
            match try_pop_frame(&mut in_buf, MAX_HEADER, MAX_FRAME) {
                Ok(Some(frame)) => {
                    if inbox.send(crate::client::Incoming::Frame(frame)).is_err() {
                        return;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = inbox.send(crate::client::Incoming::Closed(e.to_string()));
                    return;
                }
            }
        }
        loop {
            match out_rx.try_recv() {
                Ok(frame) => {
                    if let Err(e) = write_client_frame(&mut conn, &mut stream, &frame) {
                        let _ = inbox.send(crate::client::Incoming::Closed(e.to_string()));
                        return;
                    }
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => return,
            }
        }
        match conn.read_tls(&mut stream) {
            Ok(0) => {
                let _ = inbox.send(crate::client::Incoming::Closed(
                    "relay closed tls connection".into(),
                ));
                return;
            }
            Ok(_) => {
                if let Err(e) = conn.process_new_packets() {
                    let _ = inbox.send(crate::client::Incoming::Closed(format!("tls: {e}")));
                    return;
                }
                if let Err(e) = flush_client_tls(&mut conn, &mut stream) {
                    let _ = inbox.send(crate::client::Incoming::Closed(e.to_string()));
                    return;
                }
                loop {
                    match conn.reader().read(&mut scratch) {
                        Ok(0) => break,
                        Ok(k) => in_buf.extend_from_slice(&scratch[..k]),
                        Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                        Err(e) => {
                            let _ = inbox.send(crate::client::Incoming::Closed(e.to_string()));
                            return;
                        }
                    }
                }
            }
            Err(e)
                if e.kind() == io::ErrorKind::WouldBlock || e.kind() == io::ErrorKind::TimedOut =>
            {
                continue;
            }
            Err(e) => {
                let _ = inbox.send(crate::client::Incoming::Closed(e.to_string()));
                return;
            }
        }
    }
}
