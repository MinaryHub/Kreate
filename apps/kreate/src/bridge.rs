//! The connector's way in: a local socket a running Kreate answers on.
//!
//! `kreate-mcp` is what Claude talks to, over MCP on its standard input. It
//! turns each tool call into a request here and hands back what comes out —
//! so Claude acts on the document open on screen, through the same operators
//! the menus use, and every edit lands in the undo history like any other.
//!
//! # The wire
//!
//! One JSON object per line, each way, on a connection that can carry any
//! number of them:
//!
//! ```text
//! → {"token": "...", "method": "scene", "params": {}}
//! ← {"ok": {...}}            or            ← {"error": "why"}
//! ```
//!
//! # Who may connect
//!
//! Only this machine: the socket is bound to the loopback address. And only
//! what can read this user's settings folder: the port is chosen by the
//! system, and it and a random token are written to `bridge.json` there for
//! the connector to read. A request without the token is refused — another
//! user's process on a shared machine can reach the port, not the file.
//!
//! # Threads
//!
//! The document lives on the main thread, and only it may touch it. A
//! connection's thread parses a request, sends it across a channel, wakes the
//! event loop and waits for the answer; the main thread serves whatever is
//! queued when it wakes. A request that is never answered times out rather
//! than holding a connection forever.

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

/// How long a connection waits for the main thread. Long enough for a capture
/// that has to settle the picture first; short enough that a hung window does
/// not hang the connector with it.
const ANSWER_TIMEOUT: Duration = Duration::from_secs(120);

/// One request, waiting on the main thread.
pub struct Request {
    pub method: String,
    pub params: serde_json::Value,
    reply: Sender<Result<serde_json::Value, String>>,
}

impl Request {
    /// Sends the answer back to the connection that asked.
    pub fn answer(self, result: Result<serde_json::Value, String>) {
        // The connection may have given up and gone; nothing to do then.
        let _ = self.reply.send(result);
    }
}

/// The open socket, as the main thread holds it.
pub struct Bridge {
    requests: Receiver<Request>,
    /// Where the port and token were written, removed when the bridge goes.
    info: PathBuf,
    pub port: u16,
}

impl Bridge {
    /// The requests that have arrived, oldest first. Never blocks.
    pub fn pending(&self) -> Vec<Request> {
        self.requests.try_iter().collect()
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        // A file naming a port nobody answers on would send the connector to
        // whatever opened that port next.
        let _ = std::fs::remove_file(&self.info);
    }
}

/// Where the connector finds the port and the token.
pub fn info_path() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|dir| dir.join("bridge.json"))
}

/// A token nobody can guess: two words from the standard library's per-process
/// random hasher keys, which are drawn from the operating system.
fn new_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    let word = || {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        );
        h.write_u32(std::process::id());
        h.finish()
    };
    format!("{:016x}{:016x}", word(), word())
}

/// Opens the socket, writes `info`, and starts accepting.
///
/// `wake` is called whenever a request is waiting, from a connection's thread:
/// the event loop sleeps until something happens, and a request is something.
pub fn start(info: &Path, wake: impl Fn() + Send + Sync + 'static) -> std::io::Result<Bridge> {
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let token = new_token();
    let record = serde_json::json!({ "port": port, "token": token, "pid": std::process::id() });
    if let Some(parent) = info.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(info, record.to_string())?;

    let (tx, rx) = mpsc::channel();
    let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(wake);
    std::thread::Builder::new().name("bridge".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let (tx, wake, token) = (tx.clone(), wake.clone(), token.clone());
            let _ = std::thread::Builder::new()
                .name("bridge connection".into())
                .spawn(move || serve(stream, &token, &tx, &*wake));
        }
    })?;
    Ok(Bridge { requests: rx, info: info.to_path_buf(), port })
}

/// One connection, request after request until it closes.
fn serve(stream: TcpStream, token: &str, tx: &Sender<Request>, wake: &(dyn Fn() + Send + Sync)) {
    let Ok(mut out) = stream.try_clone() else { return };
    let reader = BufReader::new(stream);
    for line in reader.lines() {
        let Ok(line) = line else { return };
        if line.trim().is_empty() {
            continue;
        }
        let answer = answer_line(&line, token, tx, wake);
        let text = match answer {
            Ok(value) => serde_json::json!({ "ok": value }),
            Err(why) => serde_json::json!({ "error": why }),
        };
        if writeln!(out, "{text}").and_then(|_| out.flush()).is_err() {
            return;
        }
    }
}

fn answer_line(
    line: &str,
    token: &str,
    tx: &Sender<Request>,
    wake: &(dyn Fn() + Send + Sync),
) -> Result<serde_json::Value, String> {
    let request: serde_json::Value =
        serde_json::from_str(line).map_err(|e| format!("not JSON: {e}"))?;
    if request.get("token").and_then(|t| t.as_str()) != Some(token) {
        return Err("wrong or missing token".into());
    }
    let method = request
        .get("method")
        .and_then(|m| m.as_str())
        .ok_or("no method")?
        .to_string();
    let params = request.get("params").cloned().unwrap_or(serde_json::Value::Null);
    let (reply, answer) = mpsc::channel();
    tx.send(Request { method, params, reply })
        .map_err(|_| "Kreate is shutting down".to_string())?;
    wake();
    answer
        .recv_timeout(ANSWER_TIMEOUT)
        .map_err(|_| "Kreate did not answer in time".to_string())?
}

/// Standard base64, padded: how a PNG travels inside JSON.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = (chunk[0] as u32) << 16
            | (*chunk.get(1).unwrap_or(&0) as u32) << 8
            | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_standard_vectors() {
        // RFC 4648's own test vectors.
        for (plain, encoded) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(plain.as_bytes()), encoded);
        }
    }

    fn ask(port: u16, line: &str) -> serde_json::Value {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect");
        writeln!(stream, "{line}").expect("write");
        let mut answer = String::new();
        BufReader::new(stream).read_line(&mut answer).expect("read");
        serde_json::from_str(&answer).expect("an answer is JSON")
    }

    #[test]
    fn a_request_with_the_token_reaches_the_main_thread_and_its_answer_comes_back() {
        let dir = std::env::temp_dir().join(format!("kreate-bridge-{}", std::process::id()));
        let info = dir.join("bridge.json");
        let bridge = start(&info, || {}).expect("the socket opens");
        let record: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&info).expect("info written")).expect("json");
        assert_eq!(record["port"].as_u64(), Some(bridge.port as u64));
        let token = record["token"].as_str().expect("a token").to_string();
        assert_eq!(token.len(), 32);

        // The main thread, played by this one: answer whatever arrives.
        let port = bridge.port;
        let asker = std::thread::spawn(move || {
            ask(port, &format!(r#"{{"token":"{token}","method":"echo","params":{{"x":1}}}}"#))
        });
        let request = loop {
            if let Some(r) = bridge.pending().pop() {
                break r;
            }
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(request.method, "echo");
        let params = request.params.clone();
        request.answer(Ok(params));
        assert_eq!(asker.join().expect("asker")["ok"]["x"], 1);

        drop(bridge);
        assert!(!info.exists(), "the info file outlived the bridge");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_request_without_the_token_is_refused_and_never_reaches_the_document() {
        let dir = std::env::temp_dir().join(format!("kreate-bridge-refuse-{}", std::process::id()));
        let bridge = start(&dir.join("bridge.json"), || {}).expect("the socket opens");
        let answer = ask(bridge.port, r#"{"token":"guess","method":"scene"}"#);
        assert!(answer["error"].as_str().is_some_and(|e| e.contains("token")));
        assert!(bridge.pending().is_empty());
        drop(bridge);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
