//! Sends Help ▸ Contact's messages to the developers.
//!
//! Through a relay (`services/contact` in this repository), never straight to
//! a mail service: the mail service's key and the address messages reach
//! would have to be inside the application to do that, and anything inside
//! an application someone downloaded is theirs to read. The relay holds both;
//! this knows only the relay's address.
//!
//! # Why curl
//!
//! HTTPS needs a TLS stack, and the Rust ones bring certificate bundles under
//! licences the dependency policy does not allow. `curl` ships with macOS,
//! with Windows 10 and later, and with the Linux desktops Kreate runs on, and
//! it uses the system's own certificates — which is what a user's proxy and
//! their company's certificates are already set up for.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;

/// Where messages go. Set `KREATE_CONTACT_URL` when building to point a build
/// at another deployment of the relay.
pub const ENDPOINT: &str = match option_env!("KREATE_CONTACT_URL") {
    Some(url) => url,
    None => "https://kreate-contact.vercel.app/api/contact",
};

/// One message, as the form has it.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    pub kind: String,
    pub name: String,
    pub email: String,
    pub message: String,
    /// The version and platform line, or `None` to leave them out.
    pub details: Option<(String, String)>,
}

/// The body the relay reads.
pub fn body(m: &Message) -> serde_json::Value {
    let mut v = serde_json::json!({ "kind": m.kind, "message": m.message });
    if !m.name.is_empty() {
        v["name"] = m.name.clone().into();
    }
    if !m.email.is_empty() {
        v["email"] = m.email.clone().into();
    }
    if let Some((version, platform)) = &m.details {
        v["version"] = version.clone().into();
        v["platform"] = platform.clone().into();
    }
    v
}

/// What the relay answered: sent, or why not.
pub fn read_answer(stdout: &str) -> Result<(), String> {
    // `-w` puts the status on a line of its own after the body.
    let (json, status) = stdout
        .trim_end()
        .rsplit_once('\n')
        .unwrap_or(("", stdout.trim()));
    let status: u16 = status.trim().parse().unwrap_or(0);
    let answer: serde_json::Value = serde_json::from_str(json).unwrap_or_default();
    if (200..300).contains(&status) && answer.get("ok").and_then(|v| v.as_bool()) == Some(true) {
        return Ok(());
    }
    Err(answer
        .get("error")
        .and_then(|e| e.as_str())
        .map(str::to_string)
        .unwrap_or_else(|| match status {
            0 => "the contact service could not be reached".to_string(),
            s => format!("the contact service answered {s}"),
        }))
}

/// Messages on their way, and the answers as they come.
pub struct Outbox {
    tx: mpsc::Sender<Result<(), String>>,
    rx: Receiver<Result<(), String>>,
    wake: Arc<dyn Fn() + Send + Sync>,
}

impl Outbox {
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            tx,
            rx,
            wake: Arc::new(wake),
        }
    }

    /// Sends `m` on a thread of its own; the answer arrives through [`Outbox::poll`].
    pub fn send(&self, m: &Message) {
        let (tx, wake) = (self.tx.clone(), Arc::clone(&self.wake));
        let body = body(m).to_string();
        std::thread::spawn(move || {
            let _ = tx.send(post(&body));
            wake();
        });
    }

    /// The answers that have arrived.
    pub fn poll(&self) -> Vec<Result<(), String>> {
        self.rx.try_iter().collect()
    }
}

fn post(body: &str) -> Result<(), String> {
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut child = Command::new(curl)
        .args([
            "--silent",
            "--show-error",
            "--max-time",
            "25",
            "--request",
            "POST",
        ])
        .args(["--header", "content-type: application/json"])
        .arg("--header")
        .arg(format!(
            "x-kreate-client: kreate/{}",
            crate::version::VERSION
        ))
        .args([
            "--data-binary",
            "@-",
            "--write-out",
            "\n%{http_code}",
            ENDPOINT,
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start curl to send the message: {e}"))?;
    if let Some(mut input) = child.stdin.take() {
        input
            .write_all(body.as_bytes())
            .map_err(|e| e.to_string())?;
    }
    let mut out = String::new();
    let mut err = String::new();
    if let Some(mut s) = child.stdout.take() {
        let _ = s.read_to_string(&mut out);
    }
    if let Some(mut s) = child.stderr.take() {
        let _ = s.read_to_string(&mut err);
    }
    let _ = child.wait();
    read_answer(&out).map_err(|why| {
        if err.trim().is_empty() {
            why
        } else {
            format!("{why}: {}", err.trim())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_body_leaves_out_what_was_not_given() {
        let m = Message {
            kind: "bug".into(),
            name: String::new(),
            email: "a@b.kr".into(),
            message: "It crashed".into(),
            details: None,
        };
        assert_eq!(
            body(&m),
            serde_json::json!({ "kind": "bug", "message": "It crashed", "email": "a@b.kr" })
        );
        let with = Message {
            details: Some(("0.2.0 (abc)".into(), "macos aarch64".into())),
            ..m
        };
        assert_eq!(body(&with)["platform"], "macos aarch64");
    }

    #[test]
    fn answers_are_read() {
        assert_eq!(read_answer("{\"ok\":true}\n200"), Ok(()));
        assert_eq!(
            read_answer("{\"ok\":false,\"error\":\"the message is empty\"}\n400"),
            Err("the message is empty".into())
        );
        assert_eq!(
            read_answer("\n000"),
            Err("the contact service could not be reached".into())
        );
        assert_eq!(
            read_answer("<html>oops</html>\n502"),
            Err("the contact service answered 502".into())
        );
    }
}
