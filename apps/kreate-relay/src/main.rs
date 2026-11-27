//! A standalone relay for shared editing, for a machine that is always on.
//!
//! ```text
//! kreate-relay [--port 7878] [--key <session key>] [--max 8]
//! ```
//!
//! The key can also come from `KREATE_RELAY_KEY`, which keeps it out of the
//! process list.

use std::path::PathBuf;

use kreate_collab::{Relay, RelayConfig, DEFAULT_PORT};

fn main() {
    let mut port = DEFAULT_PORT;
    let mut config = RelayConfig {
        key: std::env::var("KREATE_RELAY_KEY").unwrap_or_default(),
        viewer_key: std::env::var("KREATE_RELAY_VIEWER_KEY")
            .ok()
            .filter(|s| !s.is_empty()),
        ..Default::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().and_then(|p| p.parse().ok()).unwrap_or(port),
            "--key" => config.key = args.next().unwrap_or_default(),
            "--viewer-key" => {
                config.viewer_key = args.next().filter(|s| !s.is_empty());
            }
            "--state-dir" => {
                config.state_dir = args.next().map(PathBuf::from);
            }
            "--tls-cert" => {
                config.tls_cert = args.next().map(PathBuf::from);
            }
            "--tls-key" => {
                config.tls_key = args.next().map(PathBuf::from);
            }
            "--max" => {
                config.max_peers = args
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(config.max_peers)
            }
            "-h" | "--help" => {
                println!(
                    "kreate-relay [--port {DEFAULT_PORT}] [--key <session key>] \
                     [--viewer-key <read-only key>] [--state-dir <dir>] \
                     [--tls-cert <cert.pem> --tls-key <key.pem>] [--max 8]"
                );
                return;
            }
            other => {
                eprintln!("unknown argument {other:?}; see --help");
                std::process::exit(2);
            }
        }
    }
    let tls_enabled = config.tls_cert.is_some() || config.tls_key.is_some();
    let relay = match Relay::start(&format!("0.0.0.0:{port}"), config.clone()) {
        Ok(relay) => relay,
        Err(e) => {
            eprintln!("could not start relay on port {port}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "kreate-relay listening on {}{}{}{}",
        if tls_enabled { "tls://" } else { "" },
        relay.addr(),
        if config.key.is_empty() {
            " (no key: anyone on the network can join)"
        } else {
            ""
        },
        if config.state_dir.is_some() {
            " [persistent]"
        } else {
            ""
        }
    );
    loop {
        std::thread::park();
    }
}
