//! A standalone relay for shared editing, for a machine that is always on.
//!
//! ```text
//! kreate-relay [--port 7878] [--key <session key>] [--max 8]
//! ```
//!
//! The key can also come from `KREATE_RELAY_KEY`, which keeps it out of the
//! process list.

use kreate_collab::{Relay, RelayConfig, DEFAULT_PORT};

fn main() {
    let mut port = DEFAULT_PORT;
    let mut config = RelayConfig {
        key: std::env::var("KREATE_RELAY_KEY").unwrap_or_default(),
        ..Default::default()
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" => port = args.next().and_then(|p| p.parse().ok()).unwrap_or(port),
            "--key" => config.key = args.next().unwrap_or_default(),
            "--max" => {
                config.max_peers = args
                    .next()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(config.max_peers)
            }
            "-h" | "--help" => {
                println!("kreate-relay [--port {DEFAULT_PORT}] [--key <session key>] [--max 8]");
                return;
            }
            other => {
                eprintln!("unknown argument {other:?}; see --help");
                std::process::exit(2);
            }
        }
    }
    let relay = match Relay::start(&format!("0.0.0.0:{port}"), config.clone()) {
        Ok(relay) => relay,
        Err(e) => {
            eprintln!("could not listen on port {port}: {e}");
            std::process::exit(1);
        }
    };
    println!(
        "kreate-relay listening on {}{}",
        relay.addr(),
        if config.key.is_empty() {
            " (no key: anyone on the network can join)"
        } else {
            ""
        }
    );
    loop {
        std::thread::park();
    }
}
