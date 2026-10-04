//! `kreate-mcp`: an MCP server on standard input and output, for Claude.
//!
//! Started by the client (Claude Code or Claude Desktop) and talking to it one
//! JSON message per line. Nothing but protocol may go to standard output, so
//! anything said for a person goes to standard error. See the library for the
//! protocol and the way to Kreate.

use std::io::{BufRead, Write};

fn main() {
    let mut app = kreate_mcp::Socket::new();
    let stdin = std::io::stdin();
    let mut stdout = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(message) => kreate_mcp::handle(&message, &mut app),
            Err(e) => Some(serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32700, "message": format!("parse error: {e}") }
            })),
        };
        if let Some(reply) = reply {
            if writeln!(stdout, "{reply}").and_then(|_| stdout.flush()).is_err() {
                break;
            }
        }
    }
}
