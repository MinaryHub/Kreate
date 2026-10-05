//! The Claude panel's other end: Claude Code, run headless for each message.
//!
//! Each thing the user says starts `claude -p` with Kreate's own connector as
//! its only MCP server, and the turns after the first resume the same session,
//! so the conversation carries on. Claude reaches the document the way any
//! connector client does — `kreate-mcp`, then the bridge, then the main thread
//! — so everything it does is an operator and an undo step.
//!
//! Using the user's Claude Code means no API key in the preferences: whatever
//! account `claude` is signed in to is the one that answers.
//!
//! # The account
//!
//! Which account that is comes from `claude auth status`, asked when the
//! panel starts and again after a sign-in. Signing in is `claude auth login`:
//! it opens the browser, prints the page's address for when it could not, and
//! reads a code from its input if the page hands one out — so the panel shows
//! the address and a box for the code.
//!
//! # Threads
//!
//! The child's output is read on a thread of its own, turned into [`Event`]s
//! and sent across a channel; the thread wakes the event loop for each one, and
//! the main thread drains them with [`Chat::poll`].

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;

use serde_json::Value;

/// What Claude is told about where it is, on top of the connector's own
/// instructions.
const SYSTEM_PROMPT: &str = "You are talking with the user in the Claude panel inside Kreate, \
a 3D modelling application, while they look at the same document. The kreate tools act on \
that document. Answer in the language the user writes in. Keep replies short: the panel is \
narrow. Say what you changed in a sentence or two rather than repeating the script.";

/// Prefix Claude Code gives the connector's tools.
const TOOL_PREFIX: &str = "mcp__kreate__";

/// Something that came out of a turn.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The session to resume for the next turn.
    Session(String),
    /// Something Claude said.
    Text(String),
    /// A tool Claude called: its short name, and a line about its input.
    Tool { name: String, detail: String },
    /// A tool that failed, and why.
    ToolError(String),
    /// The turn is over; `Some` if it ended in an error.
    Done(Option<String>),
    /// What `claude auth status` said.
    Account(Account),
    /// The sign-in page's address, from `claude auth login`.
    SignInUrl(String),
    /// Sign-in finished; `Some` if it failed. The account is asked again.
    SignedIn(Option<String>),
}

/// Who Claude Code is signed in as.
#[derive(Clone, Debug, PartialEq)]
pub enum Account {
    /// Not asked yet, or being asked.
    Checking,
    SignedIn {
        email: String,
        /// The organisation, when the account belongs to one.
        organisation: Option<String>,
        /// `pro`, `max`, `team`, … or the way it signs in when there is no
        /// subscription (an API key).
        plan: Option<String>,
    },
    SignedOut,
    /// Claude Code is not installed, or could not be asked.
    Unavailable(String),
}

/// The conversation: the session to carry on, and the turn in progress.
pub struct Chat {
    session: Option<String>,
    turn: Option<Turn>,
    wake: Arc<dyn Fn() + Send + Sync>,
    /// Account questions and sign-ins, which outlive any one turn.
    side: (Sender<Event>, Receiver<Event>),
    sign_in: Option<SignIn>,
}

/// A `claude auth login` in progress.
struct SignIn {
    child: Child,
    input: Option<ChildStdin>,
}

struct Turn {
    child: Child,
    events: Receiver<Event>,
}

impl Chat {
    /// `wake` is called from the reading thread whenever an event is waiting.
    pub fn new(wake: impl Fn() + Send + Sync + 'static) -> Self {
        Self {
            session: None,
            turn: None,
            wake: Arc::new(wake),
            side: mpsc::channel(),
            sign_in: None,
        }
    }

    /// Asks Claude Code who it is signed in as; the answer arrives as
    /// [`Event::Account`].
    pub fn check_account(&self) {
        let (tx, wake) = (self.side.0.clone(), Arc::clone(&self.wake));
        std::thread::spawn(move || {
            let _ = tx.send(Event::Account(ask_account()));
            wake();
        });
    }

    /// Whether a sign-in is waiting on the browser.
    pub fn signing_in(&self) -> bool {
        self.sign_in.is_some()
    }

    /// Starts `claude auth login`, which opens the browser. Its address
    /// arrives as [`Event::SignInUrl`] and its end as [`Event::SignedIn`].
    pub fn sign_in(&mut self) -> Result<(), String> {
        if self.sign_in.is_some() {
            return Ok(());
        }
        let claude = find_claude().ok_or(NOT_INSTALLED)?;
        let mut command = Command::new(claude);
        command.args(["auth", "login"]);
        if let Some(dir) = crate::prefs::config_dir().filter(|d| d.is_dir()) {
            command.current_dir(dir);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start Claude Code: {e}"))?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        let (tx, wake) = (self.side.0.clone(), Arc::clone(&self.wake));
        std::thread::spawn(move || {
            // Both streams: where the address is printed is the CLI's business.
            let (tx2, wake2) = (tx.clone(), Arc::clone(&wake));
            let errors = std::thread::spawn(move || scan_sign_in(stderr, &tx2, &*wake2));
            let mut said = scan_sign_in(stdout, &tx, &*wake);
            said.push_str(&errors.join().unwrap_or_default());
            let _ = tx.send(Event::SignedIn(Some(said)));
            wake();
        });
        self.sign_in = Some(SignIn { child, input });
        Ok(())
    }

    /// Hands the sign-in the code the browser showed.
    pub fn sign_in_code(&mut self, code: &str) {
        if let Some(input) = self.sign_in.as_mut().and_then(|s| s.input.as_mut()) {
            let _ = writeln!(input, "{}", code.trim());
            let _ = input.flush();
        }
    }

    /// Gives up on a sign-in.
    pub fn cancel_sign_in(&mut self) {
        if let Some(sign_in) = &mut self.sign_in {
            let _ = sign_in.child.kill();
        }
    }

    /// Whether Claude is answering.
    pub fn busy(&self) -> bool {
        self.turn.is_some()
    }

    /// Says `text` to Claude. An error if Claude Code cannot be started.
    pub fn send(&mut self, text: &str) -> Result<(), String> {
        if self.turn.is_some() {
            return Err("Claude is still answering".into());
        }
        let claude = find_claude().ok_or(NOT_INSTALLED)?;
        let connector = find_connector().ok_or(
            "kreate-mcp was not found next to Kreate. Build it with `cargo build -p kreate-mcp`.",
        )?;
        let config = serde_json::json!({
            "mcpServers": { "kreate": { "command": connector, "args": [] } }
        });

        let mut command = Command::new(claude);
        command
            .arg("-p")
            .arg(text)
            .args(["--output-format", "stream-json", "--verbose"])
            .arg("--mcp-config")
            .arg(config.to_string())
            .arg("--strict-mcp-config")
            // Only the document: no shell, no files, no web.
            .args(["--tools", ""])
            .args(["--allowedTools", "mcp__kreate"])
            .arg("--append-system-prompt")
            .arg(SYSTEM_PROMPT);
        if let Some(session) = &self.session {
            command.args(["--resume", session]);
        }
        // Somewhere with no project in it, so no project's CLAUDE.md is read.
        if let Some(dir) = crate::prefs::config_dir().filter(|d| d.is_dir()) {
            command.current_dir(dir);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start Claude Code: {e}"))?;

        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        let (tx, rx) = mpsc::channel();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || read_turn(stdout, stderr, &tx, &*wake));
        self.turn = Some(Turn { child, events: rx });
        Ok(())
    }

    /// Stops the answer in progress. What was already done stays done.
    pub fn stop(&mut self) {
        if let Some(turn) = &mut self.turn {
            let _ = turn.child.kill();
        }
    }

    /// Forgets the conversation; the next message starts another.
    pub fn reset(&mut self) {
        self.stop();
        self.session = None;
    }

    /// Everything that has happened since the last call.
    pub fn poll(&mut self) -> Vec<Event> {
        let mut out: Vec<Event> = Vec::new();
        while let Ok(event) = self.side.1.try_recv() {
            if let Event::SignedIn(said) = event {
                // The process's word on how it went is its exit status; what it
                // printed is the reason, when it failed.
                let ok = self
                    .sign_in
                    .take()
                    .and_then(|mut s| s.child.wait().ok())
                    .is_some_and(|status| status.success());
                out.push(Event::SignedIn(if ok {
                    None
                } else {
                    Some(said.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or_else(|| "stopped".into()))
                }));
                self.check_account();
                continue;
            }
            out.push(event);
        }
        out.extend(self.poll_turn());
        out
    }

    fn poll_turn(&mut self) -> Vec<Event> {
        let Some(turn) = &mut self.turn else { return Vec::new() };
        let mut out = Vec::new();
        let mut finished = false;
        loop {
            match turn.events.try_recv() {
                Ok(event) => {
                    if let Event::Session(id) = &event {
                        self.session = Some(id.clone());
                    }
                    finished |= matches!(event, Event::Done(_));
                    out.push(event);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    finished = true;
                    break;
                }
            }
        }
        if finished {
            let _ = turn.child.wait();
            self.turn = None;
        }
        out
    }
}

impl Drop for Chat {
    fn drop(&mut self) {
        self.stop();
        self.cancel_sign_in();
    }
}

const NOT_INSTALLED: &str = "Claude Code (`claude`) was not found. Install it from https://claude.com/claude-code, \
     or set KREATE_CLAUDE to its path.";

/// `claude auth status`, read.
fn ask_account() -> Account {
    let Some(claude) = find_claude() else {
        return Account::Unavailable(NOT_INSTALLED.into());
    };
    // Not logged in exits with 1 and still prints the answer, so the output is
    // read whatever the status.
    match Command::new(claude).args(["auth", "status", "--json"]).stdin(Stdio::null()).output() {
        Ok(out) => parse_account(&String::from_utf8_lossy(&out.stdout))
            .unwrap_or_else(|| Account::Unavailable(String::from_utf8_lossy(&out.stderr).trim().to_string())),
        Err(e) => Account::Unavailable(format!("cannot start Claude Code: {e}")),
    }
}

/// The account in `claude auth status --json`'s answer.
pub fn parse_account(json: &str) -> Option<Account> {
    let v: Value = serde_json::from_str(json).ok()?;
    if v.get("loggedIn").and_then(Value::as_bool) != Some(true) {
        return Some(Account::SignedOut);
    }
    let text = |key: &str| v.get(key).and_then(Value::as_str).filter(|s| !s.is_empty()).map(str::to_string);
    Some(Account::SignedIn {
        email: text("email").unwrap_or_else(|| "signed in".into()),
        organisation: text("orgName"),
        plan: text("subscriptionType").or_else(|| text("authMethod")),
    })
}

/// Reads a sign-in's output, sending the page's address when it shows, and
/// returns everything it said.
fn scan_sign_in(stream: impl Read, tx: &Sender<Event>, wake: &(dyn Fn() + Send + Sync)) -> String {
    let mut said = String::new();
    let mut sent = false;
    // By bytes rather than lines: the prompt for a code ends without a newline.
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    while reader.read_until(b'\n', &mut line).is_ok_and(|n| n > 0) {
        let text = String::from_utf8_lossy(&line).to_string();
        if !sent {
            if let Some(url) = sign_in_url(&text) {
                let _ = tx.send(Event::SignInUrl(url));
                wake();
                sent = true;
            }
        }
        said.push_str(&text);
        line.clear();
    }
    said
}

/// The first web address in a line of output.
pub fn sign_in_url(line: &str) -> Option<String> {
    let start = line.find("https://")?;
    let url: String = line[start..].chars().take_while(|c| !c.is_whitespace()).collect();
    Some(url)
}

/// Reads one turn's output to the end, then says it is over.
fn read_turn(
    stdout: impl Read,
    stderr: impl Read + Send + 'static,
    tx: &Sender<Event>,
    wake: &(dyn Fn() + Send + Sync),
) {
    // Read on its own thread so a chatty stderr cannot stall stdout.
    let stderr = std::thread::spawn(move || {
        let mut text = String::new();
        let _ = BufReader::new(stderr).read_to_string(&mut text);
        text
    });
    let mut done = false;
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        for event in parse_line(&line) {
            done |= matches!(event, Event::Done(_));
            let _ = tx.send(event);
        }
        wake();
    }
    if !done {
        // No result line: the process was stopped, or died before answering.
        let why = stderr.join().unwrap_or_default();
        let why = why.trim();
        let _ = tx.send(Event::Done(Some(if why.is_empty() {
            "stopped".to_string()
        } else {
            why.to_string()
        })));
    }
    wake();
}

/// The events in one line of `--output-format stream-json`.
pub fn parse_line(line: &str) -> Vec<Event> {
    let Ok(message) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    match message.get("type").and_then(Value::as_str) {
        Some("system") => {
            if let Some(id) = message.get("session_id").and_then(Value::as_str) {
                out.push(Event::Session(id.to_string()));
            }
        }
        Some("assistant") => {
            for block in content(&message) {
                match block.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        let text = block.get("text").and_then(Value::as_str).unwrap_or("").trim();
                        if !text.is_empty() {
                            out.push(Event::Text(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                        let name = name.strip_prefix(TOOL_PREFIX).unwrap_or(name).to_string();
                        let detail = tool_detail(&name, block.get("input").unwrap_or(&Value::Null));
                        out.push(Event::Tool { name, detail });
                    }
                    _ => {}
                }
            }
        }
        Some("user") => {
            for block in content(&message) {
                let failed = block.get("type").and_then(Value::as_str) == Some("tool_result")
                    && block.get("is_error").and_then(Value::as_bool) == Some(true);
                if failed {
                    out.push(Event::ToolError(result_text(block.get("content"))));
                }
            }
        }
        Some("result") => {
            if let Some(id) = message.get("session_id").and_then(Value::as_str) {
                out.push(Event::Session(id.to_string()));
            }
            let failed = message.get("is_error").and_then(Value::as_bool) == Some(true)
                || message.get("subtype").and_then(Value::as_str).is_some_and(|s| s != "success");
            out.push(Event::Done(failed.then(|| {
                message
                    .get("result")
                    .and_then(Value::as_str)
                    .or_else(|| message.get("subtype").and_then(Value::as_str))
                    .unwrap_or("Claude did not answer")
                    .to_string()
            })));
        }
        _ => {}
    }
    out
}

fn content(message: &Value) -> &[Value] {
    message
        .pointer("/message/content")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

/// A line about a tool's input: the script for `run_script`, otherwise the
/// arguments as they were given.
fn tool_detail(name: &str, input: &Value) -> String {
    if name == "run_script" {
        if let Some(script) = input.get("script").and_then(Value::as_str) {
            return script.trim().to_string();
        }
    }
    match input.as_object() {
        Some(fields) if !fields.is_empty() => input.to_string(),
        _ => String::new(),
    }
}

/// A tool result's text, whether it came as a string or as content blocks.
fn result_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => blocks
            .iter()
            .filter_map(|b| b.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Claude Code: `KREATE_CLAUDE`, then the search path, then where its
/// installers put it — a window opened from the Finder or a desktop menu does
/// not get the shell's path.
fn find_claude() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("KREATE_CLAUDE") {
        return Some(PathBuf::from(path));
    }
    let name = if cfg!(windows) { "claude.exe" } else { "claude" };
    if let Some(found) = std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
    {
        return Some(found);
    }
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?;
    let home = Path::new(&home);
    [
        home.join(".local/bin").join(name),
        home.join(".claude/local").join(name),
        PathBuf::from("/opt/homebrew/bin").join(name),
        PathBuf::from("/usr/local/bin").join(name),
    ]
    .into_iter()
    .find(|p| p.is_file())
}

/// `kreate-mcp`, which ships next to Kreate.
fn find_connector() -> Option<PathBuf> {
    let name = if cfg!(windows) { "kreate-mcp.exe" } else { "kreate-mcp" };
    let here = std::env::current_exe().ok()?;
    let beside = here.parent()?.join(name);
    beside.is_file().then_some(beside)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_line_gives_the_session() {
        let events = parse_line(r#"{"type":"system","subtype":"init","session_id":"abc","tools":[]}"#);
        assert_eq!(events, vec![Event::Session("abc".into())]);
    }

    #[test]
    fn assistant_text_and_tool_calls() {
        let line = r#"{"type":"assistant","message":{"content":[
            {"type":"text","text":"  Adding a sphere.  "},
            {"type":"tool_use","id":"t1","name":"mcp__kreate__run_script",
             "input":{"script":"kreate.ops.object.add_primitive(shape=\"uv_sphere\")\n"}},
            {"type":"tool_use","id":"t2","name":"mcp__kreate__get_scene","input":{}}
        ]}}"#
            .replace('\n', "");
        assert_eq!(
            parse_line(&line),
            vec![
                Event::Text("Adding a sphere.".into()),
                Event::Tool {
                    name: "run_script".into(),
                    detail: "kreate.ops.object.add_primitive(shape=\"uv_sphere\")".into(),
                },
                Event::Tool {
                    name: "get_scene".into(),
                    detail: String::new(),
                },
            ]
        );
    }

    #[test]
    fn failed_tool_results_are_reported() {
        let line = r#"{"type":"user","message":{"content":[
            {"type":"tool_result","tool_use_id":"t1","is_error":true,"content":[{"type":"text","text":"no object called Nope"}]},
            {"type":"tool_result","tool_use_id":"t2","content":"fine"}
        ]}}"#
            .replace('\n', "");
        assert_eq!(parse_line(&line), vec![Event::ToolError("no object called Nope".into())]);
    }

    #[test]
    fn result_ends_the_turn() {
        let ok = parse_line(r#"{"type":"result","subtype":"success","is_error":false,"result":"Done.","session_id":"abc"}"#);
        assert_eq!(ok, vec![Event::Session("abc".into()), Event::Done(None)]);
        let failed = parse_line(r#"{"type":"result","subtype":"error_max_turns","is_error":true,"session_id":"abc"}"#);
        assert_eq!(
            failed,
            vec![Event::Session("abc".into()), Event::Done(Some("error_max_turns".into()))]
        );
    }

    #[test]
    fn the_account_is_read_from_auth_status() {
        let signed_in = r#"{"loggedIn": true, "authMethod": "claude.ai", "email": "a@b.kr",
            "orgName": "Studio", "subscriptionType": "team"}"#;
        assert_eq!(
            parse_account(signed_in),
            Some(Account::SignedIn {
                email: "a@b.kr".into(),
                organisation: Some("Studio".into()),
                plan: Some("team".into()),
            })
        );
        assert_eq!(parse_account(r#"{"loggedIn": false, "authMethod": "none"}"#), Some(Account::SignedOut));
        assert_eq!(parse_account("Not logged in."), None);
    }

    #[test]
    fn the_sign_in_address_is_found_in_its_line() {
        assert_eq!(
            sign_in_url("If the browser didn't open, visit: https://claude.com/cai/oauth/authorize?code=true&x=1\n"),
            Some("https://claude.com/cai/oauth/authorize?code=true&x=1".into())
        );
        assert_eq!(sign_in_url("Opening browser to sign in…"), None);
    }

    #[test]
    fn noise_is_ignored() {
        assert!(parse_line("not json").is_empty());
        assert!(parse_line(r#"{"type":"stream_event"}"#).is_empty());
    }
}
