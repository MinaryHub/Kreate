//! The chat panel's other end: an assistant's own command-line tool, run
//! headless for each message — Claude Code, the Gemini CLI, or OpenAI's Codex
//! CLI for ChatGPT.
//!
//! Each thing the user says starts the tool with Kreate's own connector as its
//! MCP server, and the turns after the first resume the same session, so the
//! conversation carries on. The assistant reaches the document the way any
//! connector client does — `kreate-mcp`, then the bridge, then the main thread
//! — so everything it does is an operator and an undo step.
//!
//! Using the user's own tools means no API key in the preferences: whatever
//! account each tool is signed in to is the one that answers.
//!
//! # Keeping them to the document
//!
//! Each tool comes with tools of its own — a shell, file edits, the web — and
//! none of them is wanted here. Claude Code is given no built-in tools at all.
//! Gemini runs with its built-in tools excluded and only the connector
//! trusted, so anything else would need an approval nobody is there to give.
//! Codex runs in its read-only sandbox, in an empty folder. Each runs in a
//! folder of Kreate's own (`agents/<tool>` in the settings folder), where its
//! instructions are written for it to read: `GEMINI.md`, `AGENTS.md`.
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

/// What the assistant is told about where it is, on top of the connector's
/// own instructions.
const SYSTEM_PROMPT: &str = "You are talking with the user in the chat panel inside Kreate, \
a 3D modelling application, while they look at the same document. The kreate tools act on \
that document; use them, and nothing else, to look at and change it. Answer in the language \
the user writes in. Keep replies short: the panel is narrow. Say what you changed in a sentence \
or two rather than repeating the script.";

/// Which assistant the panel talks to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Provider {
    #[default]
    Claude,
    Gemini,
    /// ChatGPT, through OpenAI's Codex CLI.
    ChatGpt,
}

impl Provider {
    pub const ALL: [Provider; 3] = [Provider::Claude, Provider::Gemini, Provider::ChatGpt];

    /// The name a person knows it by.
    pub fn name(self) -> &'static str {
        match self {
            Provider::Claude => "Claude",
            Provider::Gemini => "Gemini",
            Provider::ChatGpt => "ChatGPT",
        }
    }

    /// The tool's command, and the variable that can point at it.
    fn command(self) -> (&'static str, &'static str) {
        match self {
            Provider::Claude => ("claude", "KREATE_CLAUDE"),
            Provider::Gemini => ("gemini", "KREATE_GEMINI"),
            Provider::ChatGpt => ("codex", "KREATE_CODEX"),
        }
    }

    /// What to say when the tool is not installed.
    fn not_installed(self) -> String {
        let (name, var) = self.command();
        let how = match self {
            Provider::Claude => "install Claude Code from https://claude.com/claude-code",
            Provider::Gemini => "install the Gemini CLI: npm install -g @google/gemini-cli",
            Provider::ChatGpt => "install the Codex CLI: npm install -g @openai/codex",
        };
        format!(
            "`{name}` was not found. To use {}, {how}, or set {var} to its path.",
            self.name()
        )
    }

    /// A folder of Kreate's own for the tool to run in.
    fn folder(self) -> Option<PathBuf> {
        let id = match self {
            Provider::Claude => return crate::prefs::config_dir().filter(|d| d.is_dir()),
            Provider::Gemini => "gemini",
            Provider::ChatGpt => "codex",
        };
        let dir = crate::prefs::config_dir()?.join("agents").join(id);
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    }
}

/// The prefixes the tools put on the connector's tool names.
const TOOL_PREFIXES: [&str; 4] = ["mcp__kreate__", "mcp_kreate_", "kreate__", "kreate."];

fn short_tool_name(name: &str) -> String {
    TOOL_PREFIXES
        .iter()
        .find_map(|p| name.strip_prefix(p))
        .unwrap_or(name)
        .to_string()
}

/// Something that came out of a turn.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The session to resume for the next turn.
    Session(String),
    /// Something the assistant said.
    Text(String),
    /// More of what the assistant is saying, as it streams.
    TextPart(String),
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

/// Who an assistant's tool is signed in as.
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
    /// The tool is not installed, or could not be asked.
    Unavailable(String),
}

/// The conversation: the session to carry on, and the turn in progress.
pub struct Chat {
    provider: Provider,
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
            provider: Provider::default(),
            session: None,
            turn: None,
            wake: Arc::new(wake),
            side: mpsc::channel(),
            sign_in: None,
        }
    }

    /// Which assistant this is talking to.
    pub fn provider(&self) -> Provider {
        self.provider
    }

    /// Talks to another assistant from now on: the conversation so far is
    /// the old one's and is not carried over.
    pub fn set_provider(&mut self, provider: Provider) {
        if provider == self.provider {
            return;
        }
        self.reset();
        self.cancel_sign_in();
        self.provider = provider;
        self.check_account();
    }

    /// Asks the tool who it is signed in as; the answer arrives as
    /// [`Event::Account`].
    pub fn check_account(&self) {
        let (tx, wake, provider) = (self.side.0.clone(), Arc::clone(&self.wake), self.provider);
        std::thread::spawn(move || {
            let _ = tx.send(Event::Account(ask_account(provider)));
            wake();
        });
    }

    /// Whether a sign-in is waiting on the browser.
    pub fn signing_in(&self) -> bool {
        self.sign_in.is_some()
    }

    /// Starts the tool's sign-in, which opens the browser: `claude auth
    /// login`, `codex login`, or for Gemini a first message that has it sign
    /// in with Google. The page's address arrives as [`Event::SignInUrl`] if
    /// the tool prints one, and the end as [`Event::SignedIn`].
    pub fn sign_in(&mut self) -> Result<(), String> {
        if self.sign_in.is_some() {
            return Ok(());
        }
        let provider = self.provider;
        let tool = find_cli(provider).ok_or_else(|| provider.not_installed())?;
        let mut command = Command::new(tool);
        match provider {
            Provider::Claude => {
                command.args(["auth", "login"]);
            }
            Provider::ChatGpt => {
                command.arg("login");
            }
            Provider::Gemini => {
                // The Gemini CLI signs in on first use; a message that needs
                // no tools is the shortest first use.
                command
                    .args(["-p", "Reply with the single word OK.", "--skip-trust"])
                    .env("GOOGLE_GENAI_USE_GCA", "true");
            }
        }
        if let Some(dir) = provider.folder() {
            command.current_dir(dir);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", provider.name()))?;
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

    /// Whether the assistant is answering.
    pub fn busy(&self) -> bool {
        self.turn.is_some()
    }

    /// Says `text` to the assistant. An error if its tool cannot be started.
    pub fn send(&mut self, text: &str) -> Result<(), String> {
        if self.turn.is_some() {
            return Err(format!("{} is still answering", self.provider.name()));
        }
        let provider = self.provider;
        let tool = find_cli(provider).ok_or_else(|| provider.not_installed())?;
        let connector = find_connector().ok_or(
            "kreate-mcp was not found next to Kreate. Build it with `cargo build -p kreate-mcp`.",
        )?;
        let folder = provider.folder();
        let mut command = Command::new(tool);
        let parse: fn(&str) -> Vec<Event> = match provider {
            Provider::Claude => {
                let config = serde_json::json!({
                    "mcpServers": { "kreate": { "command": connector, "args": [] } }
                });
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
                parse_line
            }
            Provider::Gemini => {
                let dir = folder
                    .as_ref()
                    .ok_or("no settings folder to run Gemini in")?;
                write_gemini_folder(dir, &connector)
                    .map_err(|e| format!("cannot set up Gemini's folder: {e}"))?;
                command
                    .arg("-p")
                    .arg(text)
                    .args(["--output-format", "stream-json", "--skip-trust"])
                    .args(["--allowed-mcp-server-names", "kreate"]);
                // Kreate's folder holds only Kreate's conversation, so the
                // latest session in it is this one.
                if self.session.is_some() {
                    command.args(["--resume", "latest"]);
                }
                // Signed in with Google, unless an API key says otherwise.
                if std::env::var_os("GEMINI_API_KEY").is_none() {
                    command.env("GOOGLE_GENAI_USE_GCA", "true");
                }
                parse_gemini_line
            }
            Provider::ChatGpt => {
                let dir = folder
                    .as_ref()
                    .ok_or("no settings folder to run Codex in")?;
                std::fs::write(dir.join("AGENTS.md"), SYSTEM_PROMPT)
                    .map_err(|e| format!("cannot set up Codex's folder: {e}"))?;
                command
                    .arg("exec")
                    .args(["--json", "--skip-git-repo-check", "--sandbox", "read-only"])
                    .arg("--cd")
                    .arg(dir)
                    .arg("-c")
                    .arg(format!(
                        "mcp_servers.kreate.command={}",
                        toml_string(&connector.to_string_lossy())
                    ))
                    .args(["-c", "approval_policy=\"never\""]);
                if let Some(thread) = &self.session {
                    command.args(["resume", thread]);
                }
                command.arg(text);
                parse_codex_line
            }
        };
        if let Some(dir) = &folder {
            command.current_dir(dir);
        }
        let mut child = command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot start {}: {e}", provider.name()))?;

        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        let (tx, rx) = mpsc::channel();
        let wake = Arc::clone(&self.wake);
        std::thread::spawn(move || read_turn(stdout, stderr, &tx, &*wake, parse));
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
                    Some(
                        said.map(|s| s.trim().to_string())
                            .filter(|s| !s.is_empty())
                            .unwrap_or_else(|| "stopped".into()),
                    )
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
        let Some(turn) = &mut self.turn else {
            return Vec::new();
        };
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

/// Who the provider's tool is signed in as.
fn ask_account(provider: Provider) -> Account {
    let Some(tool) = find_cli(provider) else {
        return Account::Unavailable(provider.not_installed());
    };
    match provider {
        Provider::Claude => {
            // Not logged in exits with 1 and still prints the answer, so the
            // output is read whatever the status.
            match Command::new(tool)
                .args(["auth", "status", "--json"])
                .stdin(Stdio::null())
                .output()
            {
                Ok(out) => {
                    parse_account(&String::from_utf8_lossy(&out.stdout)).unwrap_or_else(|| {
                        Account::Unavailable(
                            String::from_utf8_lossy(&out.stderr).trim().to_string(),
                        )
                    })
                }
                Err(e) => Account::Unavailable(format!("cannot start Claude Code: {e}")),
            }
        }
        Provider::Gemini => gemini_account(),
        Provider::ChatGpt => match Command::new(tool)
            .args(["login", "status"])
            .stdin(Stdio::null())
            .output()
        {
            Ok(out) => {
                let said = format!(
                    "{}{}",
                    String::from_utf8_lossy(&out.stdout),
                    String::from_utf8_lossy(&out.stderr)
                );
                codex_account(&said, read_codex_auth().as_deref())
            }
            Err(e) => Account::Unavailable(format!("cannot start Codex: {e}")),
        },
    }
}

/// Gemini's account: the Google account it signed in with, which the CLI
/// keeps in `google_accounts.json` beside its cached credentials; or an API
/// key in the environment.
fn gemini_account() -> Account {
    if std::env::var_os("GEMINI_API_KEY").is_some() {
        return Account::SignedIn {
            email: "Gemini API key".into(),
            organisation: None,
            plan: None,
        };
    }
    let Some(dir) = gemini_home() else {
        return Account::SignedOut;
    };
    if !dir.join("oauth_creds.json").is_file() {
        return Account::SignedOut;
    }
    let email = std::fs::read_to_string(dir.join("google_accounts.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|v| v.get("active").and_then(Value::as_str).map(str::to_string));
    Account::SignedIn {
        email: email.unwrap_or_else(|| "signed in with Google".into()),
        organisation: None,
        plan: None,
    }
}

/// Where the Gemini CLI keeps its settings and credentials.
fn gemini_home() -> Option<PathBuf> {
    let base = std::env::var_os("GEMINI_CLI_HOME")
        .or_else(|| std::env::var_os("HOME"))
        .or_else(|| std::env::var_os("USERPROFILE"))?;
    Some(Path::new(&base).join(".gemini"))
}

/// Codex's `auth.json`, where the signed-in account's token is.
fn read_codex_auth() -> Option<String> {
    let home = std::env::var_os("CODEX_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .map(|h| Path::new(&h).join(".codex"))
        })?;
    std::fs::read_to_string(home.join("auth.json")).ok()
}

/// Codex's account from `codex login status` and its `auth.json`: signed in
/// or not from the first, and who from the identity token in the second.
pub fn codex_account(status: &str, auth: Option<&str>) -> Account {
    let lower = status.to_lowercase();
    if lower.contains("not logged in") || !lower.contains("logged in") {
        return Account::SignedOut;
    }
    let claims = auth
        .and_then(|a| serde_json::from_str::<Value>(a).ok())
        .and_then(|v| {
            v.pointer("/tokens/id_token")
                .and_then(Value::as_str)
                .map(str::to_string)
        })
        .and_then(|token| jwt_claims(&token));
    let email = claims
        .as_ref()
        .and_then(|c| c.get("email").and_then(Value::as_str).map(str::to_string));
    let plan = claims
        .as_ref()
        .and_then(|c| {
            c.pointer("/https:~1~1api.openai.com~1auth/chatgpt_plan_type")
                .and_then(Value::as_str)
        })
        .map(str::to_string);
    let fallback = if lower.contains("api key") {
        "OpenAI API key"
    } else {
        "signed in with ChatGPT"
    };
    Account::SignedIn {
        email: email.unwrap_or_else(|| fallback.into()),
        organisation: None,
        plan,
    }
}

/// The claims of a JSON Web Token, read without checking its signature: only
/// to show whose it is, never to trust it.
fn jwt_claims(token: &str) -> Option<Value> {
    let payload = token.split('.').nth(1)?;
    serde_json::from_slice(&base64url(payload)?).ok()
}

fn base64url(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'-' | b'+' => Some(62),
        b'_' | b'/' => Some(63),
        _ => None,
    };
    let mut out = Vec::new();
    let (mut acc, mut bits) = (0u32, 0);
    for c in text.bytes().filter(|c| *c != b'=') {
        acc = (acc << 6) | value(c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// A string as TOML writes it, for a `-c key=value` override.
fn toml_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// The Gemini CLI's folder: the connector as its one trusted MCP server, its
/// own tools excluded, and the instructions where it reads them.
fn write_gemini_folder(dir: &Path, connector: &Path) -> std::io::Result<()> {
    let settings = serde_json::json!({
        "mcpServers": { "kreate": { "command": connector, "args": [], "trust": true } },
        "tools": { "exclude": GEMINI_BUILTINS },
    });
    std::fs::create_dir_all(dir.join(".gemini"))?;
    std::fs::write(
        dir.join(".gemini").join("settings.json"),
        serde_json::to_vec_pretty(&settings)?,
    )?;
    std::fs::write(dir.join("GEMINI.md"), SYSTEM_PROMPT)
}

/// The Gemini CLI's own tools, none of which the panel wants.
const GEMINI_BUILTINS: [&str; 12] = [
    "run_shell_command",
    "write_file",
    "replace",
    "read_file",
    "read_many_files",
    "list_directory",
    "glob",
    "search_file_content",
    "grep",
    "web_fetch",
    "google_web_search",
    "save_memory",
];

/// The account in `claude auth status --json`'s answer.
pub fn parse_account(json: &str) -> Option<Account> {
    let v: Value = serde_json::from_str(json).ok()?;
    if v.get("loggedIn").and_then(Value::as_bool) != Some(true) {
        return Some(Account::SignedOut);
    }
    let text = |key: &str| {
        v.get(key)
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
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
    let url: String = line[start..]
        .chars()
        .take_while(|c| !c.is_whitespace())
        .collect();
    Some(url)
}

/// Reads one turn's output to the end, then says it is over.
fn read_turn(
    stdout: impl Read,
    stderr: impl Read + Send + 'static,
    tx: &Sender<Event>,
    wake: &(dyn Fn() + Send + Sync),
    parse: fn(&str) -> Vec<Event>,
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
        for event in parse(&line) {
            done |= matches!(event, Event::Done(_));
            let _ = tx.send(event);
        }
        wake();
    }
    if !done {
        // No result line: the process was stopped, or died before answering.
        let why = stderr.join().unwrap_or_default();
        let why = first_complaint(&why);
        let _ = tx.send(Event::Done(Some(if why.is_empty() {
            "stopped".to_string()
        } else {
            why
        })));
    }
    wake();
}

/// What a tool said on its way down, cut to what a person reads: the first
/// line that is not a warning or a log line, without the stack trace and the
/// timestamps under it.
pub fn first_complaint(stderr: &str) -> String {
    let lines: Vec<&str> = stderr
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let useful = |l: &str| {
        !l.starts_with("Warning")
            && !l.starts_with("at ")
            && !l.starts_with('{')
            && !l.starts_with('}')
    };
    let line = lines
        .iter()
        .find(|l| useful(l) && !l.starts_with(|c: char| c.is_ascii_digit()))
        .or_else(|| lines.iter().find(|l| useful(l)))
        .or(lines.first())
        .copied()
        .unwrap_or("");
    let mut out: String = line.chars().take(400).collect();
    if line.chars().count() > 400 {
        out.push('…');
    }
    out
}

/// The events in one line of the Gemini CLI's `--output-format stream-json`.
pub fn parse_gemini_line(line: &str) -> Vec<Event> {
    let Ok(m) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let text = |key: &str| m.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    match m.get("type").and_then(Value::as_str) {
        Some("init") => vec![Event::Session(text("session_id"))],
        Some("message") if text("role") == "assistant" => {
            let content = text("content");
            if content.is_empty() {
                Vec::new()
            } else if m.get("delta").and_then(Value::as_bool) == Some(true) {
                vec![Event::TextPart(content)]
            } else {
                vec![Event::Text(content.trim().to_string())]
            }
        }
        Some("tool_use") => {
            let name = short_tool_name(&text("tool_name"));
            let detail = tool_detail(&name, m.get("parameters").unwrap_or(&Value::Null));
            vec![Event::Tool { name, detail }]
        }
        Some("tool_result") if text("status") == "error" => {
            let why = m
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_else(|| text("output"));
            vec![Event::ToolError(why)]
        }
        Some("error") if text("severity") != "warning" => vec![Event::ToolError(text("message"))],
        Some("result") => {
            let failed = text("status") == "error";
            vec![Event::Done(failed.then(|| {
                m.pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("Gemini did not answer")
                    .to_string()
            }))]
        }
        _ => Vec::new(),
    }
}

/// The events in one line of `codex exec --json`.
pub fn parse_codex_line(line: &str) -> Vec<Event> {
    let Ok(m) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let item = m.get("item").unwrap_or(&Value::Null);
    let field = |v: &Value, key: &str| v.get(key).and_then(Value::as_str).unwrap_or("").to_string();
    match (
        m.get("type").and_then(Value::as_str),
        field(item, "type").as_str(),
    ) {
        (Some("thread.started"), _) => vec![Event::Session(field(&m, "thread_id"))],
        (Some("item.completed"), "agent_message") => {
            let text = field(item, "text");
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![Event::Text(text.trim().to_string())]
            }
        }
        (Some("item.started"), "mcp_tool_call") => {
            let name = short_tool_name(&field(item, "tool"));
            let detail = tool_detail(&name, item.get("arguments").unwrap_or(&Value::Null));
            vec![Event::Tool { name, detail }]
        }
        (Some("item.completed"), "mcp_tool_call") if field(item, "status") == "failed" => {
            let why = item
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("the tool failed");
            vec![Event::ToolError(why.to_string())]
        }
        // Its own shell, which the read-only sandbox keeps from changing
        // anything; shown, so nothing it does is hidden.
        (Some("item.started"), "command_execution") => {
            vec![Event::Tool {
                name: "shell".into(),
                detail: field(item, "command"),
            }]
        }
        (Some("turn.completed"), _) => vec![Event::Done(None)],
        (Some("turn.failed"), _) => {
            let why = m
                .pointer("/error/message")
                .and_then(Value::as_str)
                .unwrap_or("ChatGPT did not answer");
            vec![Event::Done(Some(why.to_string()))]
        }
        (Some("error"), _) => vec![Event::ToolError(field(&m, "message"))],
        _ => Vec::new(),
    }
}

/// The events in one line of Claude Code's `--output-format stream-json`.
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
                        let text = block
                            .get("text")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .trim();
                        if !text.is_empty() {
                            out.push(Event::Text(text.to_string()));
                        }
                    }
                    Some("tool_use") => {
                        let name = block.get("name").and_then(Value::as_str).unwrap_or("?");
                        let name = short_tool_name(name);
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
                || message
                    .get("subtype")
                    .and_then(Value::as_str)
                    .is_some_and(|s| s != "success");
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

/// A provider's tool: its `KREATE_*` variable, then the search path, then
/// where its installers put it — a window opened from the Finder or a desktop
/// menu does not get the shell's path.
fn find_cli(provider: Provider) -> Option<PathBuf> {
    let (command, var) = provider.command();
    if let Some(path) = std::env::var_os(var) {
        return Some(PathBuf::from(path));
    }
    // npm installs a `.cmd` shim on Windows; Claude's installer an `.exe`.
    let names: Vec<String> = if cfg!(windows) {
        vec![format!("{command}.exe"), format!("{command}.cmd")]
    } else {
        vec![command.to_string()]
    };
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .iter()
        .flat_map(std::env::split_paths)
        .collect();
    if let Some(home) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
        let home = Path::new(&home);
        dirs.extend([
            home.join(".local/bin"),
            home.join(".claude/local"),
            home.join(".npm-global/bin"),
        ]);
        if let Some(appdata) = std::env::var_os("APPDATA") {
            dirs.push(Path::new(&appdata).join("npm"));
        }
    }
    dirs.extend([
        PathBuf::from("/opt/homebrew/bin"),
        PathBuf::from("/usr/local/bin"),
    ]);
    dirs.iter()
        .flat_map(|d| names.iter().map(move |n| d.join(n)))
        .find(|p| p.is_file())
}

/// `kreate-mcp`, which ships next to Kreate.
fn find_connector() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "kreate-mcp.exe"
    } else {
        "kreate-mcp"
    };
    let here = std::env::current_exe().ok()?;
    let beside = here.parent()?.join(name);
    beside.is_file().then_some(beside)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_line_gives_the_session() {
        let events =
            parse_line(r#"{"type":"system","subtype":"init","session_id":"abc","tools":[]}"#);
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
        assert_eq!(
            parse_line(&line),
            vec![Event::ToolError("no object called Nope".into())]
        );
    }

    #[test]
    fn result_ends_the_turn() {
        let ok = parse_line(
            r#"{"type":"result","subtype":"success","is_error":false,"result":"Done.","session_id":"abc"}"#,
        );
        assert_eq!(ok, vec![Event::Session("abc".into()), Event::Done(None)]);
        let failed = parse_line(
            r#"{"type":"result","subtype":"error_max_turns","is_error":true,"session_id":"abc"}"#,
        );
        assert_eq!(
            failed,
            vec![
                Event::Session("abc".into()),
                Event::Done(Some("error_max_turns".into()))
            ]
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
        assert_eq!(
            parse_account(r#"{"loggedIn": false, "authMethod": "none"}"#),
            Some(Account::SignedOut)
        );
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
    fn gemini_streams_its_reply_and_names_its_tools() {
        assert_eq!(
            parse_gemini_line(r#"{"type":"init","session_id":"g1","model":"gemini-2.5-pro"}"#),
            vec![Event::Session("g1".into())]
        );
        assert_eq!(
            parse_gemini_line(
                r#"{"type":"message","role":"assistant","content":"Adding ","delta":true}"#
            ),
            vec![Event::TextPart("Adding ".into())]
        );
        assert!(parse_gemini_line(r#"{"type":"message","role":"user","content":"hi"}"#).is_empty());
        assert_eq!(
            parse_gemini_line(
                r#"{"type":"tool_use","tool_name":"run_script","tool_id":"t","parameters":{"script":"kreate.ops.object.delete()"}}"#
            ),
            vec![Event::Tool {
                name: "run_script".into(),
                detail: "kreate.ops.object.delete()".into()
            }]
        );
        assert_eq!(
            parse_gemini_line(
                r#"{"type":"tool_result","tool_id":"t","status":"error","output":"","error":{"type":"X","message":"no object called Nope"}}"#
            ),
            vec![Event::ToolError("no object called Nope".into())]
        );
        assert_eq!(
            parse_gemini_line(r#"{"type":"result","status":"success","stats":{}}"#),
            vec![Event::Done(None)]
        );
        assert_eq!(
            parse_gemini_line(
                r#"{"type":"result","status":"error","error":{"type":"FatalAuthenticationError","message":"sign in first"}}"#
            ),
            vec![Event::Done(Some("sign in first".into()))]
        );
    }

    #[test]
    fn codex_reports_its_thread_messages_and_tool_calls() {
        assert_eq!(
            parse_codex_line(r#"{"type":"thread.started","thread_id":"th-1"}"#),
            vec![Event::Session("th-1".into())]
        );
        assert_eq!(
            parse_codex_line(
                r#"{"type":"item.started","item":{"id":"i1","type":"mcp_tool_call","server":"kreate","tool":"get_scene","arguments":{},"status":"in_progress"}}"#
            ),
            vec![Event::Tool {
                name: "get_scene".into(),
                detail: String::new()
            }]
        );
        assert_eq!(
            parse_codex_line(
                r#"{"type":"item.completed","item":{"id":"i1","type":"mcp_tool_call","server":"kreate","tool":"run_script","status":"failed","error":{"message":"bad line"}}}"#
            ),
            vec![Event::ToolError("bad line".into())]
        );
        assert_eq!(
            parse_codex_line(
                r#"{"type":"item.completed","item":{"id":"i2","type":"agent_message","text":" Done. "}}"#
            ),
            vec![Event::Text("Done.".into())]
        );
        assert_eq!(
            parse_codex_line(r#"{"type":"turn.completed","usage":{}}"#),
            vec![Event::Done(None)]
        );
        assert_eq!(
            parse_codex_line(r#"{"type":"turn.failed","error":{"message":"usage limit"}}"#),
            vec![Event::Done(Some("usage limit".into()))]
        );
        assert!(parse_codex_line(
            r#"{"type":"item.completed","item":{"type":"reasoning","text":"…"}}"#
        )
        .is_empty());
    }

    #[test]
    fn codex_account_names_whoever_the_token_names() {
        // A token of three parts; only the middle one is read.
        let claims =
            r#"{"email":"a@b.kr","https://api.openai.com/auth":{"chatgpt_plan_type":"plus"}}"#;
        let payload: String = {
            const T: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
            let b = claims.as_bytes();
            let mut out = String::new();
            for chunk in b.chunks(3) {
                let n =
                    chunk.iter().fold(0u32, |a, c| (a << 8) | *c as u32) << (8 * (3 - chunk.len()));
                for i in 0..=chunk.len() {
                    out.push(T[((n >> (18 - 6 * i)) & 63) as usize] as char);
                }
            }
            out
        };
        let auth = format!(r#"{{"tokens":{{"id_token":"h.{payload}.s"}}}}"#);
        assert_eq!(
            codex_account("Logged in using ChatGPT\n", Some(&auth)),
            Account::SignedIn {
                email: "a@b.kr".into(),
                organisation: None,
                plan: Some("plus".into())
            }
        );
        assert_eq!(
            codex_account("Not logged in\n", Some(&auth)),
            Account::SignedOut
        );
    }

    #[test]
    fn tool_names_lose_whatever_prefix_their_client_gave_them() {
        for name in [
            "mcp__kreate__run_script",
            "mcp_kreate_run_script",
            "kreate__run_script",
            "run_script",
        ] {
            assert_eq!(short_tool_name(name), "run_script");
        }
    }

    #[test]
    fn a_dying_tool_is_summed_up_in_its_first_complaint() {
        let gemini = "Warning: 256-color support not detected.\nError authenticating: FatalAuthenticationError: Manual authorization is required.\n    at initOauthClient (file:///x.js:1:2)\n";
        assert_eq!(
            first_complaint(gemini),
            "Error authenticating: FatalAuthenticationError: Manual authorization is required."
        );
        let codex = "2026-10-05T11:11:15Z ERROR codex_api: 401 Unauthorized\n";
        assert_eq!(
            first_complaint(codex),
            "2026-10-05T11:11:15Z ERROR codex_api: 401 Unauthorized"
        );
        assert_eq!(first_complaint(""), "");
    }

    #[test]
    fn noise_is_ignored() {
        assert!(parse_line("not json").is_empty());
        assert!(parse_line(r#"{"type":"stream_event"}"#).is_empty());
    }
}
