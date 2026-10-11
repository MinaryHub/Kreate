//! Ollama: a model on this computer, with no account and nothing sent away.
//!
//! The other assistants are programs of their own that run the conversation
//! and call Kreate's tools through `kreate-mcp`. Ollama is only a model behind
//! an HTTP endpoint, so here Kreate runs the conversation itself: it starts
//! `kreate-mcp` as the other assistants do, asks it for its tools, hands
//! those to the model with every message, and when the model asks for one,
//! calls it and hands back what came out — round after round until the model
//! answers in words.
//!
//! The endpoint is `OLLAMA_HOST`, as for Ollama's own command, or
//! `http://localhost:11434`. Requests go through `curl`, as every other
//! request in Kreate does.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use crate::claude::{quiet_command, Event};

/// The most rounds of tool calls one message may take; after them the model
/// is asked to answer with what it has. A small model can ask for the same
/// thing forever.
const MAX_ROUNDS: usize = 16;

/// How many tokens of conversation the model is given. Ollama's own default
/// is a few thousand — less than the tools' descriptions and one look at the
/// scene, and a model that has lost the question asks the tools again and
/// again.
const CONTEXT: u32 = 32_768;

/// A model Ollama has, and what it can do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Model {
    pub name: String,
    /// It can call tools — without that it can talk about the scene but not
    /// look at it or change it.
    pub tools: bool,
    /// It can see pictures.
    pub vision: bool,
    /// It thinks before it answers, and can be told not to.
    pub thinking: bool,
}

/// Where Ollama listens.
pub fn host() -> String {
    let given = std::env::var("OLLAMA_HOST").unwrap_or_default();
    let given = given.trim().trim_end_matches('/');
    if given.is_empty() {
        return "http://localhost:11434".into();
    }
    // `OLLAMA_HOST` is often just `127.0.0.1:11434`, or `0.0.0.0` — which
    // a server listens on and a client cannot reach.
    let with_scheme = if given.contains("://") {
        given.to_string()
    } else {
        format!("http://{given}")
    };
    with_scheme.replace("://0.0.0.0", "://localhost")
}

fn curl() -> &'static str {
    if cfg!(windows) {
        "curl.exe"
    } else {
        "curl"
    }
}

/// The models Ollama has, or why it could not be asked.
pub fn models() -> Result<Vec<Model>, String> {
    let host = host();
    let out = quiet_command(curl())
        .args(["-sS", "-m", "5"])
        .arg(format!("{host}/api/tags"))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    if !out.status.success() {
        return Err(not_running(&host));
    }
    parse_models(&String::from_utf8_lossy(&out.stdout)).ok_or_else(|| not_running(&host))
}

fn not_running(host: &str) -> String {
    format!(
        "Ollama is not answering at {host}. Install it from https://ollama.com, start it, \
         and fetch a model that can use tools: `ollama pull qwen3`."
    )
}

/// `/api/tags`'s answer. A model whose capabilities are not listed (an
/// Ollama older than 0.7) is taken to have tools, and the model itself
/// says if not.
pub fn parse_models(json: &str) -> Option<Vec<Model>> {
    let value: Value = serde_json::from_str(json).ok()?;
    let list = value.get("models")?.as_array()?;
    Some(
        list.iter()
            .filter_map(|m| {
                let name = m.get("name").or_else(|| m.get("model"))?.as_str()?;
                let caps: Option<Vec<&str>> = m
                    .get("capabilities")
                    .and_then(Value::as_array)
                    .map(|c| c.iter().filter_map(Value::as_str).collect());
                let has = |what: &str| caps.as_ref().is_none_or(|c| c.contains(&what));
                // Embedding models answer no chat at all.
                if caps.as_ref().is_some_and(|c| !c.contains(&"completion")) {
                    return None;
                }
                Some(Model {
                    name: name.to_string(),
                    tools: has("tools"),
                    vision: caps.as_ref().is_some_and(|c| c.contains(&"vision")),
                    thinking: caps.as_ref().is_some_and(|c| c.contains(&"thinking")),
                })
            })
            .collect(),
    )
}

fn model_path() -> Option<PathBuf> {
    Some(crate::prefs::config_dir()?.join("ollama-model"))
}

/// The model picked last time.
pub fn saved_model() -> Option<String> {
    let text = std::fs::read_to_string(model_path()?).ok()?;
    let name = text.trim();
    (!name.is_empty()).then(|| name.to_string())
}

/// Remembers the model picked.
pub fn save_model(name: &str) {
    if let Some(path) = model_path() {
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(path, name);
    }
}

/// Which of `models` to talk to: the one picked last time if Ollama still
/// has it, or else the first that can use tools, or else the first.
pub fn pick(models: &[Model], saved: Option<&str>) -> Option<usize> {
    saved
        .and_then(|s| models.iter().position(|m| m.name == s))
        .or_else(|| models.iter().position(|m| m.tools))
        .or(if models.is_empty() { None } else { Some(0) })
}

/// A turn's way to be stopped: the flag the loop looks at between steps,
/// and the request it is waiting on.
#[derive(Default)]
pub struct Stop {
    stopped: AtomicBool,
    waiting: Mutex<Option<Child>>,
}

impl Stop {
    pub fn stop(&self) {
        self.stopped.store(true, Ordering::SeqCst);
        if let Ok(mut child) = self.waiting.lock() {
            if let Some(child) = child.as_mut() {
                let _ = child.kill();
            }
        }
    }

    fn stopped(&self) -> bool {
        self.stopped.load(Ordering::SeqCst)
    }
}

/// `kreate-mcp`, spoken to as an MCP client speaks: one JSON message a line.
struct Connector {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
    next: u64,
}

impl Connector {
    fn start(path: &Path) -> Result<Self, String> {
        let mut child = quiet_command(path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("cannot start kreate-mcp: {e}"))?;
        let input = child.stdin.take().expect("piped");
        let output = BufReader::new(child.stdout.take().expect("piped"));
        Ok(Self {
            child,
            input,
            output,
            next: 1,
        })
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        writeln!(self.input, "{message}")
            .and_then(|_| self.input.flush())
            .map_err(|e| format!("kreate-mcp stopped listening: {e}"))
    }

    fn call(&mut self, method: &str, params: Value) -> Result<Value, String> {
        let id = self.next;
        self.next += 1;
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let mut line = String::new();
        loop {
            line.clear();
            let read = self
                .output
                .read_line(&mut line)
                .map_err(|e| format!("kreate-mcp: {e}"))?;
            if read == 0 {
                return Err("kreate-mcp stopped".into());
            }
            let Ok(reply) = serde_json::from_str::<Value>(&line) else {
                continue;
            };
            if reply.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            if let Some(error) = reply.get("error") {
                return Err(error
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("kreate-mcp refused")
                    .to_string());
            }
            return Ok(reply.get("result").cloned().unwrap_or(Value::Null));
        }
    }
}

impl Drop for Connector {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// The connector's tools as Ollama takes them.
pub fn ollama_tools(listed: &Value) -> Vec<Value> {
    listed
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| {
                    Some(json!({
                        "type": "function",
                        "function": {
                            "name": t.get("name")?.as_str()?,
                            "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                            "parameters": t.get("inputSchema").cloned()
                                .unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                        }
                    }))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// What one streamed line of `/api/chat` carries.
#[derive(Debug, Default, PartialEq)]
pub struct Chunk {
    pub text: String,
    pub calls: Vec<Value>,
    pub done: bool,
    pub error: Option<String>,
}

pub fn parse_chunk(line: &str) -> Chunk {
    let Ok(value) = serde_json::from_str::<Value>(line) else {
        return Chunk::default();
    };
    let message = value.get("message");
    Chunk {
        text: message
            .and_then(|m| m.get("content"))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        calls: message
            .and_then(|m| m.get("tool_calls"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default(),
        done: value.get("done").and_then(Value::as_bool).unwrap_or(false),
        error: value
            .get("error")
            .and_then(Value::as_str)
            .map(str::to_string),
    }
}

/// What a tool gave back: its words, and its pictures as base64.
pub fn tool_result(result: &Value) -> (String, Vec<String>, bool) {
    let mut words = Vec::new();
    let mut pictures = Vec::new();
    for item in result
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = item.get("text").and_then(Value::as_str) {
                    words.push(t.to_string());
                }
            }
            Some("image") => {
                if let Some(d) = item.get("data").and_then(Value::as_str) {
                    pictures.push(d.to_string());
                }
            }
            _ => {}
        }
    }
    let failed = result
        .get("isError")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    (words.join("\n"), pictures, failed)
}

/// One message from the user, answered: runs on a thread of its own and
/// says what happens on `tx`, ending with [`Event::Done`]. `history` is the
/// conversation so far and grows by this turn.
#[allow(clippy::too_many_arguments)]
pub fn turn(
    model: Model,
    history: Arc<Mutex<Vec<Value>>>,
    words: String,
    pictures: Vec<String>,
    connector: PathBuf,
    system: &str,
    stop: Arc<Stop>,
    tx: Sender<Event>,
    wake: Arc<dyn Fn() + Send + Sync>,
) {
    let say = |event: Event| {
        let _ = tx.send(event);
        wake();
    };
    let end = match run(
        &model, &history, words, pictures, &connector, system, &stop, &say,
    ) {
        Ok(()) => None,
        Err(_) if stop.stopped() => Some("stopped".to_string()),
        Err(why) => Some(why),
    };
    say(Event::Done(end));
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &Model,
    history: &Mutex<Vec<Value>>,
    words: String,
    pictures: Vec<String>,
    connector: &Path,
    system: &str,
    stop: &Stop,
    say: &dyn Fn(Event),
) -> Result<(), String> {
    let mut mcp = Connector::start(connector)?;
    let hello = mcp.call(
        "initialize",
        json!({
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": { "name": "kreate-ollama", "version": env!("CARGO_PKG_VERSION") }
        }),
    )?;
    mcp.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;
    let tools = if model.tools {
        ollama_tools(&mcp.call("tools/list", json!({}))?)
    } else {
        Vec::new()
    };
    {
        let mut history = history.lock().map_err(|_| "the conversation was lost")?;
        if history.is_empty() {
            let instructions = hello
                .get("instructions")
                .and_then(Value::as_str)
                .unwrap_or("");
            history.push(
                json!({ "role": "system", "content": format!("{system}\n\n{instructions}") }),
            );
        }
        let mut message = json!({ "role": "user", "content": words });
        if !pictures.is_empty() {
            message["images"] = json!(pictures);
        }
        history.push(message);
    }
    let mut last_call = String::new();
    let mut repeats = 0;
    let mut answer_now = false;
    for round in 0..=MAX_ROUNDS {
        if stop.stopped() {
            return Err("stopped".into());
        }
        answer_now |= round == MAX_ROUNDS;
        let messages = history
            .lock()
            .map_err(|_| "the conversation was lost")?
            .clone();
        let mut body = json!({
            "model": model.name,
            "messages": messages,
            "stream": true,
            "options": { "num_ctx": CONTEXT },
        });
        // No tools once it has had enough: then it can only answer.
        if !tools.is_empty() && !answer_now {
            body["tools"] = json!(tools);
        }
        // A thinking model's thoughts are slow and not shown; told not to.
        if model.thinking {
            body["think"] = json!(false);
        }
        let (text, calls) = ask(&body, stop, say)?;
        let mut reply = json!({ "role": "assistant", "content": text });
        if !calls.is_empty() {
            reply["tool_calls"] = json!(calls);
        }
        history
            .lock()
            .map_err(|_| "the conversation was lost")?
            .push(reply);
        if calls.is_empty() {
            return Ok(());
        }
        for call in calls {
            if stop.stopped() {
                return Err("stopped".into());
            }
            let function = call.get("function").cloned().unwrap_or(Value::Null);
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let arguments = match function.get("arguments") {
                // Some models give the arguments as a string of JSON.
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                Some(v) => v.clone(),
                None => json!({}),
            };
            // The same call a third time running: it has what it asked for.
            let this_call = format!("{name}{arguments}");
            repeats = if this_call == last_call {
                repeats + 1
            } else {
                0
            };
            last_call = this_call;
            if repeats >= 2 {
                answer_now = true;
            }
            say(Event::Tool {
                name: name.clone(),
                detail: crate::claude::tool_detail(&name, &arguments),
            });
            let (content, images, failed) = match mcp.call(
                "tools/call",
                json!({ "name": name, "arguments": arguments }),
            ) {
                Ok(result) => tool_result(&result),
                Err(why) => (why, Vec::new(), true),
            };
            if failed {
                say(Event::ToolError(content.clone()));
            }
            let mut answer = json!({ "role": "tool", "tool_name": name, "content": content });
            if model.vision && !images.is_empty() {
                answer["images"] = json!(images);
            }
            history
                .lock()
                .map_err(|_| "the conversation was lost")?
                .push(answer);
        }
    }
    Ok(())
}

/// One request to `/api/chat`, its words streamed out as they come.
fn ask(body: &Value, stop: &Stop, say: &dyn Fn(Event)) -> Result<(String, Vec<Value>), String> {
    let host = host();
    let mut child = quiet_command(curl())
        .args(["-sS", "-N", "-X", "POST"])
        .arg(format!("{host}/api/chat"))
        .args([
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            "@-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run curl: {e}"))?;
    let mut input = child.stdin.take().expect("piped");
    let stdout = child.stdout.take().expect("piped");
    let mut stderr = child.stderr.take().expect("piped");
    let request = body.to_string();
    // A conversation with pictures is more than a pipe holds.
    let writer = std::thread::spawn(move || {
        let _ = input.write_all(request.as_bytes());
    });
    *stop.waiting.lock().map_err(|_| "stopped")? = Some(child);
    let mut text = String::new();
    let mut calls = Vec::new();
    let mut error = None;
    for line in BufReader::new(stdout).lines() {
        let Ok(line) = line else { break };
        let chunk = parse_chunk(&line);
        if !chunk.text.is_empty() {
            text.push_str(&chunk.text);
            say(Event::TextPart(chunk.text));
        }
        calls.extend(chunk.calls);
        if chunk.error.is_some() {
            error = chunk.error;
        }
        if chunk.done {
            break;
        }
    }
    let _ = writer.join();
    let mut child = stop
        .waiting
        .lock()
        .map_err(|_| "stopped")?
        .take()
        .expect("put there above");
    let status = child.wait();
    if stop.stopped() {
        return Err("stopped".into());
    }
    if let Some(error) = error {
        return Err(format!("Ollama: {error}"));
    }
    if !status.is_ok_and(|s| s.success()) {
        let mut why = String::new();
        let _ = stderr.read_to_string(&mut why);
        return Err(format!("{} ({})", not_running(&host), why.trim()));
    }
    Ok((text, calls))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_models_are_read_with_what_they_can_do() {
        let json = r#"{"models":[
            {"name":"qwen3:8b","capabilities":["completion","tools","thinking"]},
            {"name":"llava:7b","capabilities":["completion","vision"]},
            {"name":"nomic-embed-text","capabilities":["embedding"]},
            {"name":"old:1b"}
        ]}"#;
        let models = parse_models(json).unwrap();
        assert_eq!(
            models
                .iter()
                .map(|m| (m.name.as_str(), m.tools, m.vision, m.thinking))
                .collect::<Vec<_>>(),
            vec![
                ("qwen3:8b", true, false, true),
                ("llava:7b", false, true, false),
                ("old:1b", true, false, false),
            ],
            "the embedding model is not one to talk to"
        );
        assert_eq!(pick(&models, None), Some(0));
        assert_eq!(pick(&models, Some("llava:7b")), Some(1));
        assert_eq!(pick(&models, Some("gone")), Some(0));
        assert_eq!(pick(&[], None), None);
    }

    #[test]
    fn streamed_lines_give_words_calls_and_the_end() {
        let words = parse_chunk(r#"{"message":{"role":"assistant","content":"Hel"},"done":false}"#);
        assert_eq!(words.text, "Hel");
        let call = parse_chunk(
            r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_scene","arguments":{}}}]},"done":false}"#,
        );
        assert_eq!(call.calls.len(), 1);
        assert!(parse_chunk(r#"{"done":true}"#).done);
        assert_eq!(
            parse_chunk(r#"{"error":"model does not support tools"}"#).error,
            Some("model does not support tools".into())
        );
    }

    #[test]
    fn the_connector_s_tools_become_functions_and_its_answers_words() {
        let listed = json!({ "tools": [{
            "name": "run_script",
            "description": "Runs lines",
            "inputSchema": { "type": "object", "properties": { "script": { "type": "string" } } }
        }]});
        let tools = ollama_tools(&listed);
        assert_eq!(tools[0]["function"]["name"], "run_script");
        assert_eq!(
            tools[0]["function"]["parameters"]["properties"]["script"]["type"],
            "string"
        );
        let (words, pictures, failed) = tool_result(&json!({
            "content": [
                { "type": "text", "text": "done" },
                { "type": "image", "data": "AAAA", "mimeType": "image/png" }
            ]
        }));
        assert_eq!((words.as_str(), pictures.len(), failed), ("done", 1, false));
    }
}
