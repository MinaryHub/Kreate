//! Direct HTTPS AI API clients for Anthropic (Claude), OpenAI (ChatGPT), and
//! Google (Gemini), running Kreate's MCP tool loop in-process over `ureq` +
//! `rustls` with no external CLI binary (Claude Code, Codex, Gemini CLI, or
//! Node.js) required whenever an API key is configured.

use std::path::{Path, PathBuf};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{json, Value};

use crate::claude::{Event, Provider};
use crate::ollama::{Connector, Stop};

const MAX_ROUNDS: usize = 20;
const MAX_RESULT_CHARS: usize = 24_000;
const MAX_HISTORY_CHARS: usize = 160_000;
const API_TIMEOUT: Duration = Duration::from_secs(180);
const MAX_RESPONSE_BYTES: u64 = 16 * 1024 * 1024;

/// Resolves an API key for `provider` from Kreate's key store first, then from
/// standard environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
/// `GEMINI_API_KEY`).
pub fn api_key_for(provider: Provider) -> Option<String> {
    let service = provider.service()?;
    if let Some(k) = crate::ai_keys::key(service) {
        return Some(k);
    }
    for var in service.variables() {
        if let Ok(val) = std::env::var(var) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

/// Default model name when the user left the model box on "Default".
pub fn default_model(provider: Provider, requested: Option<&str>) -> String {
    if let Some(m) = requested.map(str::trim).filter(|s| !s.is_empty()) {
        // Translate Claude Code short aliases to current Anthropic API model IDs.
        if provider == Provider::Claude {
            return match m {
                "opus" => "claude-opus-4-6".to_string(),
                "sonnet" => "claude-sonnet-4-6".to_string(),
                "haiku" => "claude-haiku-4-5-20251001".to_string(),
                other => other.to_string(),
            };
        }
        return m.to_string();
    }
    match provider {
        Provider::Claude => "claude-sonnet-4-6".to_string(),
        Provider::ChatGpt => "gpt-4.1".to_string(),
        Provider::Gemini => "gemini-2.5-flash".to_string(),
        Provider::Ollama => String::new(),
    }
}

/// Prunes older middle turns from a conversation history once serialized text
/// exceeds `max_chars`, keeping any leading system message and the newest turns.
pub fn prune_history(history: &mut Vec<Value>, max_chars: usize) {
    let total_len = |h: &[Value]| -> usize { h.iter().map(|v| v.to_string().len()).sum() };
    let keep_first = usize::from(
        history
            .first()
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            == Some("system"),
    );
    while history.len() > keep_first + 4 && total_len(history) > max_chars {
        history.remove(keep_first);
    }
}

fn truncate_result(text: String) -> String {
    if text.len() <= MAX_RESULT_CHARS {
        return text;
    }
    let cut = text
        .char_indices()
        .take_while(|(i, _)| *i < MAX_RESULT_CHARS)
        .last()
        .map_or(0, |(i, c)| i + c.len_utf8());
    format!(
        "{}\n[… truncated: {} characters total]",
        &text[..cut],
        text.len()
    )
}

/// Runs one assistant turn against Anthropic, OpenAI, or Gemini directly over
/// HTTPS, calling `kreate-mcp` tools until the model finishes answering.
#[allow(clippy::too_many_arguments)]
pub fn turn(
    provider: Provider,
    api_key: String,
    model: String,
    history: Arc<Mutex<Vec<Value>>>,
    words: String,
    connector: PathBuf,
    system: &'static str,
    stop: Arc<Stop>,
    tx: Sender<Event>,
    wake: Arc<dyn Fn() + Send + Sync>,
) {
    let say = |event: Event| {
        let _ = tx.send(event);
        wake();
    };
    let end = run(
        provider, &api_key, &model, &history, words, &connector, system, &stop, &say,
    )
    .err();
    say(Event::Done(end));
}

#[allow(clippy::too_many_arguments)]
fn run(
    provider: Provider,
    api_key: &str,
    model: &str,
    history: &Mutex<Vec<Value>>,
    words: String,
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
            "clientInfo": { "name": "kreate-direct-api", "version": env!("CARGO_PKG_VERSION") }
        }),
    )?;
    mcp.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))?;
    let listed = mcp.call("tools/list", json!({}))?;
    let mcp_tools = listed
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let instructions = hello
        .get("instructions")
        .and_then(Value::as_str)
        .unwrap_or("");
    let full_system = if instructions.is_empty() {
        system.to_string()
    } else {
        format!("{system}\n\n{instructions}")
    };

    match provider {
        Provider::Claude => run_anthropic(
            api_key,
            model,
            &full_system,
            &mcp_tools,
            &mut mcp,
            history,
            words,
            stop,
            say,
        ),
        Provider::ChatGpt => run_openai(
            api_key,
            model,
            &full_system,
            &mcp_tools,
            &mut mcp,
            history,
            words,
            stop,
            say,
        ),
        Provider::Gemini => run_gemini(
            api_key,
            model,
            &full_system,
            &mcp_tools,
            &mut mcp,
            history,
            words,
            stop,
            say,
        ),
        Provider::Ollama => Err("Ollama uses its own local client".into()),
    }
}

fn call_mcp_tool(
    mcp: &mut Connector,
    name: &str,
    arguments: &Value,
    say: &dyn Fn(Event),
) -> (String, bool) {
    say(Event::Tool {
        name: name.to_string(),
        detail: crate::claude::tool_detail(name, arguments),
    });
    let (content, _, failed) = match mcp.call(
        "tools/call",
        json!({ "name": name, "arguments": arguments }),
    ) {
        Ok(result) => crate::ollama::tool_result(&result),
        Err(why) => (why, Vec::new(), true),
    };
    let content = truncate_result(content);
    if failed {
        say(Event::ToolError(content.clone()));
    }
    (content, failed)
}

#[allow(clippy::too_many_arguments)]
fn run_anthropic(
    api_key: &str,
    model: &str,
    system: &str,
    mcp_tools: &[Value],
    mcp: &mut Connector,
    history: &Mutex<Vec<Value>>,
    words: String,
    stop: &Stop,
    say: &dyn Fn(Event),
) -> Result<(), String> {
    let tools: Vec<Value> = mcp_tools
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            Some(json!({
                "name": name,
                "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                "input_schema": t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"})),
            }))
        })
        .collect();

    {
        let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
        h.push(json!({ "role": "user", "content": words }));
        prune_history(&mut h, MAX_HISTORY_CHARS);
    }

    let endpoint = std::env::var("ANTHROPIC_BASE_URL")
        .unwrap_or_else(|_| "https://api.anthropic.com".to_string());
    let url = format!("{}/v1/messages", endpoint.trim_end_matches('/'));

    for round in 0..=MAX_ROUNDS {
        if stop.stopped() {
            return Err("stopped".into());
        }
        let messages = {
            let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
            prune_history(&mut h, MAX_HISTORY_CHARS);
            h.clone()
        };
        let mut body = json!({
            "model": model,
            "max_tokens": 4096,
            "system": system,
            "messages": messages,
        });
        if round < MAX_ROUNDS && !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        let resp_text = crate::http::post_text(
            &url,
            &[
                ("Content-Type", "application/json"),
                ("x-api-key", api_key),
                ("anthropic-version", "2023-06-01"),
            ],
            body.to_string().as_bytes(),
            API_TIMEOUT,
            MAX_RESPONSE_BYTES,
        )?;
        let resp: Value =
            serde_json::from_str(&resp_text).map_err(|e| format!("Anthropic invalid JSON: {e}"))?;
        if let Some(err) = resp
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            return Err(format!("Anthropic: {err}"));
        }
        let content = resp
            .get("content")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        history
            .lock()
            .map_err(|_| "conversation lock poisoned")?
            .push(json!({ "role": "assistant", "content": content }));

        let mut tool_results = Vec::new();
        for block in &content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        if !t.is_empty() {
                            say(Event::TextPart(t.to_string()));
                        }
                    }
                }
                Some("tool_use") => {
                    if stop.stopped() {
                        return Err("stopped".into());
                    }
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or_else(|| json!({}));
                    let (out, failed) = call_mcp_tool(mcp, name, &input, say);
                    tool_results.push(json!({
                        "type": "tool_result",
                        "tool_use_id": id,
                        "content": out,
                        "is_error": failed,
                    }));
                }
                _ => {}
            }
        }
        if tool_results.is_empty() {
            return Ok(());
        }
        history
            .lock()
            .map_err(|_| "conversation lock poisoned")?
            .push(json!({ "role": "user", "content": tool_results }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_openai(
    api_key: &str,
    model: &str,
    system: &str,
    mcp_tools: &[Value],
    mcp: &mut Connector,
    history: &Mutex<Vec<Value>>,
    words: String,
    stop: &Stop,
    say: &dyn Fn(Event),
) -> Result<(), String> {
    let tools: Vec<Value> = mcp_tools
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            Some(json!({
                "type": "function",
                "function": {
                    "name": name,
                    "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                    "parameters": t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"})),
                }
            }))
        })
        .collect();

    {
        let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
        if h.is_empty() {
            h.push(json!({ "role": "system", "content": system }));
        }
        h.push(json!({ "role": "user", "content": words }));
        prune_history(&mut h, MAX_HISTORY_CHARS);
    }

    let endpoint = std::env::var("OPENAI_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
    let url = format!("{}/chat/completions", endpoint.trim_end_matches('/'));
    let auth = format!("Bearer {api_key}");

    for round in 0..=MAX_ROUNDS {
        if stop.stopped() {
            return Err("stopped".into());
        }
        let messages = {
            let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
            prune_history(&mut h, MAX_HISTORY_CHARS);
            h.clone()
        };
        let mut body = json!({
            "model": model,
            "messages": messages,
        });
        if round < MAX_ROUNDS && !tools.is_empty() {
            body["tools"] = json!(tools);
        }
        let resp_text = crate::http::post_text(
            &url,
            &[
                ("Content-Type", "application/json"),
                ("Authorization", &auth),
            ],
            body.to_string().as_bytes(),
            API_TIMEOUT,
            MAX_RESPONSE_BYTES,
        )?;
        let resp: Value =
            serde_json::from_str(&resp_text).map_err(|e| format!("OpenAI invalid JSON: {e}"))?;
        if let Some(err) = resp
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            return Err(format!("OpenAI: {err}"));
        }
        let message = resp
            .get("choices")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("message"))
            .cloned()
            .ok_or("OpenAI returned no message")?;
        if let Some(text) = message.get("content").and_then(Value::as_str) {
            if !text.is_empty() {
                say(Event::TextPart(text.to_string()));
            }
        }
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        history
            .lock()
            .map_err(|_| "conversation lock poisoned")?
            .push(message);
        if tool_calls.is_empty() {
            return Ok(());
        }
        for call in tool_calls {
            if stop.stopped() {
                return Err("stopped".into());
            }
            let id = call.get("id").and_then(Value::as_str).unwrap_or("");
            let func = call.get("function").cloned().unwrap_or(Value::Null);
            let name = func.get("name").and_then(Value::as_str).unwrap_or("");
            let args: Value = match func.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or_else(|_| json!({})),
                Some(v) => v.clone(),
                None => json!({}),
            };
            let (out, _) = call_mcp_tool(mcp, name, &args, say);
            history
                .lock()
                .map_err(|_| "conversation lock poisoned")?
                .push(json!({
                    "role": "tool",
                    "tool_call_id": id,
                    "content": out,
                }));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_gemini(
    api_key: &str,
    model: &str,
    system: &str,
    mcp_tools: &[Value],
    mcp: &mut Connector,
    history: &Mutex<Vec<Value>>,
    words: String,
    stop: &Stop,
    say: &dyn Fn(Event),
) -> Result<(), String> {
    let declarations: Vec<Value> = mcp_tools
        .iter()
        .filter_map(|t| {
            let name = t.get("name")?.as_str()?;
            Some(json!({
                "name": name,
                "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                "parameters": sanitize_gemini_schema(
                    &t.get("inputSchema").cloned().unwrap_or_else(|| json!({"type": "object"}))
                ),
            }))
        })
        .collect();

    {
        let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
        h.push(json!({ "role": "user", "parts": [{ "text": words }] }));
        prune_history(&mut h, MAX_HISTORY_CHARS);
    }

    let endpoint = std::env::var("GEMINI_BASE_URL")
        .unwrap_or_else(|_| "https://generativelanguage.googleapis.com".to_string());
    let url = format!(
        "{}/v1beta/models/{model}:generateContent",
        endpoint.trim_end_matches('/')
    );

    for round in 0..=MAX_ROUNDS {
        if stop.stopped() {
            return Err("stopped".into());
        }
        let contents = {
            let mut h = history.lock().map_err(|_| "conversation lock poisoned")?;
            prune_history(&mut h, MAX_HISTORY_CHARS);
            h.clone()
        };
        let mut body = json!({
            "system_instruction": { "parts": [{ "text": system }] },
            "contents": contents,
        });
        if round < MAX_ROUNDS && !declarations.is_empty() {
            body["tools"] = json!([{ "function_declarations": declarations }]);
        }
        let resp_text = crate::http::post_text(
            &url,
            &[
                ("Content-Type", "application/json"),
                ("x-goog-api-key", api_key),
            ],
            body.to_string().as_bytes(),
            API_TIMEOUT,
            MAX_RESPONSE_BYTES,
        )?;
        let resp: Value =
            serde_json::from_str(&resp_text).map_err(|e| format!("Gemini invalid JSON: {e}"))?;
        if let Some(err) = resp
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
        {
            return Err(format!("Gemini: {err}"));
        }
        let parts = resp
            .get("candidates")
            .and_then(Value::as_array)
            .and_then(|c| c.first())
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        history
            .lock()
            .map_err(|_| "conversation lock poisoned")?
            .push(json!({ "role": "model", "parts": parts }));

        let mut fn_responses = Vec::new();
        for part in &parts {
            if let Some(text) = part.get("text").and_then(Value::as_str) {
                if !text.is_empty() {
                    say(Event::TextPart(text.to_string()));
                }
            }
            if let Some(fc) = part.get("functionCall") {
                if stop.stopped() {
                    return Err("stopped".into());
                }
                let name = fc.get("name").and_then(Value::as_str).unwrap_or("");
                let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                let (out, failed) = call_mcp_tool(mcp, name, &args, say);
                fn_responses.push(json!({
                    "functionResponse": {
                        "name": name,
                        "response": { "result": out, "error": failed }
                    }
                }));
            }
        }
        if fn_responses.is_empty() {
            return Ok(());
        }
        history
            .lock()
            .map_err(|_| "conversation lock poisoned")?
            .push(json!({ "role": "user", "parts": fn_responses }));
    }
    Ok(())
}

/// Strips JSON Schema keywords Gemini's `function_declarations` rejects (such as
/// `additionalProperties` and `$schema`).
fn sanitize_gemini_schema(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let mut out = serde_json::Map::new();
            for (k, v) in map {
                if matches!(k.as_str(), "additionalProperties" | "$schema" | "default") {
                    continue;
                }
                out.insert(k.clone(), sanitize_gemini_schema(v));
            }
            Value::Object(out)
        }
        Value::Array(arr) => Value::Array(arr.iter().map(sanitize_gemini_schema).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prune_history_keeps_system_and_recent_turns() {
        let mut h = vec![json!({"role": "system", "content": "sys"})];
        for i in 0..20 {
            h.push(json!({"role": "user", "content": format!("msg {i} {}", "x".repeat(200))}));
        }
        prune_history(&mut h, 1500);
        assert_eq!(h[0]["role"], "system");
        assert!(h.len() < 12);
        assert!(h.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .starts_with("msg 19"));
    }

    #[test]
    fn default_model_maps_claude_aliases() {
        assert_eq!(
            default_model(Provider::Claude, Some("sonnet")),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            default_model(Provider::Claude, Some("opus")),
            "claude-opus-4-6"
        );
        assert_eq!(default_model(Provider::ChatGpt, None), "gpt-4.1");
        assert_eq!(default_model(Provider::Gemini, None), "gemini-2.5-flash");
    }

    #[test]
    fn sanitize_gemini_schema_removes_unsupported_keys() {
        let s = json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "additionalProperties": false,
            "properties": { "a": { "type": "string", "default": "x" } }
        });
        let clean = sanitize_gemini_schema(&s);
        assert!(clean.get("$schema").is_none());
        assert!(clean.get("additionalProperties").is_none());
        assert!(clean["properties"]["a"].get("default").is_none());
        assert_eq!(clean["properties"]["a"]["type"], "string");
    }
}
