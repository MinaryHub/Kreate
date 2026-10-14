//! The chat panel's conversations, kept: each one a Markdown file in
//! `chats/` in the settings folder, readable as it stands, named by when it
//! began, with the document it was about; and the last one as JSON beside
//! them, so a restart shows it again and the assistant carries on from it.
//!
//! Only what the panel showed is written — words, the tools the assistant
//! called and what they said, errors. What was attached goes in by name, as
//! the panel lists it, never its content, and anything that looks like a
//! key or a token is blanked out before it reaches the disk.

use std::path::{Path, PathBuf};

use serde_json::{json, Value};

/// How many conversations are kept; the oldest go first.
pub const KEEP: usize = 200;

/// The folder the conversations are written to.
pub fn chat_dir() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|d| d.join("chats"))
}

/// One conversation, as kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Record {
    /// When it began, `YYYY-MM-DD HH:MM:SS` UTC; also its file's name.
    pub started: String,
    /// Which assistant: Claude, Gemini, ChatGPT, Ollama.
    pub assistant: String,
    /// The document open when it began.
    pub document: Option<String>,
    /// The assistant's own session, to carry on from after a restart.
    pub session: Option<String>,
    /// (who, what): `you`, `assistant`, `tool`, `error`, `note`.
    pub messages: Vec<(String, String)>,
}

/// Now, as the record writes it.
pub fn now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, m, d) = crate::logs::civil_date((secs / 86_400) as i64);
    let s = secs % 86_400;
    format!(
        "{y:04}-{m:02}-{d:02} {:02}:{:02}:{:02}",
        s / 3600,
        s / 60 % 60,
        s % 60
    )
}

/// `text` with anything that looks like a secret blanked: API keys and
/// tokens by their usual shapes, and the value after a word like `token`,
/// `key` or `password`.
pub fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let words: Vec<&str> = text.split_inclusive(|c: char| c.is_whitespace()).collect();
    let mut hide_next = false;
    for w in words {
        let core = w.trim_end();
        let tail = &w[core.len()..];
        let lower = core.to_ascii_lowercase();
        let shaped = ["sk-", "sk_", "hf_", "ghp_", "gho_", "xox", "ya29.", "eyj"]
            .iter()
            .any(|p| {
                lower
                    .trim_start_matches(['"', '\'', '(', '=', ':'])
                    .starts_with(p)
            })
            && core.len() >= 16;
        if hide_next || shaped {
            out.push_str("[hidden]");
            out.push_str(tail);
            hide_next = false;
            continue;
        }
        // "token: abc", "api_key=abc", "\"token\": \"abc\""
        let key_word = ["token", "key", "password", "secret", "apikey", "api_key"]
            .iter()
            .any(|k| lower.trim_matches(['"', '\'', ':', '=', ',']).ends_with(k));
        if let Some((name, value)) = core.split_once('=').or_else(|| core.split_once(':')) {
            let n = name.to_ascii_lowercase();
            if ["token", "key", "password", "secret"]
                .iter()
                .any(|k| n.trim_matches(['"', '\'']).ends_with(k))
                && !value.trim_matches(['"', '\'']).is_empty()
            {
                out.push_str(name);
                out.push_str(&core[name.len()..name.len() + 1]);
                out.push_str("[hidden]");
                out.push_str(tail);
                continue;
            }
        }
        if key_word && (core.ends_with(':') || core.ends_with('=')) {
            hide_next = true;
        }
        out.push_str(w);
    }
    out
}

/// The conversation as a Markdown page.
pub fn markdown(r: &Record) -> String {
    let mut out = format!(
        "# Kreate chat, {} UTC\n\n- Assistant: {}\n",
        r.started, r.assistant
    );
    if let Some(d) = &r.document {
        out.push_str(&format!("- Document: {d}\n"));
    }
    out.push('\n');
    for (who, what) in &r.messages {
        let what = redact(what);
        match who.as_str() {
            "you" => out.push_str(&format!("**You:**\n\n{what}\n\n")),
            "assistant" => out.push_str(&format!("**{}:**\n\n{what}\n\n", r.assistant)),
            "tool" => out.push_str(&format!("> tool: {}\n\n", what.replace('\n', "\n> "))),
            "error" => out.push_str(&format!("> **error:** {}\n\n", what.replace('\n', "\n> "))),
            _ => out.push_str(&format!("*{what}*\n\n")),
        }
    }
    out
}

fn file_stem(started: &str) -> String {
    started.replace([' ', ':'], "-")
}

fn to_json(r: &Record) -> Value {
    json!({
        "started": r.started,
        "assistant": r.assistant,
        "document": r.document,
        "session": r.session,
        "messages": r.messages.iter().map(|(w, t)| json!([w, redact(t)])).collect::<Vec<_>>(),
    })
}

fn from_json(v: &Value) -> Option<Record> {
    Some(Record {
        started: v.get("started")?.as_str()?.to_string(),
        assistant: v.get("assistant")?.as_str()?.to_string(),
        document: v
            .get("document")
            .and_then(Value::as_str)
            .map(str::to_string),
        session: v.get("session").and_then(Value::as_str).map(str::to_string),
        messages: v
            .get("messages")?
            .as_array()?
            .iter()
            .filter_map(|m| {
                Some((
                    m.get(0)?.as_str()?.to_string(),
                    m.get(1)?.as_str()?.to_string(),
                ))
            })
            .collect(),
    })
}

/// Writes the conversation's page and, as the last one, its JSON; keeps the
/// newest [`KEEP`] pages.
pub fn save_in(dir: &Path, r: &Record) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let page = dir.join(format!("{}.md", file_stem(&r.started)));
    std::fs::write(&page, markdown(r))?;
    std::fs::write(dir.join("last.json"), to_json(r).to_string())?;
    let mut pages: Vec<PathBuf> = std::fs::read_dir(dir)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "md"))
        .collect();
    if pages.len() > KEEP {
        pages.sort();
        for old in &pages[..pages.len() - KEEP] {
            let _ = std::fs::remove_file(old);
        }
    }
    Ok(page)
}

/// The last conversation, to show again.
pub fn load_last_in(dir: &Path) -> Option<Record> {
    let text = std::fs::read_to_string(dir.join("last.json")).ok()?;
    from_json(&serde_json::from_str(&text).ok()?)
}

/// Forgets the last conversation, so a restart begins empty: what New Chat
/// asks for.
pub fn forget_last_in(dir: &Path) {
    let _ = std::fs::remove_file(dir.join("last.json"));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d =
            std::env::temp_dir().join(format!("kreate-chat-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn record() -> Record {
        Record {
            started: "2026-10-15 01:02:03".into(),
            assistant: "Claude".into(),
            document: Some("C:/parts/Spider.kreate".into()),
            session: Some("abc-123".into()),
            messages: vec![
                ("you".into(), "Make a bracket\n[drawing.pdf]".into()),
                (
                    "tool".into(),
                    "run_script: kreate.ops.object.add_primitive()".into(),
                ),
                ("assistant".into(), "Done: a 40 mm bracket.".into()),
                ("error".into(), "no object called Cube".into()),
            ],
        }
    }

    /// Kept, it is a readable page and the last conversation, which comes
    /// back whole after a restart; forgotten, nothing comes back.
    #[test]
    fn a_conversation_is_kept_and_comes_back() {
        let d = dir("keep");
        let page = save_in(&d, &record()).unwrap();
        let md = std::fs::read_to_string(&page).unwrap();
        assert!(page.ends_with("2026-10-15-01-02-03.md"), "{page:?}");
        assert!(md.contains("- Document: C:/parts/Spider.kreate"), "{md}");
        assert!(
            md.contains("**You:**\n\nMake a bracket\n[drawing.pdf]"),
            "{md}"
        );
        assert!(md.contains("> tool: run_script"), "{md}");
        assert!(md.contains("> **error:** no object called Cube"), "{md}");
        assert_eq!(load_last_in(&d), Some(record()));
        forget_last_in(&d);
        assert_eq!(load_last_in(&d), None);
        assert!(page.exists(), "the page stays");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn secrets_are_blanked() {
        let said = "my key is sk-ant-api03-abcdefghijklmnop and token: abcdef123 \
            then \"api_key\": \"zzz\" and hf_ABCDEFGHIJKLMNOPQRST, 20 mm is fine";
        let out = redact(said);
        for secret in ["sk-ant-api03", "abcdef123", "zzz", "hf_ABCDEF"] {
            assert!(!out.contains(secret), "{secret} in {out}");
        }
        assert!(out.contains("20 mm is fine"), "{out}");
    }

    #[test]
    fn only_the_newest_are_kept() {
        let d = dir("prune");
        for i in 0..(KEEP + 3) {
            let mut r = record();
            r.started = format!("2026-01-01 00:{:02}:{:02}", i / 60, i % 60);
            save_in(&d, &r).unwrap();
        }
        let pages = std::fs::read_dir(&d)
            .unwrap()
            .flatten()
            .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
            .count();
        assert_eq!(pages, KEEP);
        assert!(
            !d.join("2026-01-01-00-00-00.md").exists(),
            "the oldest went"
        );
        let _ = std::fs::remove_dir_all(&d);
    }
}
