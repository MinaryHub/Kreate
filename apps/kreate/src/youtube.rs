//! YouTube links in a chat message, read for the assistant: the video's
//! title, channel, length, description and its captions with their times.
//!
//! Kreate reads them itself and puts what it found under the message, rather
//! than letting the assistant onto the web: the assistants here run with no
//! tools but the document's (see `claude.rs`), and that stays so. It also
//! means every assistant gets the same — Claude, Gemini, ChatGPT and a model
//! on this computer alike.
//!
//! A video's picture and sound are not read: what an assistant learns is what
//! is said in it (the captions, the uploader's or YouTube's own) and what its
//! page says about it.
//!
//! The captions come by way of the YouTube app's own player request: the
//! caption addresses the web page gives out answer with nothing unless the
//! asker proves it is a browser playing the video, which `curl` cannot.

use std::io::Write;
use std::process::Stdio;

use serde_json::Value;

/// The most of a video's captions sent, in characters: about an hour of talk.
const MAX_TRANSCRIPT: usize = 40_000;
/// The most of its description.
const MAX_DESCRIPTION: usize = 3_000;
/// The most videos read from one message.
pub const MAX_VIDEOS: usize = 3;

/// One video, as read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Video {
    pub id: String,
    pub title: String,
    pub channel: String,
    pub seconds: u64,
    pub description: String,
    /// The captions' language and whether they were made by YouTube's
    /// speech recognition, and the lines with their start in seconds.
    pub captions: Option<Captions>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Captions {
    pub language: String,
    pub automatic: bool,
    pub lines: Vec<(f64, String)>,
}

/// The video ids a message links to, each once, in order: watch pages,
/// youtu.be, Shorts, live, embeds.
pub fn video_ids(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| c.is_whitespace() || "<>()[]\"'".contains(c)) {
        let w = word
            .trim_start_matches("https://")
            .trim_start_matches("http://")
            .trim_start_matches("www.")
            .trim_start_matches("m.")
            .trim_start_matches("music.");
        let id = if let Some(rest) = w.strip_prefix("youtu.be/") {
            Some(rest)
        } else if let Some(rest) = w.strip_prefix("youtube.com/") {
            if let Some(q) = rest.strip_prefix("watch?") {
                q.split('&').find_map(|kv| kv.strip_prefix("v="))
            } else {
                ["shorts/", "live/", "embed/", "v/"]
                    .iter()
                    .find_map(|p| rest.strip_prefix(p))
            }
        } else {
            None
        };
        let Some(id) = id else { continue };
        let id: String = id
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
            .collect();
        if id.len() == 11 && !out.contains(&id) {
            out.push(id);
        }
    }
    out
}

/// `curl`, its body on its input, its answer as text.
fn curl(args: &[&str], body: Option<&str>) -> Result<String, String> {
    let program = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut child = crate::claude::quiet_command(program)
        .args(["-s", "-S", "-L", "-m", "30"])
        .args(args)
        .stdin(if body.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start curl: {e}"))?;
    if let (Some(body), Some(mut input)) = (body, child.stdin.take()) {
        let _ = input.write_all(body.as_bytes());
    }
    let out = child.wait_with_output().map_err(|e| format!("curl: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Reads a video: its details and, if it has any, its captions —
/// `languages` tried in order, captions someone wrote before YouTube's own.
pub fn fetch(id: &str, languages: &[&str]) -> Result<Video, String> {
    let body = serde_json::json!({
        "context": { "client": {
            "clientName": "ANDROID",
            "clientVersion": "20.10.38",
            "androidSdkVersion": 34,
            "hl": "en",
        }},
        "videoId": id,
    })
    .to_string();
    let answer = curl(
        &[
            "-H",
            "Content-Type: application/json",
            "-A",
            "com.google.android.youtube/20.10.38 (Linux; U; Android 14)",
            "--data-binary",
            "@-",
            "https://www.youtube.com/youtubei/v1/player?prettyPrint=false",
        ],
        Some(&body),
    )?;
    let player: Value = serde_json::from_str(&answer)
        .map_err(|_| "YouTube answered with something else".to_string())?;
    let status = player
        .pointer("/playabilityStatus/status")
        .and_then(Value::as_str)
        .unwrap_or("");
    let details = player.get("videoDetails");
    if details.is_none() {
        let why = player
            .pointer("/playabilityStatus/reason")
            .and_then(Value::as_str)
            .unwrap_or(status);
        return Err(format!("YouTube will not give this video: {why}"));
    }
    let text = |key: &str| {
        details
            .and_then(|d| d.get(key))
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let mut video = Video {
        id: id.to_string(),
        title: text("title"),
        channel: text("author"),
        seconds: text("lengthSeconds").parse().unwrap_or(0),
        description: text("shortDescription"),
        captions: None,
    };
    let tracks = player
        .pointer("/captions/playerCaptionsTracklistRenderer/captionTracks")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if let Some(track) = pick_track(&tracks, languages) {
        let url = track.get("baseUrl").and_then(Value::as_str).unwrap_or("");
        if !url.is_empty() {
            if let Ok(xml) = curl(&[url], None) {
                let lines = parse_captions(&xml);
                if !lines.is_empty() {
                    video.captions = Some(Captions {
                        language: track
                            .get("languageCode")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        automatic: track.get("kind").and_then(Value::as_str) == Some("asr"),
                        lines,
                    });
                }
            }
        }
    }
    Ok(video)
}

/// The caption track to read: written ones before YouTube's speech
/// recognition, and within each the first of `languages` there is.
pub fn pick_track<'a>(tracks: &'a [Value], languages: &[&str]) -> Option<&'a Value> {
    let lang = |t: &Value| {
        t.get("languageCode")
            .and_then(Value::as_str)
            .unwrap_or("")
            .split('-')
            .next()
            .unwrap_or("")
            .to_string()
    };
    let automatic = |t: &Value| t.get("kind").and_then(Value::as_str) == Some("asr");
    for auto in [false, true] {
        let kind: Vec<&Value> = tracks.iter().filter(|t| automatic(t) == auto).collect();
        for want in languages {
            if let Some(t) = kind.iter().find(|t| lang(t) == *want) {
                return Some(t);
            }
        }
        if let Some(t) = kind.first() {
            return Some(t);
        }
    }
    None
}

/// Caption lines from YouTube's timed text, in either of its forms: `<p
/// t="ms">` (format 3, words perhaps in `<s>`) or `<text start="s">`.
pub fn parse_captions(xml: &str) -> Vec<(f64, String)> {
    let mut out = Vec::new();
    let mut rest = xml;
    loop {
        let (open, close, scale, key) = match (rest.find("<p "), rest.find("<text ")) {
            (Some(p), Some(t)) if t < p => (t, "</text>", 1.0, "start=\""),
            (Some(p), _) => (p, "</p>", 0.001, "t=\""),
            (None, Some(t)) => (t, "</text>", 1.0, "start=\""),
            (None, None) => break,
        };
        rest = &rest[open..];
        let Some(head_end) = rest.find('>') else {
            break;
        };
        let head = &rest[..head_end];
        let start = head
            .find(key)
            .and_then(|at| {
                let v = &head[at + key.len()..];
                v[..v.find('"')?].parse::<f64>().ok()
            })
            .unwrap_or(0.0)
            * scale;
        let Some(end) = rest.find(close) else { break };
        let inner = if head.ends_with('/') {
            ""
        } else {
            &rest[head_end + 1..end]
        };
        let line = unescape(&strip_tags(inner))
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        if !line.is_empty() {
            out.push((start, line));
        }
        rest = &rest[end + close.len()..];
    }
    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut inside = false;
    for c in s.chars() {
        match c {
            '<' => inside = true,
            '>' => inside = false,
            c if !inside => out.push(c),
            _ => {}
        }
    }
    out
}

fn unescape(s: &str) -> String {
    // Twice: the timed text's own entities, then the ones in the words.
    let mut s = s.to_string();
    for _ in 0..2 {
        let mut out = String::with_capacity(s.len());
        let mut rest = s.as_str();
        while let Some(at) = rest.find('&') {
            out.push_str(&rest[..at]);
            rest = &rest[at..];
            let Some(semi) = rest[..rest.len().min(10)].find(';') else {
                out.push('&');
                rest = &rest[1..];
                continue;
            };
            let name = &rest[1..semi];
            let c = match name {
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                n if n.starts_with("#x") => u32::from_str_radix(&n[2..], 16)
                    .ok()
                    .and_then(char::from_u32),
                n if n.starts_with('#') => n[1..].parse().ok().and_then(char::from_u32),
                _ => None,
            };
            match c {
                Some(c) => {
                    out.push(c);
                    rest = &rest[semi + 1..];
                }
                None => {
                    out.push('&');
                    rest = &rest[1..];
                }
            }
        }
        out.push_str(rest);
        s = out;
    }
    s
}

fn clock(seconds: f64) -> String {
    let s = seconds.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// What goes under the message for one video.
pub fn describe(video: &Video) -> String {
    let mut out = format!(
        "[YouTube video https://youtu.be/{}, read by Kreate for this message]\nTitle: {}\nChannel: {}\nLength: {}\n",
        video.id,
        video.title,
        video.channel,
        clock(video.seconds as f64)
    );
    if !video.description.trim().is_empty() {
        let d: String = video.description.chars().take(MAX_DESCRIPTION).collect();
        out.push_str(&format!("Description:\n{}\n", d.trim()));
    }
    match &video.captions {
        Some(c) => {
            out.push_str(&format!(
                "Transcript ({}{}):\n",
                c.language,
                if c.automatic { ", automatic captions" } else { "" }
            ));
            let mut used = 0;
            for (t, line) in &c.lines {
                let row = format!("[{}] {}\n", clock(*t), line);
                used += row.len();
                if used > MAX_TRANSCRIPT {
                    out.push_str("[… the rest of the transcript left out for length]\n");
                    break;
                }
                out.push_str(&row);
            }
        }
        None => out.push_str(
            "No captions could be read: only the details above are known, not what is said or shown.\n",
        ),
    }
    out
}

/// The message with what its YouTube links lead to put under it, and a short
/// line per video for the conversation; reading a video that fails says so
/// in its place rather than holding the message back.
pub fn enrich(text: &str, languages: &[&str]) -> (String, Vec<String>) {
    let mut words = text.to_string();
    let mut notes = Vec::new();
    for id in video_ids(text).into_iter().take(MAX_VIDEOS) {
        match fetch(&id, languages) {
            Ok(v) => {
                notes.push(format!(
                    "{} — {}{}",
                    v.title,
                    v.channel,
                    if v.captions.is_some() {
                        ""
                    } else {
                        " (no captions)"
                    }
                ));
                words.push_str("\n\n");
                words.push_str(&describe(&v));
            }
            Err(why) => {
                notes.push(format!("{id}: {why}"));
                words.push_str(&format!(
                    "\n\n[YouTube video https://youtu.be/{id}: Kreate could not read it: {why}]"
                ));
            }
        }
    }
    (words, notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn links_in_every_shape_give_their_ids_once() {
        let text = "봐줘 https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10s 그리고 youtu.be/abcdefghijk?si=x \
            (https://youtube.com/shorts/ABCDEFGHIJK) https://m.youtube.com/watch?list=x&v=dQw4w9WgXcQ \
            https://example.com/watch?v=zzzzzzzzzzz";
        assert_eq!(
            video_ids(text),
            vec!["dQw4w9WgXcQ", "abcdefghijk", "ABCDEFGHIJK"]
        );
        assert!(video_ids("no links here").is_empty());
    }

    #[test]
    fn captions_are_read_in_both_forms() {
        let three = r#"<?xml version="1.0"?><timedtext format="3"><body>
            <p t="1360" d="1680">[&#9834;]</p><p t="18640" d="3000"><s>We&#39;re</s><s t="300"> no</s> strangers</p>
            <p t="20000" d="10"/></body></timedtext>"#;
        assert_eq!(
            parse_captions(three),
            vec![
                (1.36, "[♪]".to_string()),
                (18.64, "We're no strangers".to_string())
            ]
        );
        let old = r#"<transcript><text start="1.5" dur="2">a &amp;amp; b</text></transcript>"#;
        assert_eq!(parse_captions(old), vec![(1.5, "a & b".to_string())]);
    }

    #[test]
    fn written_captions_come_before_automatic_ones() {
        let tracks: Vec<Value> = serde_json::from_str(
            r#"[{"languageCode":"en","kind":"asr"},{"languageCode":"de"},{"languageCode":"ko"},{"languageCode":"ko","kind":"asr"}]"#,
        )
        .unwrap();
        let lang = |t: Option<&Value>| {
            t.and_then(|t| t["languageCode"].as_str())
                .map(str::to_string)
        };
        assert_eq!(lang(pick_track(&tracks, &["ko", "en"])), Some("ko".into()));
        assert_eq!(lang(pick_track(&tracks, &["en"])), Some("de".into()));
        assert_eq!(pick_track(&[], &["en"]), None);
    }

    #[test]
    fn a_video_is_described_with_its_times() {
        let v = Video {
            id: "dQw4w9WgXcQ".into(),
            title: "T".into(),
            channel: "C".into(),
            seconds: 213,
            description: String::new(),
            captions: Some(Captions {
                language: "en".into(),
                automatic: true,
                lines: vec![(65.0, "hello".into())],
            }),
        };
        let d = describe(&v);
        assert!(d.contains("Length: 3:33"), "{d}");
        assert!(d.contains("[1:05] hello"), "{d}");
        assert!(d.contains("automatic captions"), "{d}");
    }

    /// Against YouTube itself: needs the network, so run by hand.
    #[test]
    #[ignore]
    fn a_real_video_is_read_with_its_captions() {
        let v = fetch("dQw4w9WgXcQ", &["en"]).unwrap();
        assert!(v.title.contains("Never Gonna Give You Up"), "{v:?}");
        assert!(v.seconds > 200);
        let c = v.captions.expect("captions");
        assert!(c.lines.len() > 20, "{c:?}");
        assert!(c
            .lines
            .iter()
            .any(|(_, l)| l.to_lowercase().contains("never gonna")));
    }
}
