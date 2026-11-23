//! Built-in HTTP/HTTPS client, using the platform's certificate verifier.
//!
//! Replaces spawning external `curl` processes across asset libraries,
//! updater, Ollama, Sketchfab, Objaverse, YouTube captions, and Gradio Space
//! generation — so network features work on minimal Linux installs, Windows
//! sandboxes, and Android without external CLI binaries or console-window
//! overhead, while reusing keep-alive TLS connections.

use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Duration;

use ureq::tls::{RootCerts, TlsConfig, TlsProvider};
use ureq::Agent;

/// Default User-Agent header identifying Kreate.
pub fn user_agent() -> String {
    format!(
        "Kreate/{} (+https://github.com/MinaryHub/Kreate)",
        crate::version::VERSION
    )
}

fn tls_config() -> TlsConfig {
    TlsConfig::builder()
        .provider(TlsProvider::Rustls)
        .root_certs(RootCerts::PlatformVerifier)
        .build()
}

/// Shared connection-pooled agent for standard API calls (60 s global budget).
pub fn api_agent() -> &'static Agent {
    static AGENT: OnceLock<Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        Agent::config_builder()
            .tls_config(tls_config())
            .user_agent(user_agent())
            .timeout_connect(Some(Duration::from_secs(15)))
            .timeout_global(Some(Duration::from_secs(60)))
            .build()
            .into()
    })
}

/// Shared agent for non-throwing HTTP status inspection (e.g. 401/403/429).
pub fn status_agent(timeout: Duration) -> Agent {
    Agent::config_builder()
        .tls_config(tls_config())
        .user_agent(user_agent())
        .http_status_as_error(false)
        .timeout_connect(Some(Duration::from_secs(15)))
        .timeout_global(Some(timeout))
        .build()
        .into()
}

/// Shared connection-pooled agent for large file downloads (10 min budget).
pub fn download_agent() -> &'static Agent {
    static AGENT: OnceLock<Agent> = OnceLock::new();
    AGENT.get_or_init(|| {
        Agent::config_builder()
            .tls_config(tls_config())
            .user_agent(user_agent())
            .timeout_connect(Some(Duration::from_secs(20)))
            .timeout_global(Some(Duration::from_secs(600)))
            .build()
            .into()
    })
}

/// Shared agent for long-running streaming responses (Ollama chat, Gradio SSE).
pub fn stream_agent(connect_timeout: Duration, recv_timeout: Duration) -> Agent {
    Agent::config_builder()
        .tls_config(tls_config())
        .user_agent(user_agent())
        .timeout_connect(Some(connect_timeout))
        .timeout_per_read(Some(recv_timeout))
        .build()
        .into()
}

/// Fetches `url` as UTF-8 text (up to `max_bytes`), returning a clean error message.
pub fn get_text(url: &str, timeout: Duration, max_bytes: u64) -> Result<String, String> {
    let bytes = get_bytes_with_headers(url, &[], timeout, max_bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Fetches `url` as raw bytes with optional headers and size cap.
pub fn get_bytes_with_headers(
    url: &str,
    headers: &[(&str, &str)],
    timeout: Duration,
    max_bytes: u64,
) -> Result<Vec<u8>, String> {
    let agent: Agent = Agent::config_builder()
        .tls_config(tls_config())
        .user_agent(user_agent())
        .timeout_connect(Some(Duration::from_secs(15).min(timeout)))
        .timeout_global(Some(timeout))
        .build()
        .into();
    let mut req = agent.get(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let mut resp = req.call().map_err(|e| format_error(url, e))?;
    read_capped(resp.body_mut().as_reader(), max_bytes)
}

/// Fetches `url` returning `(status_code, body_text)` without treating 4xx/5xx as transport errors.
pub fn get_status_and_text(
    url: &str,
    headers: &[(&str, &str)],
    timeout: Duration,
    max_bytes: u64,
) -> Result<(u16, String), String> {
    let agent = status_agent(timeout);
    let mut req = agent.get(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let mut resp = req.call().map_err(|e| format_error(url, e))?;
    let code = resp.status().as_u16();
    let bytes = read_capped(resp.body_mut().as_reader(), max_bytes)?;
    Ok((code, String::from_utf8_lossy(&bytes).into_owned()))
}

/// Sends a POST request with raw body and optional headers, returning UTF-8 text.
pub fn post_text(
    url: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    timeout: Duration,
    max_bytes: u64,
) -> Result<String, String> {
    let agent: Agent = Agent::config_builder()
        .tls_config(tls_config())
        .user_agent(user_agent())
        .timeout_connect(Some(Duration::from_secs(15).min(timeout)))
        .timeout_global(Some(timeout))
        .build()
        .into();
    let mut req = agent.post(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let mut resp = req.send(body).map_err(|e| format_error(url, e))?;
    let bytes = read_capped(resp.body_mut().as_reader(), max_bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

/// Sends a POST request with JSON/bytes and returns a streaming buffered reader over the response body.
pub fn post_stream(
    url: &str,
    headers: &[(&str, &str)],
    body: &[u8],
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<BufReader<Box<dyn Read + Send + Sync>>, String> {
    let agent = stream_agent(connect_timeout, read_timeout);
    let mut req = agent.post(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.send(body).map_err(|e| format_error(url, e))?;
    let (_, body) = resp.into_parts();
    Ok(BufReader::new(Box::new(body.into_reader())))
}

/// Sends a GET request and returns a streaming buffered reader over the response body.
pub fn get_stream(
    url: &str,
    headers: &[(&str, &str)],
    connect_timeout: Duration,
    read_timeout: Duration,
) -> Result<BufReader<Box<dyn Read + Send + Sync>>, String> {
    let agent = stream_agent(connect_timeout, read_timeout);
    let mut req = agent.get(url);
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let resp = req.call().map_err(|e| format_error(url, e))?;
    let (_, body) = resp.into_parts();
    Ok(BufReader::new(Box::new(body.into_reader())))
}

/// Uploads a single file as `multipart/form-data` under field `field_name`.
pub fn post_multipart_file(
    url: &str,
    headers: &[(&str, &str)],
    field_name: &str,
    path: &Path,
    timeout: Duration,
    max_bytes: u64,
) -> Result<String, String> {
    let file_bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().replace(['"', '\r', '\n'], "_"))
        .unwrap_or_else(|| "upload.bin".into());
    let boundary = format!(
        "----KreateBoundary{:016x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x517cc1b727220a95)
    );
    let mut payload = Vec::with_capacity(file_bytes.len() + 256);
    let _ = write!(
        payload,
        "--{boundary}\r\nContent-Disposition: form-data; name=\"{field_name}\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
    );
    payload.extend_from_slice(&file_bytes);
    let _ = write!(payload, "\r\n--{boundary}--\r\n");

    let content_type = format!("multipart/form-data; boundary={boundary}");
    let mut all_headers: Vec<(&str, &str)> = headers.to_vec();
    all_headers.push(("Content-Type", &content_type));
    post_text(url, &all_headers, &payload, timeout, max_bytes)
}

/// Streams `url` atomically into `dest` via a `.part` sibling file, enforcing `max_bytes`.
pub fn download_to_file(url: &str, dest: &Path, max_bytes: u64) -> Result<(), String> {
    download_to_file_with_headers(url, &[], dest, max_bytes)
}

/// Streams `url` with custom headers atomically into `dest` via a `.part` sibling file.
pub fn download_to_file_with_headers(
    url: &str,
    headers: &[(&str, &str)],
    dest: &Path,
    max_bytes: u64,
) -> Result<(), String> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let partial = dest.with_extension("part");
    let write_res = (|| -> Result<(), String> {
        let mut req = download_agent().get(url);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let mut resp = req.call().map_err(|e| format_error(url, e))?;
        let mut out = std::fs::File::create(&partial).map_err(|e| e.to_string())?;
        let mut limited = resp.body_mut().as_reader().take(max_bytes + 1);
        let written = std::io::copy(&mut limited, &mut out).map_err(|e| e.to_string())?;
        if written > max_bytes {
            return Err(format!("response from {url} exceeded {max_bytes} bytes"));
        }
        out.flush().map_err(|e| e.to_string())
    })();
    if let Err(e) = write_res {
        let _ = std::fs::remove_file(&partial);
        return Err(e);
    }
    std::fs::rename(&partial, dest).map_err(|e| {
        let _ = std::fs::remove_file(&partial);
        e.to_string()
    })
}

fn read_capped(reader: impl Read, max_bytes: u64) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    reader
        .take(max_bytes + 1)
        .read_to_end(&mut out)
        .map_err(|e| e.to_string())?;
    if out.len() as u64 > max_bytes {
        return Err(format!("response exceeded {max_bytes} bytes"));
    }
    Ok(out)
}

fn format_error(url: &str, err: ureq::Error) -> String {
    match err {
        ureq::Error::StatusCode(code) => format!("HTTP {code} fetching {url}"),
        other => format!("could not fetch {url}: {other}"),
    }
}
