//! Models made by AI: a picture into a 3D model with TRELLIS, or words into
//! a picture with Z-Image Turbo first and then into a model — both public
//! Hugging Face Spaces, MIT and Apache-2.0, free to call.
//!
//! A Space is a Gradio app, asked through its queue: upload the picture,
//! join the queue with the function's inputs, and read the answer as it
//! streams back. The function is found by its API name in the Space's own
//! configuration, so a Space that reorders its functions still answers.
//!
//! Without a token Hugging Face lends a little GPU time by address; TRELLIS
//! asks for more than that at once. A free account's token — Settings ▸
//! Access Tokens, read access — raises it. It is kept as the Poly Pizza key
//! is: in the settings folder, readable by the user alone, handed to curl on
//! its standard input.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The Space that turns a picture into a model.
const TRELLIS: &str = "https://trellis-community-trellis.hf.space";
/// The Space that turns words into a picture.
const ZIMAGE: &str = "https://tongyi-mai-z-image-turbo.hf.space";

/// What is asked for.
#[derive(Clone, Debug, PartialEq)]
pub enum Request {
    /// Words describing one object.
    Text(String),
    /// A picture of one object, on a plain background or none.
    Image(PathBuf),
}

/// Where made models are kept: beside the downloaded assets.
pub fn folder() -> Option<PathBuf> {
    crate::assets::cache_root().map(|r| r.join("generated"))
}

/// Where the user's Hugging Face token is kept.
pub fn token_path() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|d| d.join("huggingface-token"))
}

/// The user's Hugging Face token, when they have given one.
pub fn token() -> Option<String> {
    let token = std::fs::read_to_string(token_path()?).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Keeps `token`, or forgets it when empty.
pub fn save_token(token: &str) -> Result<(), String> {
    let path = token_path().ok_or("no settings folder")?;
    let token = token.trim();
    if token.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    }
    if !token.starts_with("hf_") || !token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        return Err("a Hugging Face token starts with hf_".into());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, token).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

/// Makes a model, saying how it goes to `progress` — as catalog keys, a
/// number after `|` where one goes — and the `.glb` when done, with the
/// picture it was made from beside it (same name, `.png`).
pub fn make(request: &Request, progress: &dyn Fn(String)) -> Result<PathBuf, String> {
    let dir = folder().ok_or("no cache folder")?;
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let name = match request {
        Request::Text(words) => slug(words),
        Request::Image(path) => slug(&path.file_stem().unwrap_or_default().to_string_lossy()),
    };
    let base = dir.join(format!("{name}-{stamp}"));
    let picture = base.with_extension("png");
    let token = token();
    let space = |url: &'static str| Space {
        url,
        token: token.clone(),
        session: format!("kreate{stamp}"),
    };
    match request {
        Request::Text(words) => {
            progress("asset-gen-drawing".into());
            let zimage = space(ZIMAGE);
            let prompt = format!(
                "{words}, a single object, whole and centred, plain light grey background, \
                 soft even studio light, product photograph"
            );
            let answer = zimage.run(
                "generate",
                serde_json::json!([prompt, "1024x1024 ( 1:1 )", 42, 8, 3.0, true, []]),
                progress,
            )?;
            let file = answer[0][0]["image"].clone();
            zimage.download(&file, &picture)?;
        }
        Request::Image(path) => {
            std::fs::copy(path, &picture).map_err(|e| format!("cannot read the picture: {e}"))?;
        }
    }
    progress("asset-gen-cutting".into());
    let trellis = space(TRELLIS);
    trellis.run("start_session", serde_json::json!([]), progress)?;
    let uploaded = trellis.upload(&picture)?;
    let cut = trellis.run("preprocess_image", serde_json::json!([uploaded]), progress)?;
    progress("asset-gen-shaping".into());
    let made = trellis.run(
        "generate_and_extract_glb",
        serde_json::json!([cut[0], [], 0, 7.5, 12, 3.0, 12, "stochastic", 0.95, 1024]),
        progress,
    )?;
    let glb = base.with_extension("glb");
    trellis.download(&made[2], &glb)?;
    Ok(glb)
}

/// A name for a file, from words.
fn slug(words: &str) -> String {
    let mut out: String = words
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '-' })
        .collect::<String>()
        .split('-')
        .filter(|w| !w.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    out.truncate(48);
    if out.is_empty() {
        "model".into()
    } else {
        out
    }
}

/// One Space, asked as one visitor.
struct Space {
    url: &'static str,
    token: Option<String>,
    session: String,
}

impl Space {
    /// curl, with the token on its standard input when there is one.
    fn curl(&self, args: &[&str], body: Option<&str>) -> Result<std::process::Child, String> {
        let program = if cfg!(windows) { "curl.exe" } else { "curl" };
        let mut command: Command = crate::claude::quiet_command(program);
        command
            .args([
                "--silent",
                "--show-error",
                "--fail",
                "--location",
                "--no-buffer",
            ])
            .args(["--max-time", "900", "--max-filesize", "536870912"])
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if self.token.is_some() {
            command.args(["--header", "@-"]);
        }
        if let Some(body) = body {
            command.args([
                "--header",
                "Content-Type: application/json",
                "--data-binary",
                body,
            ]);
        }
        let mut child = command
            .spawn()
            .map_err(|e| format!("cannot start curl: {e}"))?;
        if let (Some(token), Some(mut stdin)) = (&self.token, child.stdin.take()) {
            let _ = writeln!(stdin, "Authorization: Bearer {token}");
        }
        Ok(child)
    }

    /// What curl wrote, or why it failed.
    fn output(&self, args: &[&str], body: Option<&str>) -> Result<String, String> {
        let out = self
            .curl(args, body)?
            .wait_with_output()
            .map_err(|e| e.to_string())?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            Err(format!(
                "{} did not answer: {}",
                self.url,
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
    }

    /// The number the Space's queue knows a function by.
    fn function(&self, api_name: &str) -> Result<u64, String> {
        let config: serde_json::Value =
            serde_json::from_str(&self.output(&[&format!("{}/config", self.url)], None)?)
                .map_err(|e| format!("the Space's configuration: {e}"))?;
        config["dependencies"]
            .as_array()
            .and_then(|deps| {
                deps.iter()
                    .find(|d| d["api_name"] == api_name)
                    .and_then(|d| d["id"].as_u64())
            })
            .ok_or_else(|| format!("the Space has no `{api_name}` any more"))
    }

    /// A file sent to the Space, as its functions take one.
    fn upload(&self, path: &Path) -> Result<serde_json::Value, String> {
        let form = format!("files=@{}", path.display());
        let answer = self.output(
            &["--form", &form, &format!("{}/gradio_api/upload", self.url)],
            None,
        )?;
        let paths: Vec<String> =
            serde_json::from_str(&answer).map_err(|e| format!("an upload's answer: {e}"))?;
        let remote = paths.into_iter().next().ok_or("the upload was not kept")?;
        Ok(serde_json::json!({ "path": remote, "meta": { "_type": "gradio.FileData" } }))
    }

    /// Runs a function, reading its queue until it is done; its outputs.
    fn run(
        &self,
        api_name: &str,
        inputs: serde_json::Value,
        progress: &dyn Fn(String),
    ) -> Result<serde_json::Value, String> {
        let index = self.function(api_name)?;
        let join = serde_json::json!({
            "data": inputs,
            "fn_index": index,
            "session_hash": self.session,
            "event_data": null,
            "trigger_id": null,
        })
        .to_string();
        self.output(
            &[&format!("{}/gradio_api/queue/join", self.url)],
            Some(&join),
        )?;
        let mut child = self.curl(
            &[&format!(
                "{}/gradio_api/queue/data?session_hash={}",
                self.url, self.session
            )],
            None,
        )?;
        let stdout = child.stdout.take().ok_or("no answer stream")?;
        let mut result = Err(format!("the Space stopped answering `{api_name}`"));
        for line in std::io::BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let Ok(message) = serde_json::from_str::<serde_json::Value>(data.trim()) else {
                continue;
            };
            match message["msg"].as_str() {
                Some("estimation") => {
                    if let Some(rank) = message["rank"].as_u64().filter(|r| *r > 0) {
                        progress(format!("asset-gen-queue|{rank}"));
                    }
                }
                Some("process_completed") => {
                    result = if message["success"] == true {
                        Ok(message["output"]["data"].clone())
                    } else {
                        let said = message["output"]["error"]
                            .as_str()
                            .map(str::to_string)
                            .unwrap_or_default();
                        Err(if !said.is_empty() {
                            format!("the Space said: {said}")
                        } else if self.token.is_none() {
                            "Hugging Face did not lend the GPU time this needs. A free \
                             Hugging Face token raises the limit."
                                .to_string()
                        } else {
                            "the Space could not make it this time; try again in a while, \
                             or with another picture"
                                .to_string()
                        })
                    };
                    break;
                }
                _ => {}
            }
        }
        let _ = child.kill();
        let _ = child.wait();
        result
    }

    /// A file a function made, onto the disk.
    fn download(&self, file: &serde_json::Value, dest: &Path) -> Result<(), String> {
        let url = file["url"]
            .as_str()
            .map(str::to_string)
            .or_else(|| {
                file["path"]
                    .as_str()
                    .map(|p| format!("{}/gradio_api/file={p}", self.url))
            })
            .ok_or("the Space made no file")?;
        if !url.starts_with(self.url) {
            return Err("the Space's file is somewhere else".into());
        }
        let partial = dest.with_extension("part");
        let out = self.output(&["--output", &partial.to_string_lossy(), &url], None);
        match out {
            Ok(_) => std::fs::rename(&partial, dest).map_err(|e| e.to_string()),
            Err(e) => {
                let _ = std::fs::remove_file(&partial);
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_name_is_made_of_the_words() {
        assert_eq!(slug("a small wooden chair!"), "a-small-wooden-chair");
        assert_eq!(slug("나무 의자"), "나무-의자");
        assert_eq!(slug("///"), "model");
        assert!(slug(&"long ".repeat(40)).len() <= 48);
    }

    /// Against the real Spaces: `cargo test -p kreate --lib generate -- --ignored`.
    #[test]
    #[ignore = "asks Hugging Face, over the network"]
    fn words_become_a_model() {
        let made = make(&Request::Text("a small wooden chair".into()), &|s| {
            eprintln!("  {s}")
        });
        eprintln!("{made:?}");
        let picture = folder().and_then(|d| {
            std::fs::read_dir(d)
                .ok()?
                .flatten()
                .map(|e| e.path())
                .find(|p| p.extension().is_some_and(|e| e == "png"))
        });
        assert!(picture.is_some(), "the picture was drawn");
    }
}
