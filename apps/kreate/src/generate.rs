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

/// A height map of `words` — skin, scales, bark — drawn by Z-Image as a
/// seamless grey picture, white high and black low; its path.
pub fn height_map(words: &str, progress: &dyn Fn(String)) -> Result<PathBuf, String> {
    let dir = folder().ok_or("no cache folder")?.join("height");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let space = Space {
        url: ZIMAGE,
        token: token(),
        session: format!("kreate{stamp}"),
    };
    progress("asset-gen-drawing".into());
    // Asked for as a photograph of the surface itself: a model told
    // "tileable" draws tiles, and one told "height map" draws a lit relief.
    // Kreate makes it a height map and makes it repeat.
    let prompt = format!(
        "macro photograph of {words}, the surface alone filling the whole frame edge to \
         edge, seen straight on from above, soft overhead light, evenly lit, uniform pattern, \
         no objects, no border, no frame"
    );
    let answer = space.run(
        "generate",
        serde_json::json!([prompt, "1024x1024 ( 1:1 )", 42, 8, 3.0, true, []]),
        progress,
    )?;
    let drawn = dir.join(format!("{}-{stamp}-photo.png", slug(words)));
    space.download(&answer[0][0]["image"], &drawn)?;
    let photo = image::open(&drawn).map_err(|e| format!("cannot read the picture: {e}"))?;
    let path = dir.join(format!("{}-{stamp}.png", slug(words)));
    as_height(&photo.to_luma32f())
        .save(&path)
        .map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&drawn);
    Ok(path)
}

/// A photograph's brightness as heights: the light's broad fall across it
/// taken away, stretched to the full grey range, and its edges blended into
/// the opposite edges so it repeats without a seam.
fn as_height(photo: &image::ImageBuffer<image::Luma<f32>, Vec<f32>>) -> image::GrayImage {
    let (w, h) = photo.dimensions();
    let at = |x: u32, y: u32| photo.get_pixel(x % w, y % h).0[0];
    // The broad light: the mean over a block a sixteenth of the size.
    let block = (w.min(h) / 16).max(1);
    let (bw, bh) = (w.div_ceil(block), h.div_ceil(block));
    let mut means = vec![0f32; (bw * bh) as usize];
    for by in 0..bh {
        for bx in 0..bw {
            let (mut sum, mut count) = (0.0f32, 0.0f32);
            for y in by * block..((by + 1) * block).min(h) {
                for x in bx * block..((bx + 1) * block).min(w) {
                    sum += at(x, y);
                    count += 1.0;
                }
            }
            means[(by * bw + bx) as usize] = sum / count.max(1.0);
        }
    }
    let broad = |x: u32, y: u32| {
        // Blended between the four nearest blocks' means.
        let fx = (x as f32 / block as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
        let fy = (y as f32 / block as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
        let (x0, y0) = (fx.floor() as u32, fy.floor() as u32);
        let (x1, y1) = ((x0 + 1).min(bw - 1), (y0 + 1).min(bh - 1));
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let m = |x: u32, y: u32| means[(y * bw + x) as usize];
        (m(x0, y0) * (1.0 - tx) + m(x1, y0) * tx) * (1.0 - ty)
            + (m(x0, y1) * (1.0 - tx) + m(x1, y1) * tx) * ty
    };
    let detail: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| at(x, y) - broad(x, y))
        .collect();
    // Repeating: each pixel mixed with the one half a picture away, by a
    // weight that is one in the middle and nothing at the edges — so the
    // edges are the opposite middle, which runs on into the other side.
    let d = |x: u32, y: u32| detail[((y % h) * w + (x % w)) as usize];
    let weight = |t: f32| (std::f32::consts::PI * t).sin().max(0.0);
    let blended: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (x, y)))
        .map(|(x, y)| {
            let a = weight(x as f32 / w as f32) * weight(y as f32 / h as f32);
            let b = weight(((x + w / 2) % w) as f32 / w as f32)
                * weight(((y + h / 2) % h) as f32 / h as f32);
            (d(x, y) * a + d(x + w / 2, y + h / 2) * b) / (a + b).max(1e-6)
        })
        .collect();
    // Stretched to the grey range, the few extremes clipped.
    let mut sorted = blended.clone();
    sorted.sort_by(f32::total_cmp);
    let lo = sorted[sorted.len() / 100];
    let hi = sorted[sorted.len() * 99 / 100];
    let span = (hi - lo).max(1e-6);
    image::GrayImage::from_fn(w, h, |x, y| {
        let v = ((blended[(y * w + x) as usize] - lo) / span).clamp(0.0, 1.0);
        image::Luma([(v * 255.0).round() as u8])
    })
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

    /// Against the real Space: `cargo test -p kreate --lib height -- --ignored`.
    #[test]
    #[ignore = "asks Hugging Face, over the network"]
    fn words_become_a_height_map() {
        let path = height_map("reptile scales", &|s| eprintln!("  {s}")).expect("drawn");
        let image = image::open(&path).expect("a picture");
        eprintln!("{} {}x{}", path.display(), image.width(), image.height());
        assert!(image.width() >= 512);
    }

    /// A photo lit from one side, with a fine pattern: the height keeps the
    /// pattern, loses the light, and runs on from one edge into the other.
    #[test]
    fn a_photo_becomes_a_level_repeating_height() {
        let photo = image::ImageBuffer::from_fn(256, 256, |x, y| {
            let light = x as f32 / 256.0 * 0.6;
            let pattern = if (x / 8 + y / 8) % 2 == 0 { 0.2 } else { 0.0 };
            image::Luma([light + pattern])
        });
        let height = as_height(&photo);
        let mean = |x0: u32, x1: u32| {
            let mut sum = 0.0;
            for y in 0..256 {
                for x in x0..x1 {
                    sum += height.get_pixel(x, y).0[0] as f64;
                }
            }
            sum / (256.0 * (x1 - x0) as f64)
        };
        // The light's slope is gone: left and right are alike on average.
        assert!(
            (mean(0, 64) - mean(192, 256)).abs() < 25.0,
            "{} {}",
            mean(0, 64),
            mean(192, 256)
        );
        // It repeats: the last column runs on into the first.
        let jump: f64 = (0..256)
            .map(|y| {
                (height.get_pixel(255, y).0[0] as f64 - height.get_pixel(0, y).0[0] as f64).abs()
            })
            .sum::<f64>()
            / 256.0;
        let step: f64 = (0..256)
            .map(|y| {
                (height.get_pixel(128, y).0[0] as f64 - height.get_pixel(127, y).0[0] as f64).abs()
            })
            .sum::<f64>()
            / 256.0;
        assert!(jump <= step + 20.0, "edge {jump} against inside {step}");
    }
}
