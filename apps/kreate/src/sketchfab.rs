//! Sketchfab, signed in: your own models, the ones you liked, and every
//! model its maker lets be downloaded — fetched with Sketchfab's download
//! API, which wants to know who is asking.
//!
//! Signing in is your own API token, from sketchfab.com/settings/password,
//! kept as the other sources' keys are: in Kreate's settings folder, readable
//! by you alone, never in a document. It goes to `curl` on its standard input,
//! not its command line, where any process on the machine could read it.
//!
//! Objaverse, beside it, needs no account but only has what the dataset
//! gathered; this has everything downloadable, as it is today, in the
//! format its maker uploaded — a `.glb` when Sketchfab made one, otherwise
//! a glTF with its textures, from a zip.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde_json::Value;

use crate::assets::{cache_root, download, safe_id, Entry, Kind, Query, Source};

const API: &str = "https://api.sketchfab.com/v3";

/// The reason a list was not asked: there is no token to ask with. The panel
/// shows the box for one instead of this.
pub const TOKEN_MISSING: &str = "Sketchfab needs your API token";
/// The reason Sketchfab refused, worded for the panel.
pub const TOKEN_REFUSED: &str = "Sketchfab did not accept the API token";

/// Where the token is kept.
pub fn token_path() -> Option<PathBuf> {
    crate::prefs::config_dir().map(|d| d.join("sketchfab-token"))
}

/// The user's token, when they have given one.
pub fn token() -> Option<String> {
    let token = std::fs::read_to_string(token_path()?).ok()?;
    let token = token.trim().to_string();
    (!token.is_empty()).then_some(token)
}

/// Keeps `token` for next time, or signs out when it is empty.
pub fn save_token(token: &str) -> Result<(), String> {
    let path = token_path().ok_or("no settings folder")?;
    let token = token.trim();
    if let Ok(mut who) = WHO.lock() {
        *who = None;
    }
    if token.is_empty() {
        let _ = std::fs::remove_file(&path);
        return Ok(());
    }
    // A header's worth of letters and nothing else: a line break would add a
    // header of its own.
    if !token.chars().all(|c| c.is_ascii_alphanumeric()) || token.len() > 128 {
        return Err("that does not look like a Sketchfab API token".into());
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

/// Who the token signs in as, once Sketchfab has said.
static WHO: Mutex<Option<String>> = Mutex::new(None);

/// The account's name, when a list has asked for it.
pub fn signed_in_as() -> Option<String> {
    WHO.lock().ok()?.clone()
}

/// `url` from Sketchfab's API, signed in.
fn get(url: &str, token: &str) -> Result<Value, String> {
    use std::io::Write;
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let mut child = crate::claude::quiet_command(curl)
        .args(["--silent", "--show-error", "--location"])
        .args(["--max-time", "60", "--max-filesize", "16777216"])
        .args(["--write-out", "\n%{http_code}"])
        .args(["--header", "@-"])
        .arg(url)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start curl: {e}"))?;
    if let Some(mut stdin) = child.stdin.take() {
        let _ = writeln!(stdin, "Authorization: Token {token}");
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let (body, code) = text.rsplit_once('\n').unwrap_or((&text, ""));
    match code.trim() {
        "200" => serde_json::from_str(body).map_err(|e| format!("Sketchfab's answer: {e}")),
        "401" => Err(TOKEN_REFUSED.into()),
        "403" => Err("Sketchfab does not let this model be downloaded".into()),
        "404" => Err("Sketchfab has no such model".into()),
        "429" => Err("Sketchfab asks to wait a moment; try again shortly".into()),
        "" => Err(String::from_utf8_lossy(&out.stderr).trim().to_string()),
        other => Err(format!("Sketchfab answered {other}")),
    }
}

/// One model of an answer, as an entry.
pub fn entry_of(m: &Value) -> Option<Entry> {
    let id = m["uid"].as_str()?.to_string();
    if !safe_id(&id) {
        return None;
    }
    let licence = m["license"]["label"]
        .as_str()
        .or_else(|| m["license"].as_str())
        .map(str::to_string);
    let thumbnail_url = m["thumbnails"]["images"].as_array().and_then(|images| {
        images
            .iter()
            .filter(|i| i["width"].as_u64().is_some_and(|w| w >= 200))
            .min_by_key(|i| i["width"].as_u64().unwrap_or(u64::MAX))
            .and_then(|i| i["url"].as_str())
            .map(str::to_string)
    });
    let tags = m["tags"]
        .as_array()
        .map(|t| {
            t.iter()
                .filter_map(|t| t["name"].as_str().or_else(|| t.as_str()))
                .map(str::to_lowercase)
                .collect()
        })
        .unwrap_or_default();
    Some(Entry {
        name: m["name"].as_str().unwrap_or(&id).to_string(),
        kind: Kind::Model,
        source: Source::Sketchfab,
        tags,
        authors: m["user"]["username"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        thumbnail_url: thumbnail_url.filter(|u| u.starts_with("https://")),
        licence,
        animated: m["animationCount"].as_u64().unwrap_or(0) > 0,
        id,
        ..Entry::blank()
    })
}

/// Up to `pages` pages of a list Sketchfab gives, as entries.
fn pages(first: String, token: &str, pages: usize, out: &mut Vec<Entry>) -> Result<(), String> {
    let mut next = Some(first);
    for _ in 0..pages {
        let Some(page) = next.take() else { break };
        let answer = get(&page, token)?;
        for m in answer["results"].as_array().into_iter().flatten() {
            // A like is the model itself, or wraps it.
            let model = if m["model"].is_object() {
                &m["model"]
            } else {
                m
            };
            if let Some(e) = entry_of(model) {
                if !out.iter().any(|o| o.id == e.id) {
                    out.push(e);
                }
            }
        }
        next = answer["next"]
            .as_str()
            .filter(|n| n.starts_with(API))
            .map(str::to_string);
    }
    Ok(())
}

/// With no words, your own models and then the ones you liked; with words,
/// every downloadable model that matches.
pub fn list(kind: Kind, query: &Query) -> Result<Vec<Entry>, String> {
    if kind != Kind::Model {
        return Ok(Vec::new());
    }
    let token = token().ok_or(TOKEN_MISSING)?;
    // Who this is — which also says whether the token is good.
    let me = get(&format!("{API}/me"), &token)?;
    let name = me["displayName"]
        .as_str()
        .filter(|n| !n.is_empty())
        .or_else(|| me["username"].as_str())
        .unwrap_or("")
        .to_string();
    if let Ok(mut who) = WHO.lock() {
        *who = Some(name);
    }
    let mut out = Vec::new();
    let words = query.text.trim();
    if words.is_empty() {
        pages(format!("{API}/me/models?count=24"), &token, 2, &mut out)?;
        pages(format!("{API}/me/likes?count=24"), &token, 2, &mut out)?;
    } else {
        let mut url = format!(
            "{API}/search?type=models&downloadable=true&count=24&q={}",
            encode(words)
        );
        if query.animated {
            url.push_str("&animated=true");
        }
        pages(url, &token, 3, &mut out)?;
    }
    if query.animated {
        out.retain(|e| e.animated);
    }
    Ok(out)
}

/// One model by its uid.
pub fn find(uid: &str) -> Result<Entry, String> {
    if !safe_id(uid) {
        return Err(format!("`{uid}` is not a model's uid"));
    }
    let token = token().ok_or(TOKEN_MISSING)?;
    entry_of(&get(&format!("{API}/models/{uid}"), &token)?)
        .ok_or_else(|| format!("Sketchfab has no model `{uid}`"))
}

/// Of the download API's answer, the file to fetch: the `.glb` when there is
/// one, or else the zip of the glTF and its textures.
pub fn pick_download(answer: &Value) -> Option<(String, bool)> {
    let url = |format: &str| {
        answer[format]["url"]
            .as_str()
            .filter(|u| u.starts_with("https://"))
            .map(str::to_string)
    };
    url("glb")
        .map(|u| (u, true))
        .or_else(|| url("gltf").map(|u| (u, false)))
}

/// A model on the disk, kept for next time: a `.glb`, or the `.gltf` in the
/// folder its zip unpacked to.
pub fn fetch(entry: &Entry) -> Result<PathBuf, String> {
    if !safe_id(&entry.id) {
        return Err(format!("`{}` is not a model's uid", entry.id));
    }
    let dir = cache_root()
        .ok_or("no cache folder")?
        .join("sketchfab")
        .join("models");
    let glb = dir.join(format!("{}.glb", entry.id));
    if glb.exists() {
        return Ok(glb);
    }
    let unpacked = dir.join(&entry.id);
    if let Some(found) = find_gltf(&unpacked) {
        return Ok(found);
    }
    let token = token().ok_or(TOKEN_MISSING)?;
    let answer = get(&format!("{API}/models/{}/download", entry.id), &token)?;
    // The address is signed and short-lived; fetched now, without the token.
    let (url, is_glb) = pick_download(&answer).ok_or("Sketchfab offered no glTF of this model")?;
    if is_glb {
        download(&url, &glb)?;
        return Ok(glb);
    }
    let zip = dir.join(format!("{}.zip", entry.id));
    download(&url, &zip)?;
    let unzipped = unzip_tree(&zip, &unpacked);
    let _ = std::fs::remove_file(&zip);
    unzipped?;
    find_gltf(&unpacked).ok_or_else(|| "the download had no glTF in it".to_string())
}

/// The first `.gltf` or `.glb` under `dir`, nearest the top.
fn find_gltf(dir: &Path) -> Option<PathBuf> {
    let mut level = vec![dir.to_path_buf()];
    for _ in 0..4 {
        let mut below = Vec::new();
        for d in level {
            let Ok(read) = std::fs::read_dir(&d) else {
                continue;
            };
            let mut items: Vec<PathBuf> = read.flatten().map(|e| e.path()).collect();
            items.sort();
            for p in items {
                if p.is_dir() {
                    below.push(p);
                } else if p.extension().is_some_and(|x| {
                    x.eq_ignore_ascii_case("gltf") || x.eq_ignore_ascii_case("glb")
                }) {
                    return Some(p);
                }
            }
        }
        level = below;
    }
    None
}

/// A zip unpacked into `dir` with its folders kept — a glTF names its
/// textures by where they are beside it. A name that would climb out of
/// `dir` is skipped, and what one file unpacks to is capped.
pub fn unzip_tree(zip: &Path, dir: &Path) -> Result<(), String> {
    use std::io::Read;
    let file = std::fs::File::open(zip).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;
    for i in 0..archive.len() {
        let mut item = archive.by_index(i).map_err(|e| e.to_string())?;
        let Some(relative) = item.enclosed_name() else {
            continue;
        };
        let dest = dir.join(relative);
        if item.is_dir() {
            std::fs::create_dir_all(&dest).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut bytes = Vec::new();
        item.by_ref()
            .take(1 << 30)
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        std::fs::write(&dest, bytes).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Words, safe in a URL's query.
fn encode(text: &str) -> String {
    text.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn a_model_reads_with_its_maker_licence_picture_and_movement() {
        let m = json!({
            "uid": "abc123def456",
            "name": "Fox",
            "user": { "username": "maker" },
            "license": { "label": "CC Attribution" },
            "animationCount": 2,
            "tags": [{ "name": "Animal" }],
            "thumbnails": { "images": [
                { "width": 100, "url": "https://media/100.jpg" },
                { "width": 256, "url": "https://media/256.jpg" },
                { "width": 1024, "url": "https://media/1024.jpg" }
            ]}
        });
        let e = entry_of(&m).unwrap();
        assert_eq!(e.name, "Fox");
        assert_eq!(e.authors, "maker");
        assert_eq!(e.licence.as_deref(), Some("CC Attribution"));
        assert!(e.animated);
        assert_eq!(e.tags, vec!["animal"]);
        assert_eq!(e.thumbnail_url.as_deref(), Some("https://media/256.jpg"));
        assert_eq!(e.source, Source::Sketchfab);
        assert!(
            entry_of(&json!({ "uid": "../etc" })).is_none(),
            "a uid becomes a file name"
        );
    }

    #[test]
    fn the_download_prefers_one_file_and_takes_only_https() {
        let both = json!({
            "gltf": { "url": "https://s3/model.zip" },
            "glb": { "url": "https://s3/model.glb" }
        });
        assert_eq!(
            pick_download(&both),
            Some(("https://s3/model.glb".into(), true))
        );
        let zip_only = json!({ "gltf": { "url": "https://s3/model.zip" } });
        assert_eq!(
            pick_download(&zip_only),
            Some(("https://s3/model.zip".into(), false))
        );
        assert_eq!(
            pick_download(&json!({ "gltf": { "url": "http://x" } })),
            None
        );
    }

    #[test]
    fn a_zip_unpacks_with_its_folders_and_none_climb_out() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("kreate-sketchfab-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let zip_path = dir.join("m.zip");
        {
            let mut z = zip::ZipWriter::new(std::fs::File::create(&zip_path).unwrap());
            let o = zip::write::SimpleFileOptions::default();
            z.start_file("scene.gltf", o).unwrap();
            z.write_all(b"{}").unwrap();
            z.start_file("textures/base.png", o).unwrap();
            z.write_all(b"png").unwrap();
            z.start_file("../escape.txt", o).unwrap();
            z.write_all(b"no").unwrap();
            z.finish().unwrap();
        }
        let out = dir.join("out");
        unzip_tree(&zip_path, &out).unwrap();
        assert!(out.join("textures/base.png").is_file());
        assert!(!dir.join("escape.txt").exists());
        assert_eq!(find_gltf(&out), Some(out.join("scene.gltf")));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
