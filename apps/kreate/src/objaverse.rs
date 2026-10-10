//! Objaverse: the eight hundred thousand Creative Commons models Sketchfab's
//! makers allowed to be downloaded, gathered by the Allen Institute and kept
//! on Hugging Face — free to fetch, no account asked.
//!
//! Found through Sketchfab's own public search, which knows names, tags,
//! licences and pictures; kept to the results the dataset holds (its uids
//! are Sketchfab's) and whose licence lets a model be changed and used
//! commercially — Attribution, ShareAlike or CC0, not NonCommercial or
//! NoDerivs. Fetched from the dataset. The dataset's own index of where
//! each model is, twenty megabytes, is fetched once and kept.

use std::path::PathBuf;
use std::sync::OnceLock;

use crate::assets::{cache_root, download, safe_id, Entry, Kind, Query, Source};

const SEARCH: &str = "https://api.sketchfab.com/v3/search";
const MODEL: &str = "https://api.sketchfab.com/v3/models";
const DATASET: &str = "https://huggingface.co/datasets/allenai/objaverse/resolve/main";

/// Where the dataset keeps each model: its uid, as a number, and the shard
/// (`glbs/000-NNN/`) it is in. Sorted by uid, to be searched.
fn index() -> Result<&'static Vec<(u128, u16)>, String> {
    static INDEX: OnceLock<Result<Vec<(u128, u16)>, String>> = OnceLock::new();
    INDEX
        .get_or_init(|| {
            let file = cache_root()
                .ok_or("no cache folder")?
                .join("objaverse")
                .join("object-paths.json.gz");
            if !file.exists() {
                download(&format!("{DATASET}/object-paths.json.gz"), &file)?;
            }
            let reader = flate2::read::GzDecoder::new(
                std::fs::File::open(&file).map_err(|e| e.to_string())?,
            );
            let paths: std::collections::HashMap<String, String> =
                serde_json::from_reader(std::io::BufReader::new(reader))
                    .map_err(|e| format!("the dataset's index: {e}"))?;
            let mut out: Vec<(u128, u16)> = paths
                .iter()
                .filter_map(|(uid, path)| {
                    let shard = path.strip_prefix("glbs/000-")?.get(..3)?.parse().ok()?;
                    Some((u128::from_str_radix(uid, 16).ok()?, shard))
                })
                .collect();
            out.sort_unstable();
            Ok(out)
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// The shard a model is in, if the dataset has it.
fn shard_of(uid: &str) -> Option<u16> {
    let key = u128::from_str_radix(uid, 16).ok()?;
    let index = index().ok()?;
    index
        .binary_search_by_key(&key, |(k, _)| *k)
        .ok()
        .map(|i| index[i].1)
}

/// Whether a licence lets a model be changed and sold: not NonCommercial,
/// not NoDerivs.
pub fn licence_allows(label: &str) -> bool {
    let l = label.to_ascii_lowercase();
    !l.contains("noncommercial") && !l.contains("non-commercial") && !l.contains("noderiv")
}

/// `url` as text.
fn get(url: &str) -> Result<String, String> {
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let out = crate::claude::quiet_command(curl)
        .args([
            "--silent",
            "--show-error",
            "--fail",
            "--location",
            "--max-time",
            "60",
        ])
        .arg(url)
        .output()
        .map_err(|e| format!("cannot start curl: {e}"))?;
    if out.status.success() {
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

/// One Sketchfab model, as an entry — if the dataset has it and its licence
/// allows.
fn entry_of(m: &serde_json::Value) -> Option<Entry> {
    let id = m["uid"].as_str()?.to_string();
    if !safe_id(&id) {
        return None;
    }
    let licence = m["license"]["label"]
        .as_str()
        .or_else(|| m["license"].as_str())?
        .to_string();
    if !licence_allows(&licence) {
        return None;
    }
    let shard = shard_of(&id)?;
    // A picture near the tile's size.
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
        source: Source::Objaverse,
        tags,
        authors: m["user"]["username"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
        thumbnail_url: thumbnail_url.filter(|u| u.starts_with("https://")),
        download_url: Some(format!("{DATASET}/glbs/000-{shard:03}/{id}.glb")),
        licence: Some(licence),
        animated: m["animationCount"].as_u64().unwrap_or(0) > 0,
        id,
        ..Entry::blank()
    })
}

/// What Sketchfab finds for `query`, kept to what the dataset holds and the
/// licences allow: up to three pages of its answer.
pub fn list(kind: Kind, query: &Query) -> Result<Vec<Entry>, String> {
    if kind != Kind::Model {
        return Ok(Vec::new());
    }
    index()?;
    let words = query.text.trim();
    let mut url = format!("{SEARCH}?type=models&downloadable=true&count=24");
    if words.is_empty() {
        url.push_str("&sort_by=-likeCount");
    } else {
        url.push_str(&format!("&q={}", encode(words)));
    }
    if query.animated {
        url.push_str("&animated=true");
    }
    let mut out = Vec::new();
    let mut next = Some(url);
    for _ in 0..3 {
        let Some(page) = next.take() else { break };
        let answer: serde_json::Value =
            serde_json::from_str(&get(&page)?).map_err(|e| format!("Sketchfab's answer: {e}"))?;
        for m in answer["results"].as_array().into_iter().flatten() {
            if let Some(e) = entry_of(m).filter(|e| !query.animated || e.animated) {
                out.push(e);
            }
        }
        next = answer["next"]
            .as_str()
            .filter(|n| n.starts_with(SEARCH))
            .map(str::to_string);
    }
    Ok(out)
}

/// One model by its uid.
pub fn find(uid: &str) -> Result<Entry, String> {
    if !safe_id(uid) {
        return Err(format!("`{uid}` is not a model's uid"));
    }
    index()?;
    let m: serde_json::Value = serde_json::from_str(&get(&format!("{MODEL}/{uid}"))?)
        .map_err(|e| format!("Sketchfab's answer: {e}"))?;
    entry_of(&m).ok_or_else(|| {
        format!("`{uid}` is not in Objaverse, or its licence does not allow changing or selling it")
    })
}

/// A model on the disk, kept for next time.
pub fn fetch(entry: &Entry) -> Result<PathBuf, String> {
    if !safe_id(&entry.id) {
        return Err(format!("`{}` is not a model's uid", entry.id));
    }
    let url = entry
        .download_url
        .as_deref()
        .filter(|u| u.starts_with(DATASET))
        .ok_or("the model has no file in the dataset")?;
    let dest = cache_root()
        .ok_or("no cache folder")?
        .join("objaverse")
        .join("models")
        .join(format!("{}.glb", entry.id));
    if !dest.exists() {
        download(url, &dest)?;
    }
    Ok(dest)
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

    #[test]
    fn licences_that_forbid_changing_or_selling_are_left_out() {
        for ok in [
            "CC Attribution",
            "CC0 Public Domain",
            "CC Attribution-ShareAlike",
        ] {
            assert!(licence_allows(ok), "{ok}");
        }
        for no in [
            "CC Attribution-NonCommercial",
            "CC Attribution-NoDerivs",
            "CC Attribution-NonCommercial-NoDerivs",
        ] {
            assert!(!licence_allows(no), "{no}");
        }
    }

    /// Against the real services: `cargo test -p kreate --lib objaverse -- --ignored`.
    #[test]
    #[ignore = "asks Sketchfab and Hugging Face, over the network"]
    fn people_are_found_and_one_is_fetched() {
        let found = list(
            Kind::Model,
            &Query {
                text: "human".into(),
                animated: false,
            },
        )
        .expect("listed");
        eprintln!("{} found", found.len());
        assert!(found.len() > 5);
        let first = &found[0];
        eprintln!("{} by {} ({:?})", first.name, first.authors, first.licence);
        let path = fetch(first).expect("fetched");
        assert!(std::fs::metadata(&path).expect("there").len() > 1000);
    }
}
