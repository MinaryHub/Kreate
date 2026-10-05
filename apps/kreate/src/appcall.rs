//! Script lines the application answers itself.
//!
//! Most of a script is operator calls, which the history runs. A few things a
//! script needs are not operators, because they are not edits of the
//! document: which mode the window is in, and reading an image file off the
//! disk. Without them a script cannot reach any `mesh.*` or `uv.*` operator,
//! which all work in edit mode, and cannot give a material a texture.
//!
//! They are written like operator calls so a script reads as one language:
//!
//! ```text
//! kreate.ops.object.mode_set(mode="edit")
//! kreate.ops.image.open(path="/textures/oak.jpg", color=True)
//! ```

/// One line the application runs.
#[derive(Clone, Debug, PartialEq)]
pub enum AppCall {
    /// Switch mode: `object`, `edit`, `sculpt`, `weight_paint` or `pose`.
    /// `edit` means the edit mode of whatever the active object is.
    ModeSet(String),
    /// Read an image file into the document. `color` is false for numbers
    /// that are not colour: a normal map, a roughness.
    OpenImage { path: String, color: bool },
}

/// What [`AppCall`]s look like, for the operator listing.
pub const LISTING: [(&str, &str, &[(&str, &str)]); 2] = [
    (
        "object.mode_set",
        "Switch the editing mode. mesh.* and uv.* operators need edit mode; leave it with mode=\"object\".",
        &[("mode", "object | edit | sculpt | weight_paint | pose")],
    ),
    (
        "image.open",
        "Read an image file into the document. The result is its uid, for material.set_image, node.set_image or world.set_image.",
        &[
            ("path", "the file, absolute"),
            ("color", "True for colour (albedo, emission, sky); False for data (normal, roughness)"),
        ],
    ),
];

/// The call on `line`, if it is one of these; `Some(Err)` if it is one but
/// cannot be read.
pub fn parse(line: &str) -> Option<Result<AppCall, String>> {
    let line = line.trim();
    let call = line.strip_prefix("kreate.ops.").unwrap_or(line);
    let (name, rest) = call.split_once('(')?;
    let name = name.trim();
    if !matches!(name, "object.mode_set" | "image.open") {
        return None;
    }
    Some(read(name, rest))
}

fn read(name: &str, rest: &str) -> Result<AppCall, String> {
    let inner = rest
        .trim_end()
        .strip_suffix(')')
        .ok_or_else(|| format!("{name}: missing `)`"))?;
    let args = arguments(inner).map_err(|e| format!("{name}: {e}"))?;
    let get = |key: &str| args.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str());
    for (key, _) in &args {
        let known = match name {
            "object.mode_set" => key == "mode",
            _ => key == "path" || key == "color",
        };
        if !known {
            return Err(format!("{name} has no parameter `{key}`"));
        }
    }
    match name {
        "object.mode_set" => {
            let mode = get("mode").ok_or("object.mode_set: `mode` is required")?;
            Ok(AppCall::ModeSet(mode.to_string()))
        }
        _ => {
            let path = get("path").ok_or("image.open: `path` is required")?;
            let color = match get("color") {
                None | Some("True") | Some("true") => true,
                Some("False") | Some("false") => false,
                Some(other) => return Err(format!("image.open: `color` is True or False, not {other}")),
            };
            Ok(AppCall::OpenImage {
                path: path.to_string(),
                color,
            })
        }
    }
}

/// `key=value` pairs, separated by commas; a value is a quoted string (with
/// `\"` and `\\` inside) or a bare word.
fn arguments(text: &str) -> Result<Vec<(String, String)>, String> {
    let mut out = Vec::new();
    let mut chars = text.chars().peekable();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace() || *c == ',') {
            chars.next();
        }
        if chars.peek().is_none() {
            return Ok(out);
        }
        let key: String = chars.by_ref().take_while(|c| *c != '=').collect();
        let key = key.trim().to_string();
        if key.is_empty() {
            return Err("expected name=value".into());
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let value = if chars.peek() == Some(&'"') {
            chars.next();
            let mut value = String::new();
            loop {
                match chars.next() {
                    Some('"') => break,
                    Some('\\') => match chars.next() {
                        Some('n') => value.push('\n'),
                        Some(c) => value.push(c),
                        None => return Err("unfinished string".into()),
                    },
                    Some(c) => value.push(c),
                    None => return Err("unfinished string".into()),
                }
            }
            value
        } else {
            let value: String = chars.by_ref().take_while(|c| *c != ',').collect();
            value.trim().to_string()
        };
        out.push((key, value));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_lines_are_read() {
        assert_eq!(
            parse("kreate.ops.object.mode_set(mode=\"edit\")"),
            Some(Ok(AppCall::ModeSet("edit".into())))
        );
        assert_eq!(parse("object.mode_set( mode = \"object\" )"), Some(Ok(AppCall::ModeSet("object".into()))));
    }

    #[test]
    fn image_lines_are_read() {
        assert_eq!(
            parse(r#"kreate.ops.image.open(path="C:\\tex\\oak, dark.jpg", color=False)"#),
            Some(Ok(AppCall::OpenImage {
                path: r"C:\tex\oak, dark.jpg".into(),
                color: false,
            }))
        );
        assert_eq!(
            parse(r#"image.open(path="/a.png")"#),
            Some(Ok(AppCall::OpenImage {
                path: "/a.png".into(),
                color: true
            }))
        );
    }

    #[test]
    fn operators_are_left_to_the_history() {
        assert_eq!(parse("kreate.ops.object.add_primitive(shape=\"cube\")"), None);
        assert_eq!(parse("object.mode_setting(mode=\"edit\")"), None);
    }

    #[test]
    fn mistakes_are_reported() {
        assert!(matches!(parse("object.mode_set()"), Some(Err(_))));
        assert!(matches!(parse("object.mode_set(mod=\"edit\")"), Some(Err(_))));
        assert!(matches!(parse("image.open(path=\"/a.png\""), Some(Err(_))));
        assert!(matches!(parse("image.open(path=\"/a.png)"), Some(Err(_))));
    }
}
