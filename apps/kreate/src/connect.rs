//! Help ▸ Connect to AI Apps: Kreate's connector added to the assistants
//! the user already has — Claude Desktop, Claude Code, Cursor, VS Code, the
//! Gemini CLI and Codex — so any of them can work on the open document.
//!
//! Each keeps its MCP servers in a settings file of its own. Kreate adds one
//! entry, `kreate`, pointing at the `kreate-mcp` beside it, and touches
//! nothing else: the file is read, the entry set, and the file written back
//! through a temporary one. The first time a file is changed a copy of it
//! is kept beside it. A file that does not parse is refused, not replaced —
//! it is the user's, and an assistant's other servers are in it.
//!
//! Claude Code's file is written by Claude Code while it runs, so it is
//! asked to add the entry itself: `claude mcp add`.

use std::path::{Path, PathBuf};

/// The assistants Kreate can add itself to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum App {
    ClaudeDesktop,
    ClaudeCode,
    Cursor,
    VsCode,
    Gemini,
    Codex,
}

/// Where each one stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    /// Not on this computer, as far as can be told.
    Absent,
    /// There, without Kreate.
    Ready,
    /// There, with Kreate's entry pointing at this Kreate.
    Connected,
}

impl App {
    pub const ALL: [App; 6] = [
        App::ClaudeDesktop,
        App::ClaudeCode,
        App::Cursor,
        App::VsCode,
        App::Gemini,
        App::Codex,
    ];

    /// Its own name.
    pub fn name(self) -> &'static str {
        match self {
            App::ClaudeDesktop => "Claude Desktop",
            App::ClaudeCode => "Claude Code",
            App::Cursor => "Cursor",
            App::VsCode => "VS Code",
            App::Gemini => "Gemini CLI",
            App::Codex => "Codex CLI",
        }
    }

    /// The settings file it reads its servers from, and the folder whose
    /// being there says it is installed.
    fn place(self) -> Option<(PathBuf, PathBuf)> {
        let home =
            PathBuf::from(std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE"))?);
        let support = if cfg!(target_os = "macos") {
            home.join("Library/Application Support")
        } else if cfg!(windows) {
            PathBuf::from(std::env::var_os("APPDATA")?)
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| home.join(".config"))
        };
        Some(match self {
            App::ClaudeDesktop => {
                let dir = support.join("Claude");
                (dir.join("claude_desktop_config.json"), dir)
            }
            App::ClaudeCode => (home.join(".claude.json"), home.join(".claude")),
            App::Cursor => {
                let dir = home.join(".cursor");
                (dir.join("mcp.json"), dir)
            }
            App::VsCode => {
                let dir = support.join("Code").join("User");
                (dir.join("mcp.json"), dir)
            }
            App::Gemini => {
                let dir = home.join(".gemini");
                (dir.join("settings.json"), dir)
            }
            App::Codex => {
                let dir = home.join(".codex");
                (dir.join("config.toml"), dir)
            }
        })
    }

    /// Where it stands with this Kreate's connector.
    pub fn status(self, connector: &Path) -> Status {
        let Some((file, dir)) = self.place() else {
            return Status::Absent;
        };
        let installed = match self {
            App::ClaudeCode => crate::claude::find_cli(crate::claude::Provider::Claude).is_some(),
            _ => dir.is_dir(),
        };
        if !installed {
            return Status::Absent;
        }
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        if names_connector(self, &text, connector) {
            Status::Connected
        } else {
            Status::Ready
        }
    }

    /// Adds Kreate, or points its entry at this Kreate. Says what to do next.
    pub fn connect(self, connector: &Path) -> Result<String, String> {
        if self == App::ClaudeCode {
            return claude_code(connector);
        }
        let (file, dir) = self.place().ok_or("no home folder")?;
        if !dir.is_dir() {
            return Err(format!("{} is not installed here", self.name()));
        }
        let old = match std::fs::read_to_string(&file) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => String::new(),
            Err(e) => return Err(format!("cannot read {}: {e}", file.display())),
        };
        let new = with_connector(self, &old, connector)?;
        write_keeping_a_copy(&file, &old, &new)?;
        Ok(format!(
            "Kreate is added to {} ({}). Restart it to see Kreate's tools.",
            self.name(),
            file.display()
        ))
    }
}

/// Whether `text`, an app's settings, has Kreate's entry naming `connector`.
fn names_connector(app: App, text: &str, connector: &Path) -> bool {
    let wanted = connector.to_string_lossy();
    if app == App::Codex {
        return section(text, "mcp_servers.kreate")
            .is_some_and(|s| s.contains(&*toml_string(&wanted)));
    }
    let Ok(value) = serde_json::from_str::<serde_json::Value>(text) else {
        return false;
    };
    let key = if app == App::VsCode {
        "servers"
    } else {
        "mcpServers"
    };
    value[key]["kreate"]["command"].as_str() == Some(&*wanted)
}

/// `text` with Kreate's entry set, everything else as it was.
fn with_connector(app: App, text: &str, connector: &Path) -> Result<String, String> {
    let command = connector.to_string_lossy().into_owned();
    if app == App::Codex {
        let block = format!(
            "[mcp_servers.kreate]\ncommand = {}\nargs = []\n",
            toml_string(&command)
        );
        return Ok(match section(text, "mcp_servers.kreate") {
            Some(old) => text.replacen(old, &block, 1),
            None if text.trim().is_empty() => block,
            None => format!("{}\n\n{block}", text.trim_end()),
        });
    }
    let mut value: serde_json::Value = if text.trim().is_empty() {
        serde_json::json!({})
    } else {
        serde_json::from_str(text).map_err(|e| {
            format!("its settings are not JSON Kreate can read, so they were left alone: {e}")
        })?
    };
    let root = value
        .as_object_mut()
        .ok_or("its settings are not an object, so they were left alone")?;
    let (key, entry) = if app == App::VsCode {
        (
            "servers",
            serde_json::json!({ "type": "stdio", "command": command, "args": [] }),
        )
    } else {
        (
            "mcpServers",
            serde_json::json!({ "command": command, "args": [] }),
        )
    };
    let servers = root
        .entry(key)
        .or_insert_with(|| serde_json::json!({}))
        .as_object_mut()
        .ok_or_else(|| format!("its `{key}` is not an object, so it was left alone"))?;
    servers.insert("kreate".into(), entry);
    serde_json::to_string_pretty(&value).map_err(|e| e.to_string())
}

/// A TOML table's text, from its header to the next one.
fn section<'a>(text: &'a str, table: &str) -> Option<&'a str> {
    let header = format!("[{table}]");
    let start = text.find(&header)?;
    let rest = &text[start + header.len()..];
    let end = rest
        .find("\n[")
        .map(|i| start + header.len() + i + 1)
        .unwrap_or(text.len());
    Some(&text[start..end])
}

/// `s` as a TOML string.
fn toml_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Writes `new` over `file` through a temporary file, keeping the first
/// version Kreate changed beside it.
fn write_keeping_a_copy(file: &Path, old: &str, new: &str) -> Result<(), String> {
    if !old.is_empty() {
        let copy = file.with_extension(format!(
            "{}.before-kreate",
            file.extension().and_then(|e| e.to_str()).unwrap_or("")
        ));
        if !copy.exists() {
            std::fs::write(&copy, old).map_err(|e| e.to_string())?;
        }
    }
    let temporary = file.with_extension("kreate-writing");
    std::fs::write(&temporary, new).map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, file).map_err(|e| e.to_string())
}

/// Claude Code adds the entry itself, for the user wherever they work.
fn claude_code(connector: &Path) -> Result<String, String> {
    let claude = crate::claude::find_cli(crate::claude::Provider::Claude)
        .ok_or("Claude Code is not installed here")?;
    // An old entry first, so the new one is this Kreate's.
    let _ = crate::claude::quiet_command(&claude)
        .args(["mcp", "remove", "--scope", "user", "kreate"])
        .output();
    let out = crate::claude::quiet_command(&claude)
        .args(["mcp", "add", "--scope", "user", "kreate", "--"])
        .arg(connector)
        .output()
        .map_err(|e| format!("cannot start Claude Code: {e}"))?;
    if out.status.success() {
        Ok(
            "Kreate is added to Claude Code, for every folder. Start a new session to use it."
                .into(),
        )
    } else {
        Err(String::from_utf8_lossy(&out.stderr).trim().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONNECTOR: &str = "/Applications/Kreate.app/Contents/MacOS/kreate-mcp";

    #[test]
    fn an_entry_joins_the_servers_already_there() {
        let old = r#"{ "mcpServers": { "files": { "command": "files-mcp" } }, "theme": "dark" }"#;
        let new = with_connector(App::ClaudeDesktop, old, Path::new(CONNECTOR)).expect("set");
        let value: serde_json::Value = serde_json::from_str(&new).expect("json");
        assert_eq!(value["mcpServers"]["files"]["command"], "files-mcp");
        assert_eq!(value["mcpServers"]["kreate"]["command"], CONNECTOR);
        assert_eq!(value["theme"], "dark");
        assert!(names_connector(
            App::ClaudeDesktop,
            &new,
            Path::new(CONNECTOR)
        ));
        assert!(!names_connector(
            App::ClaudeDesktop,
            old,
            Path::new(CONNECTOR)
        ));
        // Set again, pointing elsewhere: replaced, not doubled.
        let moved = with_connector(App::ClaudeDesktop, &new, Path::new("/elsewhere/kreate-mcp"))
            .expect("set");
        assert!(!names_connector(
            App::ClaudeDesktop,
            &moved,
            Path::new(CONNECTOR)
        ));
    }

    #[test]
    fn vs_code_has_its_own_shape_and_an_empty_file_is_a_new_one() {
        let new = with_connector(App::VsCode, "", Path::new(CONNECTOR)).expect("set");
        let value: serde_json::Value = serde_json::from_str(&new).expect("json");
        assert_eq!(value["servers"]["kreate"]["type"], "stdio");
        assert_eq!(value["servers"]["kreate"]["command"], CONNECTOR);
    }

    #[test]
    fn a_file_that_does_not_parse_is_refused_not_replaced() {
        let broken = r#"{ "mcpServers": { "files": "#;
        assert!(with_connector(App::Cursor, broken, Path::new(CONNECTOR)).is_err());
        let wrong = r#"{ "mcpServers": [] }"#;
        assert!(with_connector(App::Cursor, wrong, Path::new(CONNECTOR)).is_err());
    }

    #[test]
    fn codex_gets_a_table_of_its_own_and_keeps_the_rest() {
        let old = "model = \"o4\"\n\n[mcp_servers.files]\ncommand = \"files-mcp\"\n";
        let new = with_connector(App::Codex, old, Path::new(CONNECTOR)).expect("set");
        assert!(new.contains("model = \"o4\"") && new.contains("[mcp_servers.files]"));
        assert!(names_connector(App::Codex, &new, Path::new(CONNECTOR)));
        // Again, elsewhere: the one table, changed.
        let moved = with_connector(App::Codex, &new, Path::new("/other/kreate-mcp")).expect("set");
        assert_eq!(moved.matches("[mcp_servers.kreate]").count(), 1);
        assert!(moved.contains("/other/kreate-mcp") && !moved.contains(CONNECTOR));
        assert!(moved.contains("[mcp_servers.files]"));
    }

    #[test]
    fn a_changed_file_keeps_its_first_version_beside_it() {
        let dir = std::env::temp_dir().join(format!("kreate-connect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("made");
        let file = dir.join("mcp.json");
        std::fs::write(&file, "{}").expect("written");
        write_keeping_a_copy(&file, "{}", "{\"a\":1}").expect("first");
        write_keeping_a_copy(&file, "{\"a\":1}", "{\"a\":2}").expect("second");
        let copy = dir.join("mcp.json.before-kreate");
        assert_eq!(std::fs::read_to_string(copy).expect("copy"), "{}");
        assert_eq!(std::fs::read_to_string(&file).expect("file"), "{\"a\":2}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
