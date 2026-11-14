//! The assistants' API keys, for whoever would rather pay by the call than
//! sign the assistant's program in to an account: Anthropic's for Claude,
//! Google's for Gemini, OpenAI's for ChatGPT.
//!
//! Each is kept in the settings folder, `ai-keys/<assistant>`, readable only
//! by the user (0600, in a folder that is 0700), and never shown again
//! whole — the panel says which ones are held by their last four characters.
//! It reaches the assistant's program in the environment variable that
//! program reads, when Kreate starts it for a message (see
//! `crate::claude`); a program started from a Finder window never sees the
//! shell's own variables, which is why the keys are kept here at all.

use std::path::PathBuf;

/// An assistant that takes a key: the name of the file it is kept in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Service {
    Anthropic,
    Google,
    OpenAi,
}

impl Service {
    pub const ALL: [Service; 3] = [Service::Anthropic, Service::Google, Service::OpenAi];

    fn file(self) -> &'static str {
        match self {
            Service::Anthropic => "anthropic",
            Service::Google => "gemini",
            Service::OpenAi => "openai",
        }
    }

    /// Who the key is from and what it is for, as the panel names it.
    pub fn label(self) -> &'static str {
        match self {
            Service::Anthropic => "Claude (Anthropic)",
            Service::Google => "Gemini (Google)",
            Service::OpenAi => "ChatGPT (OpenAI)",
        }
    }

    /// Where a key is made.
    pub fn console(self) -> &'static str {
        match self {
            Service::Anthropic => "https://console.anthropic.com/settings/keys",
            Service::Google => "https://aistudio.google.com/apikey",
            Service::OpenAi => "https://platform.openai.com/api-keys",
        }
    }

    /// The environment variables the assistant's program reads it from.
    /// Codex has read both, in its different versions.
    pub fn variables(self) -> &'static [&'static str] {
        match self {
            Service::Anthropic => &["ANTHROPIC_API_KEY"],
            Service::Google => &["GEMINI_API_KEY"],
            Service::OpenAi => &["OPENAI_API_KEY", "CODEX_API_KEY"],
        }
    }
}

fn path(service: Service) -> Option<PathBuf> {
    crate::prefs::config_dir().map(|d| d.join("ai-keys").join(service.file()))
}

/// The key held for `service`, if one is.
pub fn key(service: Service) -> Option<String> {
    let key = std::fs::read_to_string(path(service)?).ok()?;
    let key = key.trim().to_string();
    (!key.is_empty()).then_some(key)
}

/// The held key's last four characters, for the panel to say which it is.
pub fn tail(service: Service) -> Option<String> {
    let key = key(service)?;
    let chars: Vec<char> = key.chars().collect();
    Some(chars[chars.len().saturating_sub(4)..].iter().collect())
}

/// Keeps `key` for `service`, or forgets the one held when `key` is empty.
pub fn save(service: Service, key: &str) -> Result<(), String> {
    let path = path(service).ok_or("no settings folder")?;
    let key = key.trim();
    if key.is_empty() {
        match std::fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e.to_string()),
            _ => return Ok(()),
        }
    }
    // A key is one word of printable characters: what was pasted with a
    // space or a line in it is something else.
    if key.len() < 16 || !key.chars().all(|c| c.is_ascii_graphic()) {
        return Err("that does not look like an API key".into());
    }
    let dir = path.parent().ok_or("no settings folder")?;
    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let _ = std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700));
        // Created 0600, so it is never readable by others even for a moment.
        let _ = std::fs::remove_file(&path);
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true).mode(0o600);
        let mut file = options.open(&path).map_err(|e| e.to_string())?;
        std::io::Write::write_all(&mut file, key.as_bytes()).map_err(|e| e.to_string())?;
    }
    #[cfg(not(unix))]
    std::fs::write(&path, key).map_err(|e| e.to_string())?;
    Ok(())
}

/// The variables to start the program for `service` with, when a key is
/// held.
pub fn environment(service: Service) -> Vec<(&'static str, String)> {
    match key(service) {
        Some(k) => service
            .variables()
            .iter()
            .map(|v| (*v, k.clone()))
            .collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Kept, said by its tail, private to the user, handed on in the
    /// program's variables; forgotten when saved empty. Nothing that is not
    /// a key is kept.
    #[test]
    fn a_key_is_kept_privately_and_forgotten() {
        crate::prefs::keep_to_a_test_folder();
        let s = Service::OpenAi;
        save(s, "").unwrap();
        assert_eq!(key(s), None);
        assert!(save(s, "not a key").is_err());
        assert!(save(s, "short").is_err());
        save(s, "  sk-test-0123456789abcdWXYZ\n").unwrap();
        assert_eq!(key(s).as_deref(), Some("sk-test-0123456789abcdWXYZ"));
        assert_eq!(tail(s).as_deref(), Some("WXYZ"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(path(s).unwrap())
                .unwrap()
                .permissions()
                .mode();
            assert_eq!(mode & 0o777, 0o600);
        }
        let env = environment(s);
        assert_eq!(env.len(), 2);
        assert!(env.iter().all(|(_, v)| v == "sk-test-0123456789abcdWXYZ"));
        save(s, "").unwrap();
        assert_eq!(key(s), None);
        assert!(environment(s).is_empty());
    }
}
